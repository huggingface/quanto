import ast
from copy import copy

import torch
from torch.autograd import Function
from torch.utils import _pytree as pytree

from .core import group
from .packed import PackedTensor
from .qtensor import QTensor
from .qtype import qint2, qint4, qint8, qtype, qtypes


__all__ = ["QBitsTensor"]


class AffineQuantizer(Function):
    """A standard affine quantizer."""

    @staticmethod
    def forward(ctx, base: torch.Tensor, qtype: qtype, axis: int, group_size: int):
        if qtype not in (qint2, qint4):
            raise ValueError("QBitsTensor can only be of qint2 or qint4 qtype")
        if axis not in (0, -1):
            raise ValueError("QBitsTensor axis parameter must be 0 (first axis) or -1 (last axis)")
        size = base.size()
        stride = base.stride()
        if group_size is not None:
            base = group(base, axis=axis, group_size=group_size)
        bits = qtype.bits
        dim = list(range(1, base.ndim)) if (axis == 0) else list(range(0, base.ndim - 1))
        rmin = torch.amin(base, dim=dim, keepdim=True)
        rmax = torch.amax(base, dim=dim, keepdim=True)
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        scale = (rmax - rmin) / (qmax - qmin)
        zeropoint = torch.round(-rmin / scale).to(torch.int8)
        data = torch.clamp(torch.round((base - rmin) / scale), min=0, max=2**bits - 1).to(torch.uint8)
        packed = PackedTensor.pack(data, bits)

        return QBitsTensor(qtype, axis, size, stride, packed, scale, zeropoint)

    @staticmethod
    def backward(ctx, gO):
        # For autograd, quantization is a no-op
        return gO, None, None, None


class QBitsToQTensor(Function):
    @staticmethod
    def forward(ctx, t):
        unpacked = t._data.unpack()
        int8_data = unpacked.to(torch.int8) - t._zeropoint.to(torch.int8)
        return QTensor(qint8, t.axis, t.size(), t.stride(), int8_data, t._scale)

    @staticmethod
    def backward(ctx, gO):
        return gO


class QBitsTensor(QTensor):
    @staticmethod
    def __new__(cls, qtype, axis, size, stride, data, scale, zeropoint, requires_grad=False):
        assert isinstance(data, PackedTensor)
        assert data.device == scale.device
        assert data.device == zeropoint.device
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, axis, size, stride, data, scale, zeropoint, requires_grad=False):
        super().__init__(qtype, axis, size, stride, data, scale, requires_grad=requires_grad)
        self._zeropoint = zeropoint

    def __repr__(self):
        autograd_info = (
            f", grad_fn={self.grad_fn}" if self.grad_fn else ", requires_grad=True" if self.requires_grad else ""
        )
        return f"QBitsTensor({self._data}, scale={self._scale}, zeropoint={self._zeropoint}, dtype={self.dtype}{autograd_info})"

    @classmethod
    def quantize(cls, base, qtype=qint4, axis=0, group_size=None):
        """Differentiable quantization function"""
        if axis not in (0, -1):
            raise ValueError("QBitsTensor axis parameter must be 0 (first axis) or -1 (last axis)")
        return AffineQuantizer.apply(base, qtype, axis, group_size)

    def qtensor(self):
        return QBitsToQTensor.apply(self)

    def dequantize(self):
        return self.qtensor().dequantize()

    @property
    def qtype(self):
        return self._qtype

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale", "_zeropoint"]
        # Since meta can be used for serialization, use only strings
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 3
        assert len(meta) == 4
        data, scale, zeropoint = inner_tensors["_data"], inner_tensors["_scale"], inner_tensors["_zeropoint"]
        # Meta should only contain strings, AST compatible except qtype
        qtype = qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return QBitsTensor(qtype, axis, size, stride, data, scale, zeropoint)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        if op.overloadpacket is torch.ops.aten.detach:
            # Detach is required when copying and deserializing
            t = args[0]
            data = op(t._data)
            scale = op(t._scale)
            zeropoint = op(t._zeropoint)
            return QBitsTensor(t._qtype, t._axis, t.size(), t.stride(), data, scale, zeropoint)
        elif op.overloadpacket is torch.ops.aten._to_copy:
            t = args[0]
            # Copy scale
            scale = op(t._scale, **kwargs)
            # Move data and zeropoint, ignoring dtype (it only applies to scale)
            data_kwargs = copy(kwargs)
            data_kwargs["dtype"] = torch.uint8
            data = op(t._data, **data_kwargs)
            zeropoint_kwargs = copy(kwargs)
            zeropoint_kwargs["dtype"] = torch.int8
            zeropoint = op(t._zeropoint, **data_kwargs)
            return QBitsTensor(t._qtype, t._axis, t.size(), t.stride(), data, scale, zeropoint)
        elif op.overloadpacket is torch.ops.aten.mm:
            input = args[0]
            t = args[1]
            scale = t._scale
            # TODO: deal with zero points?
            return torch.ops.quanto.udqmm(input, t, scale, t._bits)
        args, kwargs = pytree.tree_map_only(QBitsTensor, lambda x: x.qtensor(), (args, kwargs or {}))
        return op(*args, **kwargs)

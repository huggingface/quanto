from abc import ABC
from inspect import signature
from typing import Optional, Union

import torch

from ..tensor import Optimizer, QBitsTensor, QTensor, qint2, qint4, qtype, qtypes


__all__ = ["QModuleMixin", "register_qmodule", "quantize_module"]


_QMODULE_TABLE = {}


def register_qmodule(module_cls):
    """
    Used for registering a new quantized module.

    The QModule must implement two abstract methods:

    - qcreate: class method to instantiate a new QModule from an nn.Module, without copying its weights,
    - qforward: instance method for quantized inference.

    The code to register a new module looks like:

    ```
    @register_qmodule(<base torch.nn.Module>)
    class MyQModule(QModuleMixin, <base torch.nn.Module>):
        <implementation>

        @classmethod
        def qcreate(cls,
                    module: torch.nn.Module,
                    weights: Optional[qtype],
                    activations: Optional[qtype] = None):
            ...

        def qforward(self, input: torch.Tensor) -> torch.Tensor:
            ...
    ```

    """

    def wrapper(cls):
        # Inspect the signature of the QModule creation method
        sig = signature(cls.from_module)
        cls_params = []
        for i, param in enumerate(sig.parameters.values()):
            if i == 0:
                # Skip the first parameter which is the module
                continue
            if param.kind == param.POSITIONAL_ONLY:
                raise ValueError(f"{cls}.from_module() can only have a single positional parameter")
            cls_params.append(param.name)
        _QMODULE_TABLE[module_cls] = [cls, cls_params]
        return cls

    return wrapper


def quantize_module(module, **kwargs):
    for cls in _QMODULE_TABLE:
        if isinstance(module, cls):
            qcls, qparams = _QMODULE_TABLE[cls]
            module_kwargs = {}
            for name in qparams:
                if name in kwargs:
                    module_kwargs[name] = kwargs[name]
            return qcls.from_module(module, **module_kwargs)
    return None


class QModuleMixin(ABC):
    def __init__(
        self,
        *args,
        weights: Optional[Union[qtype, str]] = None,
        activations: Optional[Union[qtype, str]] = None,
        optimizer: Optional[Optimizer] = None,
        **kwargs,
    ):
        # The tests below are meant to help people writing their own quantized Module class
        mro = self.__class__.__mro__
        if torch.nn.Module not in mro:
            raise TypeError("Quantized modules must inherit from a torch.nn.Module class")
        if mro.index(__class__) > mro.index(torch.nn.Module):
            raise TypeError(
                "QModuleMixin must be placed before any torch.nn.Module class in quantized module inheritance."
            )
        # This will setup the torch.nn.Module
        super().__init__(*args, **kwargs)
        if weights is not None and not isinstance(weights, qtype):
            weights = qtypes[weights]
        if activations is not None and not isinstance(activations, qtype):
            activations = qtypes[activations]
        self.weight_qtype = weights
        self.weight_group_size = None
        if self.weight_qtype in (qint2, qint4):
            out_features = self.weight.shape[0]
            if out_features >= 128:
                group_size = self.weight.numel() // out_features
                while group_size >= 128 and group_size % 2 == 0:
                    group_size = group_size // 2
                self.weight_group_size = group_size
        self.activation_qtype = activations
        self.optimizer = optimizer
        self.register_buffer("input_scale", torch.ones(()))
        self.register_buffer("output_scale", torch.ones(()))

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self.weight_qtype is None or not self.frozen:
            # Save standard weight Tensor
            destination[prefix + "weight"] = self.weight if keep_vars else self.weight.detach()
        else:

            def serialize_tensor_subclass(t, destination, prefix, keep_vars):
                inner_tensors, meta = t.__tensor_flatten__()
                for name in inner_tensors:
                    inner_tensor = getattr(t, name)
                    if type(inner_tensor) == torch.Tensor:
                        # Leaf Tensor, we can serialize it
                        destination[prefix + name] = inner_tensor if keep_vars else inner_tensor.detach()
                    else:
                        # Flatten also this inner Tensor
                        serialize_tensor_subclass(inner_tensor, destination, prefix + name + ".", keep_vars)
                for name, value in meta.items():
                    destination[prefix + name] = value

            # Frozen module: flatten QTensor weight into individual tensors
            serialize_tensor_subclass(self.weight, destination, prefix + "weight.", keep_vars)
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()
        destination[prefix + "input_scale"] = self.input_scale if keep_vars else self.input_scale.detach()
        destination[prefix + "output_scale"] = self.output_scale if keep_vars else self.output_scale.detach()
        destination[prefix + "weight_qtype"] = "none" if self.weight_qtype is None else self.weight_qtype.name
        destination[prefix + "activation_qtype"] = (
            "none" if self.activation_qtype is None else self.activation_qtype.name
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        weight_qtype = state_dict.pop(prefix + "weight_qtype")
        self.weight_qtype = None if weight_qtype == "none" else qtypes[weight_qtype]
        activation_qtype = state_dict.pop(prefix + "activation_qtype")
        self.activation_qtype = None if activation_qtype == "none" else qtypes[activation_qtype]

        weight_name = prefix + "weight"
        if self.weight_qtype is not None and weight_name not in state_dict:
            # The weight Tensor is not present because it is a flattened QTensor
            def deserialize_tensor_subclass(t, state_dict, prefix):
                inner_tensors, meta = t.__tensor_flatten__()
                inner_tensors_dict = {}
                meta_dict = {}
                for name in inner_tensors:
                    if (prefix + name) in state_dict:
                        # Leaf Tensor, we can deserialize it
                        inner_tensors_dict[name] = state_dict.pop(prefix + name)
                    else:
                        # Flattened inner Tensor
                        inner_tensor = getattr(t, name)
                        inner_tensors_dict[name] = deserialize_tensor_subclass(
                            inner_tensor, state_dict, prefix + name + "."
                        )
                for name in meta:
                    meta_dict[name] = state_dict.pop(prefix + name)
                return t.__class__.__tensor_unflatten__(inner_tensors_dict, meta_dict, None, None)

            deserialized_weight = deserialize_tensor_subclass(self.qweight, state_dict, weight_name + ".")
            device = self.weight.device if self.weight.device.type != "meta" else deserialized_weight.device
            assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
            if assign_to_params_buffers:
                self.weight = torch.nn.Parameter(deserialized_weight)
            else:
                if type(self.weight.data) != type(deserialized_weight):
                    # Reloading frozen weights into unfrozen module: move to the correct device and force assignment
                    self.weight = torch.nn.Parameter(deserialized_weight.to(device))
                else:
                    # FIXME: here we should copy frozen weights into frozen module, but this leads to grad error
                    self.weight = torch.nn.Parameter(deserialized_weight.to(device))
        # this is needed because we can't load it correctly when the bias is on the meta device
        if prefix + "bias" in state_dict:
            self.bias = torch.nn.Parameter(state_dict.pop(prefix + "bias"))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs
        )

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        weights: Optional[qtype] = None,
        activations: Optional[qtype] = None,
        optimizer: Optional[Optimizer] = None,
    ):
        qmodule = cls.qcreate(module, weights, activations, optimizer)
        if qmodule is None:
            return None
        with torch.no_grad():
            qmodule.weight.copy_(module.weight)
            if module.bias is not None:
                qmodule.bias.copy_(module.bias)
        return qmodule.to(module.weight.device)

    @classmethod
    def qcreate(cls, module: torch.nn.Module, weights: Optional[qtype], activations: Optional[qtype] = None):
        raise NotImplementedError

    @property
    def qweight(self):
        """Return the module quantized weight

        When the module is frozen or does not quantize its weight parameter, it simply
        returns the weight.
        When the module is not frozen, this property is required to add the dynamic quantization
        of the weight parameter to the graph and allow gradients to be propagated to the
        underlying weight float values.
        """
        if self.weight_qtype is None:
            # QModule that does not quantize its weights
            return None
        if isinstance(self.weight, QTensor):
            # Frozen QModule
            return self.weight
        # Quantize dynamically the weights per-axis
        if self.weight_qtype in (qint2, qint4):
            return QBitsTensor.quantize(
                self.weight,
                qtype=self.weight_qtype,
                axis=0,
                group_size=self.weight_group_size,
                optimizer=self.optimizer,
            )
        elif isinstance(self.weight_qtype, qtype):
            return QTensor.quantize(
                self.weight, qtype=self.weight_qtype, axis=0, group_size=None, optimizer=self.optimizer
            )
        raise ValueError(f"Invalid quantized weights type {self.weight_qtype}")

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        def maybe_requantize(t, scale):
            if t.qtype == self.activation_qtype and t.axis is None:
                return t
            return QTensor.quantize(
                t.dequantize(), qtype=self.activation_qtype, axis=None, group_size=None, scale=scale
            )

        if self.activation_qtype is not None and isinstance(input, QTensor):
            input = maybe_requantize(input, self.input_scale)
        output = self.qforward(input)
        if self.activation_qtype is not None:
            if isinstance(output, QTensor):
                output = maybe_requantize(output, self.output_scale)
            else:
                output = QTensor.quantize(
                    output, qtype=self.activation_qtype, axis=None, group_size=None, scale=self.output_scale
                )
        return output

    def freeze(self):
        qweight = self.qweight
        if qweight is not None:
            # Replace float weights by quantized weights
            self.weight = torch.nn.Parameter(qweight)

    @property
    def frozen(self):
        return isinstance(self.weight, QTensor)

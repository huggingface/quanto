import warnings
from contextlib import contextmanager

import torch


# This file contains the definitions of all operations under torch.ops.quanto


_ext_enabled = True


@contextmanager
def disable_extensions():
    """Disable quanto extensions (debug)"""
    try:
        global _ext_enabled
        _ext_enabled = False
        yield
    finally:
        _ext_enabled = True


def define(name, schema):
    """Define a new quanto operation.

    The operation will actually be defined in three libraries:
    - the top-level quanto library as quanto::<op>,
    - the quanto python library as quanto_py::<op>,
    - the quanto extension library as quanto_ext::<op>.

    Only the implementations for the python and extension library need
    to be provided: the top-level implementation for the operation is
    provided when calling this method and simply routes the calls towards
    either the python or extension implementations based on the selected
    mode.
    """
    for libname in ["quanto", "quanto_py", "quanto_ext"]:
        torch.library.define(f"{libname}::{name}", schema)

    # Provide the inplementation for all dispatch keys in the main library
    @torch.library.impl(f"quanto::{name}", "default")
    def impl(*args, **kwargs):
        if _ext_enabled:
            try:
                return getattr(torch.ops.quanto_ext, name)(*args, **kwargs)
            except Exception as e:
                if isinstance(e, NotImplementedError):
                    message = f"No optimized kernel found for quanto::{name}."
                else:
                    message = f"An exception was raised while calling the optimized kernel for quanto::{name}: {e}"
                warnings.warn(message + " Falling back to default implementation.")
        return getattr(torch.ops.quanto_py, name)(*args, **kwargs)


define("dqmm", "(Tensor input, Tensor other, Tensor other_scale) -> Tensor")
define("quantize_symmetric", "(Tensor self, Tensor scale, ScalarType dtype) -> Tensor")
define("unpack", "(Tensor self, int bits) -> Tensor")
define("udqmm", "(Tensor input, Tensor weight, Tensor scales, int bits) -> Tensor")

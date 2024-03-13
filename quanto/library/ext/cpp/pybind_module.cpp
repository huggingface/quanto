#include <torch/extension.h>
#include "mm.h"
#include "quantize.h"
#include "unpack.h"
#include "udqmm.h"

// !IMPORTANT! Some python objects such as dtype, device, are not mapped to C++ types,
// and need to be explicitly converted using dedicated helpers before calling a C++ method.
// As a consequence, when an operation takes such an object as parameter, instead
// of creating a binding directly to the C++ method, you must create a binding to a
// lambda method that converts the unmapped types and calls the C++ method.
// See the binding of quantize_symmetric for instance.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dqmm", &dqmm, "dqmm");
  m.def("udqmm", &udqmm, "udqmm");
  m.def("quantize_symmetric",
        [](const torch::Tensor& t, const torch::Tensor& scale, py::object dtype) {
          return quantize_symmetric(t,
                                    scale,
                                    torch::python::detail::py_object_to_dtype(dtype));
        }, "quantize_symmetric");
  m.def("unpack", &unpack, "unpack");
}

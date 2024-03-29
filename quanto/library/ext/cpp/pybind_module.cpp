#include <torch/extension.h>
#include "mm.h"
#include "unpack.h"

// !IMPORTANT! Some python objects such as dtype, device, are not mapped to C++ types,
// and need to be explicitly converted using dedicated helpers before calling a C++ method.
// As a consequence, when an operation takes such an object as parameter, instead
// of creating a binding directly to the C++ method, you must create a binding to a
// lambda method that converts the unmapped types and calls the C++ method.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dqmm", &dqmm, "dqmm");
  m.def("unpack", &unpack, "unpack");
}

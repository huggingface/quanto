#include <torch/extension.h>
#include "unpack.h"
#include "udqmm.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unpack", &unpack, "unpack");
  m.def("udqmm", &udqmm, "udqmm");
}

#include <pybind11/detail/common.h>
#include "pybind11/eigen.h"
#include "torch/extension.h"

#include "nvblox/nvblox.h"
#include "nvblox/serialization/internal/serializer.h"

using namespace pybind11::literals;  // For the _a shorthand notation

nvblox::TsdfLayer::Ptr loadTsdfFromFile_cstr(const char* filename);
bool writeLayerCakeToFile_cstr(const char* filename, nvblox::TsdfLayer::Ptr layer);

class NvbloxSerializer {
public:
  static nvblox::TsdfLayer::Ptr loadTsdfFromFile(const char* filename) {
    return loadTsdfFromFile_cstr(filename);
  }

  static bool writeLayerCakeToFile(const char* filename, nvblox::TsdfLayer::Ptr layer) {
    return writeLayerCakeToFile_cstr(filename, layer);
  }
};


PYBIND11_MODULE(nvbloxSerializerBindings, m) {
  py::class_<NvbloxSerializer>(m, "NvbloxSerializer")
    .def_static("load_tsdf_from_file", &NvbloxSerializer::loadTsdfFromFile)
    .def_static("write_layer_cake_to_file", &NvbloxSerializer::writeLayerCakeToFile);
};

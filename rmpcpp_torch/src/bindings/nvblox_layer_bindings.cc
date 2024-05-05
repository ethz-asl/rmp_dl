#include "nvblox/core/types.h"
#include "nvblox/nvblox.h"
#include "pybind11/eigen.h"
#include "torch/extension.h"


using namespace pybind11::literals;

PYBIND11_MODULE(nvbloxLayerBindings, m) {
  py::enum_<nvblox::MemoryType>(m, "MemoryType")
    .value("kDevice", nvblox::MemoryType::kDevice)
    .value("kUnified", nvblox::MemoryType::kUnified)
    .value("kHost", nvblox::MemoryType::kHost);

  py::class_<nvblox::EsdfLayer, std::shared_ptr<nvblox::EsdfLayer>>(
      m, "EsdfLayer")
      .def(py::init<float, nvblox::MemoryType>(),
        "voxel_size"_a, "memory_type"_a);

  py::class_<nvblox::TsdfLayer, std::shared_ptr<nvblox::TsdfLayer>>(
      m, "TsdfLayer")
      .def(py::init<float, nvblox::MemoryType>(),
        "voxel_size"_a, "memory_type"_a)
      .def("get_voxel_size", &nvblox::TsdfLayer::voxel_size)
      .def("memory_type", &nvblox::TsdfLayer::memory_type);
}

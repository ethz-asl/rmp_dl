#include "pybind11/eigen.h"
#include "torch/extension.h"


#include <core/types.h>
#include <math.h>       /* sqrt */
#include "rmpcpp_torch/conversions/nvblox_conversions.h"
#include "rmpcpp_torch/conversions/multi_array_conversions.h"
#include "rmpcpp_planner/testing/worldgen.h"

using namespace pybind11::literals;  // For the _a shorthand notation

using NumType = float;

PYBIND11_MODULE(nvbloxConversionsBindings, m) {
  /**
   * @brief Function to convert esdf layer to a 1d vector with the distance to nearest obstacle 
   * @return tuple of (shape, 1d vector)
   */
  m.def("esdf_distance_to_vector", 
  [](nvblox::EsdfLayer::Ptr esdf_layer) {
    std::function<NumType(nvblox::EsdfVoxel)> converter = [](const nvblox::EsdfVoxel& voxel){
      return voxel.is_inside ? -sqrt(NumType(voxel.squared_distance_vox)) : sqrt(NumType(voxel.squared_distance_vox));
    };

    boost::multi_array<NumType, 3> multi_array = NVBloxConverter::toMultiArray<nvblox::EsdfVoxel, NumType>(esdf_layer, converter);
    
    const auto shape = *reinterpret_cast<const std::array<size_t, 3>*>(multi_array.shape());
    std::vector<NumType> vector = MultiArrayConversions::toVector<NumType, 3>(multi_array);
    
    return py::make_tuple(shape, vector);
  });

  /**
   * @brief Function to convert tsdf layer to a 1d vector with the distance to nearest obstacle 
   * @return tuple of (shape, 1d vector)
   */
  m.def("tsdf_distance_to_vector", 
  [](nvblox::TsdfLayer::Ptr tsdf_layer) {
    std::function<NumType(nvblox::TsdfVoxel)> converter = [](const nvblox::TsdfVoxel& voxel){
      return voxel.weight < 0.1 ? 0.0 : voxel.distance;
    };

    boost::multi_array<NumType, 3> multi_array = NVBloxConverter::toMultiArray<nvblox::TsdfVoxel, NumType>(tsdf_layer, converter);
    
    const auto shape = *reinterpret_cast<const std::array<size_t, 3>*>(multi_array.shape());
    std::vector<NumType> vector = MultiArrayConversions::toVector<NumType, 3>(multi_array);
    
    return py::make_tuple(shape, vector);
  });

  m.def("get_aabb_of_observed_voxels", 
  [](nvblox::TsdfLayer::Ptr tsdf_layer) {
    Eigen::AlignedBox3f aabb = nvblox::getAABBOfObservedVoxels(*tsdf_layer.get());
    return py::make_tuple(aabb.min(), aabb.max());
  });

  m.def("get_max_indices", 
    [](nvblox::TsdfLayer::Ptr tsdf_layer) {
    std::vector<Eigen::Vector3i> indices = tsdf_layer->getAllBlockIndices();
    constexpr int voxelsPerSide = nvblox::VoxelBlock<nvblox::TsdfVoxel>::kVoxelsPerSide;

    return NVBloxConverter::getMaxIndices(indices, voxelsPerSide);
  });
  m.def("get_min_indices", 
    [](nvblox::TsdfLayer::Ptr tsdf_layer) {
    std::vector<Eigen::Vector3i> indices = tsdf_layer->getAllBlockIndices();
    constexpr int voxelsPerSide = nvblox::VoxelBlock<nvblox::TsdfVoxel>::kVoxelsPerSide;

    return NVBloxConverter::getMinIndices(indices, voxelsPerSide);
  });

  m.def("tsdf_to_esdf_with_tsdf_voxeltype", 
    [](nvblox::TsdfLayer::Ptr tsdf_input, nvblox::TsdfLayer::Ptr tsdf_output) {
      auto esdf_layer = std::make_shared<nvblox::EsdfLayer>(tsdf_input->voxel_size(), nvblox::MemoryType::kUnified);
      rmpcpp::WorldGen::generateEsdfFromTsdf(*tsdf_input, esdf_layer.get());

      NVBloxConverter::convertEsdfToTsdf(esdf_layer, tsdf_output);
    }
  );
}
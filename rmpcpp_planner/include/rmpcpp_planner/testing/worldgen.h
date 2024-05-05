#ifndef RMPCPP_PLANNER_WORLDGEN_H
#define RMPCPP_PLANNER_WORLDGEN_H

#include <random>
#include <vector>

// #include "nvblox/core/common_names.h"
// #include "nvblox/core/types.h"
// #include "nvblox/core/voxels.h"
// #include "nvblox/integrators/esdf_integrator.h"
// #include "nvblox/mesh/mesh_integrator.h"
// #include "nvblox/primitives/scene.h"
#include "nvblox/nvblox.h"
#include "rmpcpp/core/space.h"
#include "rmpcpp_planner/testing/settings.h"

namespace rmpcpp {

/**
 * Random world generator
 * @tparam Space
 */
class WorldGen {
 public:
  WorldGen() = default;
  explicit WorldGen(const struct WorldGenSettings &new_settings);

  void generateRandomWorld(const int &n, const float &r = 1.0,
                           const float &r_std = 0.2, const float& p_sphere = 0.5, const float& margin = 0.5);

  void reset();
  void seed(int seed) { settings_.seed = seed; };

  void exportToPly(const std::string& path);
  static void exportToPly(const std::string& path, nvblox::TsdfLayer& tsdf_layer);
  static void exportToPlyC_STRING(const char* c_str, nvblox::TsdfLayer& tsdf_layer);
  
  nvblox::TsdfLayer::Ptr getTsdfLayer() { return tsdf_layer_; };
  nvblox::TsdfLayer::Ptr getEsdfLayer() { return esdf_layer_; };

  static void generateEsdfFromTsdf(const nvblox::TsdfLayer& tsdf_layer, 
    nvblox::EsdfLayer* esdf_layer) {
      nvblox::EsdfIntegrator esdf_integrator;
      std::vector<Eigen::Vector3i> block_indices =
          tsdf_layer.getAllBlockIndices();
      esdf_integrator.integrateBlocks(tsdf_layer, block_indices,
                                          esdf_layer);
  }

  Eigen::Vector3d getStart() { return settings_.startpos; };
  Eigen::Vector3d getGoal() { return settings_.goal; };

  WorldType getWorldType() { return settings_.world_type; };

  inline std::pair<Eigen::Vector3d, Eigen::Vector3d> getLimits() {
    return {settings_.world_limits.first.cast<double>(),
            settings_.world_limits.second.cast<double>()};
  };

  double getDensity();

 private:
  std::unique_ptr<nvblox::primitives::Sphere> getRandomSphere(const float &r,
                                                              const float &std);
  std::unique_ptr<nvblox::primitives::Cube> getRandomCube(const float &r,
                                                          const float &std);
  Eigen::Vector3d getRandomLocation();

  struct WorldGenSettings settings_;
  nvblox::primitives::Scene scene_;
  nvblox::TsdfLayer::Ptr tsdf_layer_;

  /** Underlying datastructure of the esdf layer is still a tsdf layer, but just with a high truncation distance. 
  The nvblox esdflayer is not as accurate as the tsdf layer.  */
  nvblox::TsdfLayer::Ptr esdf_layer_;


  std::default_random_engine generator_;
};

}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_WORLDGEN_H

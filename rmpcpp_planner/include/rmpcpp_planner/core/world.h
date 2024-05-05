#ifndef RMPCPP_PLANNER_WORLD_H
#define RMPCPP_PLANNER_WORLD_H

/** RMPCPP_PLANNER */
#include "parameters.h"

/** RMPCPP */
#include "rmpcpp/core/space.h"

/** NVBLOX */
#include <Eigen/Dense>

// #include "nvblox/core/common_names.h"
// #include "nvblox/core/layer.h"
#include "nvblox/nvblox.h"

namespace rmpcpp {

/***
 * Defines the general world in which the robot moves
 * @tparam Space Space in which the world is defined (from rmpcpp/core/space)
 */
template <class Space>
class World {
 protected:
  using Vector = Eigen::Matrix<double, Space::dim, 1>;

 public:
  World() = default;
  virtual ~World() = default;

  virtual bool collision(const Vector& pos) = 0;
};

/**
 * World that interfaces with NVBlox. TODO: Figure out if keeping this templated
 * makes sense, as 2d in nvblox does not work well due to the blocks
 */
template <class Space>
class NVBloxWorld : public World<Space> {
 public:
  using Vector = typename World<Space>::Vector;

  virtual ~NVBloxWorld() = default;
  NVBloxWorld() = delete;
  NVBloxWorld(const float truncation_distance_meters)
      : truncation_distance_meters_(truncation_distance_meters){};

  bool collision(const Vector& pos) override {
    return collision(pos, tsdf_layer_);
  };
  static bool collision(const Vector& pos, nvblox::TsdfLayer::Ptr layer);

  double distanceToObstacle(const Vector& pos) {
    return distanceToObstacle(pos, esdf_layer_.get());
  };
  static double distanceToObstacle(const Vector& pos, nvblox::TsdfLayer* layer);

  Vector gradientToObstacle(const Vector& pos) {
    return gradientToObstacle(pos, esdf_layer_.get());
  };

  static Vector gradientToObstacle(const Vector& pos,
                                   nvblox::TsdfLayer* layer);

  void setTsdfLayer(const nvblox::TsdfLayer::Ptr newlayer) {
    tsdf_layer_ = newlayer;
  };
  void setEsdfLayer(const nvblox::TsdfLayer::Ptr newlayer) {
    esdf_layer_ = newlayer;
  };
  nvblox::TsdfLayer* getEsdfLayer() { return esdf_layer_.get(); };
  nvblox::TsdfLayer* getTsdfLayer() { return tsdf_layer_.get(); };

  bool checkMotion(const Vector& s1, const Vector& s2) const;
  static bool checkMotion(const Vector& s1, const Vector& s2,
                                const nvblox::TsdfLayer::Ptr layer);

  template <class VoxelType>
  static VoxelType getVoxel(const Vector& pos,
                            const nvblox::VoxelBlockLayer<VoxelType>* layer,
                            bool* succ);

  double getDensity();

  static bool collisionWithUnobservedInvalid(const Vector& pos,
                 nvblox::TsdfLayer::Ptr layer, const float margin=0.0f);

 protected:
  // We use a tsdf layer voxeltype for the esdf layer. The tsdf voxel type is more accurate, so we just use a tsdf with large trunctation distance.
  std::shared_ptr<nvblox::TsdfLayer> esdf_layer_;
  std::shared_ptr<nvblox::TsdfLayer> tsdf_layer_;

  const float truncation_distance_meters_;

  // We define our own sphere tracer to make the castOnGPU method public
  class NVbloxSphereTracer : public nvblox::SphereTracer {
    public:
    bool castOnGPU(const nvblox::Ray& ray, const nvblox::TsdfLayer& tsdf_layer,
                  const float truncation_distance_m, float* t) const {
                    return nvblox::SphereTracer::castOnGPU(ray, tsdf_layer, truncation_distance_m, t);
                  }
    };
  };

template class NVBloxWorld<Space<3>>;

}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_WORLD_H

#include "rmpcpp_planner/core/world.h"

// #include "nvblox/core/layer.h"
// #include "nvblox/ray_tracing/sphere_tracer.h"
#include "nvblox/nvblox.h"
#include "rmpcpp/core/policy_value.h"
#include "rmpcpp/core/space.h"
#include "rmpcpp_planner/policies/raycasting_CUDA.h"
#include "rmpcpp_planner/policies/simple_ESDF.h"

/** Specialized get voxel functions */
/**
 * Get the voxel at a specific location in the layer
 * @tparam VoxelType
 * @param pos Position
 * @param layer
 * @param succ
 * @return
 */
template <>
template <class VoxelType>
VoxelType rmpcpp::NVBloxWorld<rmpcpp::Space<2>>::getVoxel(
    const Vector& pos, const nvblox::VoxelBlockLayer<VoxelType>* layer,
    bool* succ) {
  Eigen::Matrix<double, 3, 1> matrix = {pos[0], pos[1], 0.0};

  std::vector<VoxelType> voxels;
  std::vector<bool> succv = {false};
  layer->getVoxels({matrix.cast<float>()}, &voxels, &succv);
  if (!succv[0]) {
    *succ = false;
    return VoxelType();
  }
  *succ = true;
  return voxels[0];
}

template <>
template <class VoxelType>
VoxelType rmpcpp::NVBloxWorld<rmpcpp::Space<3>>::getVoxel(
    const Vector& pos, const nvblox::VoxelBlockLayer<VoxelType>* layer,
    bool* succ) {
  std::vector<VoxelType> voxels;
  std::vector<bool> succv = {false};
  layer->getVoxels({pos.cast<float>()}, &voxels, &succv);
  if (!succv[0]) {
    *succ = false;
    return VoxelType();
  }
  *succ = true;
  return voxels[0];
}

/**
 * Check for collision
 * @tparam Space
 * @param pos
 * @param layer
 * @return True if collided
 */
template <class Space>
bool rmpcpp::NVBloxWorld<Space>::collision(const Vector& pos,
                                           nvblox::TsdfLayer::Ptr layer) {
  bool succ = false;
  nvblox::TsdfVoxel voxel = getVoxel<nvblox::TsdfVoxel>(pos, layer.get(), &succ);
  if (!succ) {
    return true;  // Treat unallocated space as collision?
  }
  return voxel.distance < 0.0f;
}

/**
 * Check for collision
 * @tparam Space
 * @param pos
 * @param layer
 * @return True if collided
 */
template <class Space>
bool rmpcpp::NVBloxWorld<Space>::collisionWithUnobservedInvalid(const Vector& pos,
                                           nvblox::TsdfLayer::Ptr layer, const float margin) {
  bool succ = false;
  nvblox::TsdfVoxel voxel = getVoxel<nvblox::TsdfVoxel>(pos, layer.get(), &succ);
  if (!succ) {
    return true;  // Treat unallocated space as collision?
  }
  return voxel.weight < 0.5 || voxel.distance < margin;
}

/**
 * Distance to nearest obstacle. 
 * @tparam Space
 * @param pos
 * @return
 */
template <class Space>
double rmpcpp::NVBloxWorld<Space>::distanceToObstacle(
    const Vector& pos, nvblox::TsdfLayer* layer) {
  bool succ = false;
  nvblox::TsdfVoxel voxel = getVoxel<nvblox::TsdfVoxel>(pos, layer, &succ);
  if (!succ) {
    return 0.0;
  } 
  return voxel.distance;
}

/**
 * Gradient to nearest obstacle. 
 * @tparam Space
 * @param pos
 * @return
 */
template <>
typename rmpcpp::NVBloxWorld<rmpcpp::Space<3>>::Vector rmpcpp::NVBloxWorld<rmpcpp::Space<3>>::gradientToObstacle(
    const Vector& pos, nvblox::TsdfLayer* layer) {
    
  Vector grad = Vector::Zero();

  float vs = layer->voxel_size();
  for(int dim = 0; dim < 3; dim++) {
    Vector pos1, pos2; pos1 = pos2 = pos;
    pos1[dim] += vs;
    pos2[dim] -= vs;
    bool succ1, succ2; succ1 = succ2 = false;
    nvblox::TsdfVoxel voxel1 = getVoxel<nvblox::TsdfVoxel>(pos1, layer, &succ1);
    nvblox::TsdfVoxel voxel2 = getVoxel<nvblox::TsdfVoxel>(pos2, layer, &succ2);
    if (!succ1 || !succ2) {
      return Vector::Zero();
    }
    grad[dim] = voxel1.distance - voxel2.distance;
  }

  return grad.normalized();
}

/** Check motion between 2 subsequent states by casting a ray between them. The
 * assumption is that state 1 is valid. */
/** Only implemented for 3d below */
template <class Space>
bool rmpcpp::NVBloxWorld<Space>::checkMotion(const Vector& s1,
                                             const Vector& s2) const {
  throw std::runtime_error("Not implemented");
}

template<class Space>
bool rmpcpp::NVBloxWorld<Space>::checkMotion(const Vector &s1, const Vector &s2, const nvblox::TsdfLayer::Ptr layer) {
  throw std::runtime_error("Not implemented");
}

template<>
bool rmpcpp::NVBloxWorld<rmpcpp::Space<3>>::checkMotion(
    const Vector& s1, const Vector& s2, const nvblox::TsdfLayer::Ptr layer) {
  if (collision(s2, layer)) {
    return false;
  }
  if ((s2 - s1).norm() <
      0.0001) {  // Raycasting is inconsistent if they're almost on top of each
                 // other. Assume its okay
    return true;
  }
  nvblox::Ray ray(s1.cast<float>(), ((s2 - s1) / (s2 - s1).norm()).cast<float>());

  NVbloxSphereTracer st;
  st.maximum_ray_length_m(float((s2 - s1).norm()));
  st.surface_distance_epsilon_vox(0.001);
  float t;
  st.castOnGPU(ray, *layer.get(), 0.1, &t);

  /** We don't care about the returned bool of castOnGPU, just the distance */
  if (t < (s2 - s1).norm()) {
    return false;
  }
  return true;
}

template <>
bool rmpcpp::NVBloxWorld<rmpcpp::Space<3>>::checkMotion(
    const Vector& s1, const Vector& s2) const {
  return checkMotion(s1, s2, tsdf_layer_);
}

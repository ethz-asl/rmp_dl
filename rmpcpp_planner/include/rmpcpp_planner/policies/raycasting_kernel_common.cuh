#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>

#include "nvblox/gpu_hash/internal/cuda/gpu_hash_interface.cuh"
#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"
#include "nvblox/nvblox.h"

/********************************************************************
 ****** Ray tracing kernel code (parts adapted from nvblox)
 ********************************************************************/

/** Adapted from NVBLOX */
__device__ inline bool isTsdfVoxelValid(const nvblox::TsdfVoxel &voxel) {
  constexpr float kMinWeight = 1e-4;
  return voxel.weight > kMinWeight;
}

/**
 * Casts a ray on a GPU, adapted from NVBLOX
 */
__device__ thrust::pair<float, bool> cast(
    const nvblox::Ray &ray,                                          // NOLINT
    nvblox::Index3DDeviceHashMapType<nvblox::TsdfBlock> block_hash,  // NOLINT
    float truncation_distance_m,                                     // NOLINT
    float block_size_m,                                              // NOLINT
    int maximum_steps,                                               // NOLINT
    float maximum_ray_length_m,                                      // NOLINT
    float surface_distance_epsilon_m) {
  // Approach: Step along the ray until we find the surface, or fail to
  bool last_distance_positive = false;
  // t captures the parameter scaling along ray.direction. We assume
  // that the ray is normalized which such that t has units meters.
  float t = 0.0f;
  for (int i = 0; (i < maximum_steps) && (t < maximum_ray_length_m); i++) {
    // Current point to sample
    const Eigen::Vector3f p_L = ray.origin() + t * ray.direction();

    // Evaluate the distance at this point
    float step;
    nvblox::TsdfVoxel *voxel_ptr;

    // Can't get distance, let's see what to do...
    if (!nvblox::getVoxelAtPosition(block_hash, p_L, block_size_m,
                                    &voxel_ptr) ||
        !isTsdfVoxelValid(*voxel_ptr)) {
      // 1) We weren't in observed space before this, let's step through this
      // (unobserved) st
      // uff and hope to hit something allocated.
      if (!last_distance_positive) {
        // step forward by the truncation distance
        step = truncation_distance_m;
        last_distance_positive = false;
      }
      // 2) We were in observed space, now we've left it... let's kill this
      // ray, it's risky to continue.
      // Note(alexmillane): The "risk" here is that we've somehow passed
      // through the truncation band. This occurs occasionally. The risk
      // of continuing is that we can then see through an object. It's safer
      // to stop here and hope for better luck in the next frame.
      else {
        return {t, false};
      }
    }
    // We got a valid distance
    else {
      // Distance negative (or close to it)!
      // We're gonna terminate, let's determine how.
      if (voxel_ptr->distance < surface_distance_epsilon_m) {
        // 1) We found a zero crossing. Terminate successfully.
        if (last_distance_positive) {
          // We "refine" the distance by back stepping the (now negative)
          // distance value
          t += voxel_ptr->distance;
          // Output - Success!
          return {t, true};
        }
        // 2) We just went from unobserved to negative. We're observing
        // something from behind, terminate.
        else {
          return {t, false};
        }
      }
      // Distance positive!
      else {
        // Step by this amount
        step = voxel_ptr->distance;
        last_distance_positive = true;
      }
    }

    // Step further along the ray
    t += step;
  }
  // Ran out of number of steps or distance... Fail
  return {t, false};
}
inline __device__ thrust::pair<float, float> get_angles(const float u,
                                                        const float v) {
  /** Convert uniform sample idx/dimx and idy/dimy to uniform sample on sphere
   */
  float phi = acos(1 - 2 * u);
  float theta = 2.0f * M_PI * v;
  return {phi, theta};
}
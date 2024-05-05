#include "nvblox/nvblox.h"

#include "rmpcpp_torch/kernels/get_rays.h"

#include <thrust/pair.h>

#include "rmpcpp_planner/policies/raycasting_kernel_common.cuh"
#include "rmpcpp_planner/policies/misc.cuh"

#include <torch/extension.h>

#define checkCudaErrorsNvblox(val) nvblox::check_cuda((val), #val, __FILE__, __LINE__)

#define BLOCKSIZE 8

template<typename scalar_t>
__global__ void getRaysKernel(
    // For some reason passing an  Eigen::vector3f does not work in combination with AT_DISPATCH_FLOATING_TYPES.
    // Results in some weird compilation error. So I just pass them as separate floats.  
    float originX, float originY, float originZ,
    nvblox::Index3DDeviceHashMapType<nvblox::TsdfBlock> block_hash,
    float truncation_distance, 
    float block_size, 
    int maximum_steps,
    float maximum_ray_length, 
    float surface_distance_eps,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> raycasted_depth
){
  // Thread ids 
  const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int dimx = gridDim.x * blockDim.x;
  const unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned int id = idx + idy * dimx;

  // Generate halton sequence and get angles 
  const float u = halton_seq(id, 2);
  const float v = halton_seq(id, 3);
  thrust::pair<float, float> angles = get_angles(u, v);
  float phi = angles.first;
  float theta = angles.second;

  // Convert to direction 
  float x = sin(phi) * cos(theta);
  float y = sin(phi) * sin(theta);
  float z = cos(phi);

  nvblox::Ray ray({originX, originY, originZ}, {x, y, z});

  thrust::pair<float, bool> result =
      cast(ray, block_hash, truncation_distance, block_size, maximum_steps,
           maximum_ray_length, surface_distance_eps);

    raycasted_depth[id] = result.first;
}


void getRays(
    torch::Tensor raycasted_depth,
    const Eigen::Vector3f origin,
    nvblox::TsdfLayer::Ptr layer,
    int N_sqrt, 
    float truncation_distance_vox, 
    int maximum_steps,
    float maximum_ray_length, 
    float surface_distance_epsilon_vox
    ) { 
        nvblox::GPULayerView<nvblox::TsdfBlock> gpu_layer_view = layer->getGpuLayerView(); 
        const float surface_distance_epsilon = surface_distance_epsilon_vox * layer->voxel_size();

        constexpr dim3 blockDim(BLOCKSIZE, BLOCKSIZE, 1);

        const int blocks = N_sqrt / BLOCKSIZE;
        const dim3 gridDim(blocks, blocks, 1);

        cudaDeviceSynchronize();
        checkCudaErrorsNvblox(cudaPeekAtLastError());

        AT_DISPATCH_FLOATING_TYPES(raycasted_depth.scalar_type(), "test", ([&]() {
            getRaysKernel<scalar_t><<<gridDim, blockDim>>>(
                // For some reason passing an  Eigen::vector3f does not work in combination with AT_DISPATCH_FLOATING_TYPES.
                // Results in some weird compilation error. So I just pass them as separate floats.  
                origin.x(), origin.y(), origin.z(), 
                gpu_layer_view.getHash().impl_,
                truncation_distance_vox * layer->voxel_size(), 
                gpu_layer_view.block_size(), 
                maximum_steps,
                maximum_ray_length, 
                surface_distance_epsilon, 
                raycasted_depth.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
            );
        }));  
        
        cudaDeviceSynchronize();
        checkCudaErrorsNvblox(cudaPeekAtLastError());   
    }
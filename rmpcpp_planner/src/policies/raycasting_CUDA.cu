#include <Eigen/QR>
#include <cmath>

#include <cuda.h>

// #include "nvblox/gpu_hash/cuda/gpu_hash_interface.cuh"
// #include "nvblox/gpu_hash/cuda/gpu_indexing.cuh"
// #include "nvblox/ray_tracing/sphere_tracer.h"
// #include "nvblox/utils/timing.h"
#include "nvblox/nvblox.h"
#include "rmpcpp_planner/core/parameters.h"
#include "rmpcpp_planner/policies/misc.cuh"
#include "rmpcpp_planner/policies/raycasting_CUDA.h"
#include "rmpcpp_planner/policies/raycasting_kernel_common.cuh"

#define BLOCKSIZE 8

__global__ void raycastKernel(
    const Eigen::Vector3f origin, const Eigen::Vector3f vel,
    Eigen::Matrix3f *metric_sum, Eigen::Vector3f *metric_x_force_sum,
    nvblox::Index3DDeviceHashMapType<nvblox::TsdfBlock> block_hash,
    float truncation_distance, float block_size, int maximum_steps,
    float maximum_ray_length, float surface_distance_eps,
    const rmpcpp::RaycastingCudaPolicyParameters parameters
) {
  using Vector = Eigen::Vector3f;
  using Matrix = Eigen::Matrix3f;

  /** Thread ids */
  const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int dimx = gridDim.x * blockDim.x;
  const unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned int dimy = gridDim.y * blockDim.y;
  const unsigned int id = idx + idy * dimx;

  /** Generate halton sequence and get angles*/
  const float u = halton_seq(id, 2);
  const float v = halton_seq(id, 3);
  thrust::pair<float, float> angles = get_angles(u, v);
  float phi = angles.first;
  float theta = angles.second;

  /** Convert to direction */
  float x = sin(phi) * cos(theta);
  float y = sin(phi) * sin(theta);
  float z = cos(phi);
  nvblox::Ray ray(origin, {x, y, z});

  thrust::pair<float, bool> result =
      cast(ray, block_hash, truncation_distance, block_size, maximum_steps,
           maximum_ray_length, surface_distance_eps);

  Matrix A;
  Vector metric_x_force;

  if (result.first >= maximum_ray_length) {
    /** No obstacle hit: return */
    A = Matrix::Zero();
    metric_x_force = Vector::Zero();

  } else {
    /** Calculate resulting RMP for this obstacle */
    float distance = result.first;

    Vector direction = ray.direction();

    /** Unit vector pointing away from the obstacle */
    Vector delta_d =
        -direction / direction.norm();  // Direction should be normalized so the
                                        // norm step is maybe redundant

    /** Simple RMP obstacle policy */
    Vector f_rep =
        alpha_rep(distance, parameters.eta_rep, parameters.v_rep, 0.0) *
        delta_d;

    float a_d = -alpha_damp(distance, parameters.eta_damp, parameters.v_damp,
                            parameters.epsilon_damp);
    // This looks a bit different than what we wrote in our paper, but it is just a bit of reordering of operations
    // (appendix of the rmp paper has the same ordering)
    float m = max(0.0, (-vel.transpose() * delta_d).value());
    Matrix d_d = delta_d * delta_d.transpose();
    Vector f_damp = a_d * m * d_d * vel; 

    Vector f = f_rep + f_damp;
    Vector f_norm = softnorm(f, parameters.c_softmax_obstacle);

    if (parameters.metric) {
      A = w(distance, parameters.r) * f_norm * f_norm.transpose();
    } else {
      A = w(distance, parameters.r) * Matrix::Identity();
    }
    A = A / float(dimx * dimy);  // scale with number of rays
    metric_x_force = A * f;
  }

  const int blockId = blockIdx.x + blockIdx.y * gridDim.x;

  /** Reduction within CUDA block: Start with metric reduction */
  using BlockReduceMatrix =
      typename cub::BlockReduce<Matrix, BLOCKSIZE, cub::BLOCK_REDUCE_RAKING,
                                BLOCKSIZE>;
  __shared__ typename BlockReduceMatrix::TempStorage temp_storage_matrix;
  Matrix sum_matrices0 = BlockReduceMatrix(temp_storage_matrix)
                             .Sum(A);  // Sum calculated on thread 0

  /** Metric x force reduction */
  using BlockReduceVector =
      typename cub::BlockReduce<Vector, BLOCKSIZE, cub::BLOCK_REDUCE_RAKING,
                                BLOCKSIZE>;
  __shared__ typename BlockReduceVector::TempStorage temp_storage_vector;
  Vector sum_vectors0 =
      BlockReduceVector(temp_storage_vector).Sum(metric_x_force);

  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    metric_x_force_sum[blockId] = sum_vectors0;
    metric_sum[blockId] = sum_matrices0;
  }
}

template<class Space>
rmpcpp::RaycastingCudaPolicy<Space>::~RaycastingCudaPolicy() {
  cudaStreamSynchronize(stream_);
  cudaFreeHost(metric_sum_);
  cudaFreeHost(metric_x_force_sum_);
  cudaStreamDestroy(stream_);
  checkCudaErrorsNvblox(cudaPeekAtLastError());   
}

template <class Space>
rmpcpp::RaycastingCudaPolicy<Space>::RaycastingCudaPolicy(
    RaycastingCudaPolicyParameters params, nvblox::TsdfLayer *layer)
    : layer_(layer),
      parameters_(params){
  cudaStreamCreate(&stream_);
  const int blockdim = parameters_.N_sqrt / BLOCKSIZE;
  /** Somehow trying to malloc device memory here and only deleting it at the
   * end in the destructor, instead of redoing it every integration step does
   * not work. So we only do this for the host memory, which does work
   */
  cudaMallocHost(&metric_sum_, sizeof(Eigen::Matrix3f) * blockdim * blockdim);
  cudaMallocHost(&metric_x_force_sum_,
                 sizeof(Eigen::Vector3f) * blockdim * blockdim);
}

template <class Space>
void rmpcpp::RaycastingCudaPolicy<Space>::cudaStartEval(
    const PState &state) {
  /** State vectors */
  Eigen::Vector3f pos = state.pos_.template cast<float>();
  Eigen::Vector3f vel = state.vel_.template cast<float>();

  nvblox::GPULayerView<nvblox::TsdfBlock> gpu_layer_view =
      layer_->getGpuLayerView();
  const float surface_distance_epsilon =
      parameters_.surface_distance_epsilon_vox * layer_->voxel_size();

  const int blockdim = parameters_.N_sqrt / BLOCKSIZE;
  cudaMalloc((void **)&metric_sum_device_,
             sizeof(Eigen::Matrix3f) * blockdim * blockdim);
  cudaMalloc((void **)&metric_x_force_sum_device_,
             sizeof(Eigen::Vector3f) * blockdim * blockdim);
  constexpr dim3 kThreadsPerThreadBlock(BLOCKSIZE, BLOCKSIZE, 1);
  const dim3 num_blocks(blockdim, blockdim, 1);

  cudaStreamSynchronize(stream_);
  checkCudaErrorsNvblox(cudaPeekAtLastError());   
  raycastKernel<<<num_blocks, kThreadsPerThreadBlock, 0, stream_>>>(
      pos, vel, metric_sum_device_, metric_x_force_sum_device_,
      gpu_layer_view.getHash().impl_,
      parameters_.truncation_distance_vox * layer_->voxel_size(),
      gpu_layer_view.block_size(), parameters_.max_steps, parameters_.r,
      surface_distance_epsilon, parameters_
  );       
  cudaStreamSynchronize(stream_);
  checkCudaErrorsNvblox(cudaPeekAtLastError());   
}

/**
 * Blocking call to evaluate at state.
 * @param state
 * @return
 */
template <class Space>
typename rmpcpp::RaycastingCudaPolicy<Space>::PValue
rmpcpp::RaycastingCudaPolicy<Space>::evaluateAt(
    const PState &state) {
  if (!async_eval_started_) {
    cudaStartEval(state);
  }
  /** If an asynchronous eval was started, no check is done whether the state is
   * the same. (As for now this should never happen)*/
  checkCudaErrorsNvblox(cudaStreamSynchronize(stream_));
  async_eval_started_ = false;

  const int blockdim = parameters_.N_sqrt / BLOCKSIZE;

  cudaMemcpy(metric_sum_, metric_sum_device_,
             sizeof(Eigen::Matrix3f) * blockdim * blockdim,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(metric_x_force_sum_, metric_x_force_sum_device_,
             sizeof(Eigen::Vector3f) * blockdim * blockdim,
             cudaMemcpyDeviceToHost);

  cudaFree(metric_sum_device_);
  cudaFree(metric_x_force_sum_device_);

  Eigen::Matrix3f sum = Eigen::Matrix3f::Zero();
  Eigen::Vector3f sumv = Eigen::Vector3f::Zero();
  for (int i = 0; i < blockdim * blockdim; i++) {
    sum += metric_sum_[i];
    sumv += metric_x_force_sum_[i];
  }

  Eigen::Matrix3d sumd = sum.cast<double>();
  Eigen::Matrix3d sumd_inverse =
      sumd.completeOrthogonalDecomposition().pseudoInverse(); 
  Eigen::Vector3d sumvd_scaled = sumv.cast<double>();

  Eigen::Vector3d f = sumd_inverse * sumvd_scaled;
  last_evaluated_state_.pos_ = state.pos_;
  last_evaluated_state_.vel_ = state.vel_;

  return {f, sumd};
}

/**
 * Starts asynchronous evaluation (so returns before it is done)
 This is not actually asynchronous anymore, as we now sync everywhere. TODO: Test and turn back on at some point
 * @tparam Space
 * @param state
 */
template <class Space>
void rmpcpp::RaycastingCudaPolicy<Space>::startEvaluateAsync(
    const PState &state) {
  cudaStartEval(state);
  async_eval_started_ = true;
}

/**
 * Abort asynchronous evaluation
 * @tparam Space
 */
template <class Space>
void rmpcpp::RaycastingCudaPolicy<Space>::abortEvaluateAsync() {
  cudaStreamSynchronize(stream_);
  cudaFree(metric_sum_device_);
  cudaFree(metric_x_force_sum_device_);
  async_eval_started_ = false;
}
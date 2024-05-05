
#ifndef RMPCPP_PLANNER_RAYCASTING_CUDA_H
#define RMPCPP_PLANNER_RAYCASTING_CUDA_H

// #include "nvblox/core/common_names.h"
#include "nvblox/nvblox.h"
#include "rmpcpp/core/policy_base.h"
#include "rmpcpp_planner/core/parameters.h"

#define checkCudaErrorsNvblox(val) nvblox::check_cuda((val), #val, __FILE__, __LINE__)

namespace rmpcpp {

/*
 * Implements a nvblox- map based raycasting
 * obstacle avoidance policy
 */
template <class Space>
class RaycastingCudaPolicy : public rmpcpp::PolicyBase<Space> {
 public:
  using Vector = typename PolicyBase<Space>::Vector;
  using Matrix = typename PolicyBase<Space>::Matrix;
  using PValue = typename PolicyBase<Space>::PValue;
  using PState = typename PolicyBase<Space>::PState;

  RaycastingCudaPolicy(RaycastingCudaPolicyParameters parameters,
                       nvblox::TsdfLayer *layer);

  ~RaycastingCudaPolicy();

  virtual PValue evaluateAt(const PState &state);
  virtual void startEvaluateAsync(const PState &state);
  virtual void abortEvaluateAsync();

 private:
  void cudaStartEval(const PState &state);

  const nvblox::TsdfLayer *layer_;
  const RaycastingCudaPolicyParameters parameters_;
  PState last_evaluated_state_;

  bool async_eval_started_ = false;
  cudaStream_t stream_;
  Eigen::Matrix3f *metric_sum_device_ = nullptr;
  Eigen::Vector3f *metric_x_force_sum_device_ = nullptr;
  Eigen::Matrix3f *metric_sum_ = nullptr;
  Eigen::Vector3f *metric_x_force_sum_ = nullptr;
};

}  // namespace rmpcpp

template class rmpcpp::RaycastingCudaPolicy<rmpcpp::Space<3>>;

#endif  // RMPCPP_PLANNER_RAYCASTING_CUDA_H

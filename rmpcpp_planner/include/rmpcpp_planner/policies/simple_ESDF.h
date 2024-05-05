#ifndef RMPCPP_PLANNER_SIMPLE_ESDF_H
#define RMPCPP_PLANNER_SIMPLE_ESDF_H

#include <stdexcept>
#include "nvblox/nvblox.h"
#include "rmpcpp/core/policy_base.h"
#include "rmpcpp_planner/core/parameters.h"

namespace rmpcpp {

/*
 * Implements a policy that follows the obstacle gradient
 * in an ESDF.
 */
template <class Space>
class SimpleEsdfPolicy : public rmpcpp::PolicyBase<Space> {
 public:
  using Vector = typename PolicyBase<Space>::Vector;
  using Matrix = typename PolicyBase<Space>::Matrix;
  using PValue = typename PolicyBase<Space>::PValue;
  using PState = typename PolicyBase<Space>::PState;

  SimpleEsdfPolicy(EsdfPolicyParameters parameters, nvblox::TsdfLayer* layer)
      : layer_(layer),
        parameters_(parameters){
          throw std::runtime_error("I changed the underlying datastructure to a tsdf layer with high truncation distance. \
          The evaluateAt method does not work currently, and \
          parameters will have to change and this will have to be tested before using this!!");
        };
  ~SimpleEsdfPolicy() {}

  virtual PValue evaluateAt(const PState& state);

 private:
  const nvblox::TsdfLayer* layer_;
  const EsdfPolicyParameters parameters_;
};

}  // namespace rmpcpp
#endif  // RMPCPP_PLANNER_SIMPLE_ESDF_H

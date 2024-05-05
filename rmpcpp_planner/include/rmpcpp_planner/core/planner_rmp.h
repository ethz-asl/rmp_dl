#ifndef RMPCPP_PLANNER_PLANNER_RMP_H
#define RMPCPP_PLANNER_PLANNER_RMP_H

#include <queue>

#include "parameters.h"
#include <rmpcpp/geometry/linear_geometry.h>
#include <rmpcpp/eval/trapezoidal_integrator.h>
#include "planner_base.h"
#include "rmpcpp/core/policy_base.h"
#include "rmpcpp/core/space.h"
#include "trajectory_rmp.h"

namespace rmpcpp {
/**
 * The planner class is the top-level entity that handles all planning
 * @tparam Space Space in which the world is defined (from rmpcpp/core/space)
 */
template <class Space>
class PlannerRMP : public PlannerBase<Space> {
  using Vector = Eigen::Matrix<double, Space::dim, 1>;

 public:
  friend class TrajectoryRMP<Space>;
  const int dim = Space::dim;
  PlannerRMP(const PlannerParameters& parameters);
  ~PlannerRMP() = default;

  const std::shared_ptr<TrajectoryRMP<Space>> getTrajectory() const override {
    return trajectory_;  // only valid if planner has run.
  };

  bool hasTrajectory() const override{ return trajectory_.operator bool(); }

  rmpcpp::NVBloxWorld<Space>* getWorld() {
    return dynamic_cast<rmpcpp::NVBloxWorld<Space>*>(this->world_.get());
  };

  void setup(const rmpcpp::State<Space::dim>& start,
            const Vector& goal) override;
  void step(int steps = 1) override;


  void addPolicy(const std::shared_ptr<PolicyBase<Space>>& policy);

  Vector getPos() const {return integrator_.getPos();};
  Vector getVel() const {return integrator_.getVel();};
  Vector getAcc() const {return integrator_.getAcc();};

  Vector getPreviousPos() const {return trajectory_->previous().position;};
  Vector getPreviousVel() const {return trajectory_->previous().velocity;};
  Vector getPreviousAcc() const {return trajectory_->previous().acceleration;};

 private:
  const PlannerParameters& parameters_;
  std::shared_ptr<TrajectoryRMP<Space>> trajectory_;
    
  LinearGeometry<Space::dim> geometry_;
  TrapezoidalIntegrator<PolicyBase<Space>, LinearGeometry<Space::dim>>
      integrator_;
  
  size_t num_steps_;

  std::vector<std::shared_ptr<PolicyBase<Space>>> policies_;
};

// explicit instantation
template class rmpcpp::PlannerRMP<rmpcpp::Space<3>>;

}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_PLANNER_RMP_H

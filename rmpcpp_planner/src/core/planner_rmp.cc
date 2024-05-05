
#include "rmpcpp_planner/core/planner_rmp.h"

#include <rmpcpp/core/policy_base.h>
#include <rmpcpp/eval/trapezoidal_integrator.h>
#include <rmpcpp/geometry/linear_geometry.h>
#include <stdexcept>

#include "rmpcpp_planner/core/parameters.h"
#include "rmpcpp_planner/core/trajectory_rmp.h"

/**
 * Constructor
 * @tparam Space
 * @param parameters
 */
template <class Space>
rmpcpp::PlannerRMP<Space>::PlannerRMP(const PlannerParameters &parameters)
    : PlannerBase<Space>(
          std::make_unique<rmpcpp::NVBloxWorld<Space>>(parameters.truncation_distance_vox * parameters.voxel_size)),
      parameters_(parameters) {}


template <class Space>
void rmpcpp::PlannerRMP<Space>::step(int steps) {
  int i = 0;
  while (!this->collided_ && !this->goal_reached_ && !this->diverged_) {
    // evaluate policies
    auto policies = this->policies_;
    /** Convert shared pointers to normal pointers for integration step */
    std::vector<PolicyBase<Space> *> policiesRaw;
    policiesRaw.reserve(policies.size());
    std::transform(policies.cbegin(), policies.cend(),
                   std::back_inserter(policiesRaw),
                   [](auto &ptr) { return ptr.get(); });

    // integrate
    this->integrator_.forwardIntegrate(policiesRaw, geometry_, parameters_.dt);

    // get new positions
    Vector position, velocity, acceleration;
    this->integrator_.getState(position, velocity, acceleration);

    // update exit conditions
    /** Collision check */
    if (!this->getWorld()->checkMotion(trajectory_->current().position,
                                       position)) {
      this->collided_ = true;
    }

    if ((position - this->goal_).norm() <
        this->goal_tolerance_) {
      this->goal_reached_ = true;
    }

    if (this->num_steps_ > parameters_.max_length) {
      this->diverged_ = true;
    }

    this->num_steps_++;
    // store results
    trajectory_->addPoint(position, velocity, acceleration);
    
    if(steps != -1 && ++i >= steps) break;
  }
}

/**
 * Start planning run
 * @tparam Space
 * @param start
 */
template <class Space>
void rmpcpp::PlannerRMP<Space>::setup(const rmpcpp::State<Space::dim> &start,
                                     const Vector &goal) {
  this->setGoal(goal);

  if(this->getWorld()->getTsdfLayer() == nullptr){
    std::cout << "Set TSDF before planning" << std::endl;
    throw std::runtime_error("Set TSDF before planning");
  }
                                      
  // Reset states
  this->collided_ = false;
  this->goal_reached_ = false;
  this->diverged_ = false;
  trajectory_ = std::make_unique<TrajectoryRMP<Space>>(start.pos_, start.vel_);
  this->integrator_.resetTo(trajectory_->current().position,
                            trajectory_->current().velocity);
  this->num_steps_ = 0;
}


template <class Space>
void rmpcpp::PlannerRMP<Space>::addPolicy(const std::shared_ptr<PolicyBase<Space>> &policy) {
  this->policies_.push_back(policy);
}


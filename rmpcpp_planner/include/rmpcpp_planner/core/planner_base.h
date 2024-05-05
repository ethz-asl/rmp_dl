#ifndef RMPCPP_PLANNER_PLANNER_BASE_H
#define RMPCPP_PLANNER_PLANNER_BASE_H

#include <rmpcpp_planner/core/trajectory_rmp.h>

#include <numeric>

#include "Eigen/Dense"
#include "nvblox/nvblox.h"
#include "world.h"

namespace rmpcpp {

/**
 * Base class for a planner that uses nvblox as a world.
 * @tparam Space
 */
template <class Space>
class PlannerBase {
 public:
  using Vector = Eigen::Matrix<double, Space::dim, 1>;

  PlannerBase() = default;
  virtual ~PlannerBase() = default;
  explicit PlannerBase(std::unique_ptr<rmpcpp::NVBloxWorld<Space>> world) {
    this->world_ = std::move(world);
  };

  /** Pure virtual */
  virtual void setup(const rmpcpp::State<Space::dim>& start,
                    const Vector& end) = 0;
  virtual void step(int steps) = 0;

  void setTsdf(const nvblox::TsdfLayer::Ptr& tsdflayer) {
    this->world_->setTsdfLayer(tsdflayer);
  };
  void setEsdf(const nvblox::TsdfLayer::Ptr& esdflayer) {
    this->world_->setEsdfLayer(esdflayer);
  };

  bool success() const { return goal_reached_ && !diverged_ && !collided_; };
  bool collided() const { return collided_; };
  bool diverged() const { return diverged_; };

  virtual bool hasTrajectory() const = 0;
  virtual const std::shared_ptr<TrajectoryRMP<Space>> getTrajectory() const = 0;

 protected:
  virtual void setGoal(const Vector& new_goal) {
    this->goal_ = new_goal;
  };
  Vector goal_ = Vector::Zero();
  std::unique_ptr<rmpcpp::NVBloxWorld<Space>> world_;

  double goal_tolerance_ = 0.30;
  bool collided_ = false;
  bool goal_reached_ = false;
  bool diverged_ = false;
};

}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_PLANNER_BASE_H

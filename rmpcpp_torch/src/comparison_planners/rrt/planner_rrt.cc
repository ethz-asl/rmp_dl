
#include "comparison_planners/rrt/planner_rrt.h"
#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <memory>

#include "comparison_planners/rrt/parameters_rrt.h"


#include "rmpcpp_torch/conversions/nvblox_conversions.h"

template <typename Space>
void PlannerRRT<Space>::plan(const Vector &start, const Vector &goal, const double time) {
  si->setStateValidityChecker([this](const ob::State *state) {
    return isStateValid(state->as<ob::RealVectorStateSpace::StateType>());
  });
  // The resolution is expected as a fraction of the maximum extent of the space
  std::cout << "Maximum extent: " << si->getMaximumExtent() << std::endl;
  si->setStateValidityCheckingResolution(
      this->_parameters.stateValidityCheckerResolution_m  / si->getMaximumExtent());

  planner->clear();
  path.clear();
  status = ob::PlannerStatus::UNKNOWN;
  planner->setRange(_parameters.range);
  planner->setup();

  ob::ScopedState<> ob_start(space);
  for (int i = 0; i < Space::dim; i++) {
    ob_start->as<ob::RealVectorStateSpace::StateType>()->values[i] = start[i];
  }

  ob::ScopedState<> ob_goal(space);
  for (int i = 0; i < Space::dim; i++) {
    ob_goal->as<ob::RealVectorStateSpace::StateType>()->values[i] = goal[i];
  }
  
  ss->setStartAndGoalStates(ob_start, ob_goal);

  ss->setup();
  ss->setPlanner(planner);

  status = ss->solve(time);
  
  og::PathGeometric path = ss->getSolutionPath();

  for (size_t i = 0; i < path.getStateCount(); i++) {
    const auto *state =
        path.getState(i)->as<ob::RealVectorStateSpace::StateType>();
    Vector pos;
    for (int j = 0; j < Space::dim; j++) {
      pos[j] = state->values[j];
    }
    this->path.push_back(pos);
  }
}

template <typename Space>
PlannerRRT<Space>::PlannerRRT(const RRT::RRTParameters &parameters)
    : _parameters(parameters) {
}

template <typename Space>
void PlannerRRT<Space>::setEsdf(const nvblox::TsdfLayer::Ptr &esdfLayer) {
  // Set the esdf layer
  this->esdfLayer = esdfLayer;

  Eigen::AlignedBox3f aabb = nvblox::getAABBOfObservedVoxels(*this->esdfLayer);

  ob::RealVectorBounds bounds(Space::dim);
  for (int i = 0; i < Space::dim; i++) {
    bounds.setLow(i, aabb.min()[i]);
    bounds.setHigh(i, aabb.max()[i]);
  }
  space->setBounds(bounds);

  std::vector<Eigen::Vector3i> indices = esdfLayer->getAllBlockIndices();
  constexpr int voxelsPerSide =
      nvblox::VoxelBlock<nvblox::TsdfVoxel>::kVoxelsPerSide;

  const std::vector<int> max_indices =
      NVBloxConverter::getMaxIndices(indices, voxelsPerSide);
  const std::vector<int> min_indices =
      NVBloxConverter::getMinIndices(indices, voxelsPerSide);

  this->cpu_esdf_.resize((std::vector<int>{max_indices[0] - min_indices[0],
                                           max_indices[1] - min_indices[1],
                                           max_indices[2] - min_indices[2]}));

  this->cpu_esdf_ = NVBloxConverter::toMultiArray<nvblox::TsdfVoxel>(esdfLayer);

  // The min indices are important to figure out the correct conversion from
  // world coordinates to voxel coordinates for the cpu converted esdf
  this->min_indices_ = NVBloxConverter::getMinIndices(
      esdfLayer->getAllBlockIndices(),
      nvblox::VoxelBlock<nvblox::TsdfVoxel>::kVoxelsPerSide);
  
}

template <typename Space>
double PlannerRRT<Space>::distanceToObstacle(const Vector &pos) {
  float voxel_size = this->esdfLayer->voxel_size();

  // Convert world coordinates to voxel coordinates
  // The min indices are important to figure out the correct conversion, and are
  // the (voxel) indices of the first voxel. E.g. if the first voxel starts at
  // position 10, with voxel size 0.2, then the min index is 50.
  std::vector<int> voxel_pos = {
      std::floor(pos[0] / voxel_size) - this->min_indices_[0],
      std::floor(pos[1] / voxel_size) - this->min_indices_[1],
      std::floor(pos[2] / voxel_size) - this->min_indices_[2]};

  // We now need to make sure that we are within bounds. If we are outside of
  // the array, we simply pick the nearest voxel on the edge of the array.
  for (int i = 0; i < 3; i++) {
    if (voxel_pos[i] < 0) {
      voxel_pos[i] = 0;
    }
    if (voxel_pos[i] >= this->cpu_esdf_.shape()[i]) {
      voxel_pos[i] = this->cpu_esdf_.shape()[i] - 1;
    }
  }
  nvblox::TsdfVoxel voxel =
      this->cpu_esdf_[voxel_pos[0]][voxel_pos[1]][voxel_pos[2]];
  return voxel.weight > 0.5 ? voxel.distance : -1.0;
}

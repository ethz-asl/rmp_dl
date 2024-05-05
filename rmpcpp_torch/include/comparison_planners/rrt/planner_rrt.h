#ifndef RMPCPP_PLANNER_PLANNER_RRT_H
#define RMPCPP_PLANNER_PLANNER_RRT_H

#include "rmpcpp_planner/core/world.h"
#include "Eigen/Dense"
#include <boost/multi_array.hpp>
#include <memory>

#include "rmpcpp/core/state.h"


#include "comparison_planners/rrt/parameters_rrt.h"
#include <map/common_names.h>
#include <ompl/base/Planner.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>


namespace og = ompl::geometric;
namespace ob = ompl::base;

template<class Space>
class PlannerRRT {
public:
    using Vector = Eigen::Matrix<double, Space::dim, 1>;

    explicit PlannerRRT(const RRT::RRTParameters& parameters);

    void plan(const Vector& start, const Vector& goal, double time);
    void setEsdf(const nvblox::TsdfLayer::Ptr& tsdfLayer);
    std::vector<Vector> getPath() {return path;}
    bool success() {return status == ob::PlannerStatus::EXACT_SOLUTION;}


protected:
    double distanceToObstacle(const Vector& pos);

    const RRT::RRTParameters _parameters;

    std::shared_ptr<ob::RealVectorStateSpace> space = std::make_shared<ob::RealVectorStateSpace>(Space::dim);
    std::shared_ptr<ob::SpaceInformation> si = std::make_shared<ob::SpaceInformation>(space);
    std::shared_ptr<og::SimpleSetup> ss = std::make_shared<og::SimpleSetup>(si);
    std::shared_ptr<og::RRTstar> planner = std::make_shared<og::RRTstar>(si);
    
    ob::PlannerStatus status;
    std::vector<Vector> path;
    
    boost::multi_array<nvblox::TsdfVoxel, 3> cpu_esdf_;
    std::vector<int> min_indices_;

    Vector start;

    // We use tsdf voxel type for esdf layer as this is more accurate than the esdf voxel
    // When using perfect simulated worlds tsdf with high truncation distance should be the same as esdf 
    nvblox::TsdfLayer::Ptr esdfLayer;

    bool isStateValid(const ob::State* state) {
        Vector pos = Eigen::Map<Vector>(state->as<ob::RealVectorStateSpace::StateType>()->values, Space::dim, 1);

        return distanceToObstacle(pos) > this->_parameters.margin_to_obstacles;
    }
};

// explicit instantiation
template class PlannerRRT<rmpcpp::Space<3>>;

#endif  // RMPCPP_PLANNER_PLANNER_RRT_H
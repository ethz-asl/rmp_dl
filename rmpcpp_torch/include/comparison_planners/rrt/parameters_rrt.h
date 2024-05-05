#ifndef RMPCPP_PLANNER_PARAMETERS_RRT_H
#define RMPCPP_PLANNER_PARAMETERS_RRT_H


#include "Eigen/Dense"
#include "rmpcpp_planner/core/parameters.h"

namespace RRT {

enum TypeRRT{
    RRT_STAR
};

struct RRTParameters {
    TypeRRT type = RRT_STAR;
    double stateValidityCheckerResolution_m = 0.01;
    double range = 10.0;
    double margin_to_obstacles = 0.0;
};

} // namespace RRT

#endif //RMPCPP_PLANNER_PARAMETERS_RRT_Hq
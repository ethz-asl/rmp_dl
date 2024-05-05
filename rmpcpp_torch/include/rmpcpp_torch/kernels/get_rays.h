#ifndef GET_RAYS_H
#define GET_RAYS_H

#include <Eigen/Core>
#include <torch/extension.h>
#include "nvblox/nvblox.h"

void getRays(
    torch::Tensor raycasted_depth,
    Eigen::Vector3f origin, 
    nvblox::TsdfLayer::Ptr layer,
    int N_sqrt, 
    float truncation_distance_vox, 
    int maximum_steps, 
    float maximum_ray_length, 
    float surface_distance_epsilon_vox);

#endif // GET_RAYS_H
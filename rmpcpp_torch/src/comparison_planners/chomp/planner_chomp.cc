#include "comparison_planners/chomp/planner_chomp.h"
#include "comparison_planners/chomp/chomp_optimizer.h"
#include "nvblox/nvblox.h"

template<class Space>
PlannerChomp<Space>::PlannerChomp(chomp::ChompParameters parameters, const int N, const bool cpu) :
    world_(std::make_unique<rmpcpp::NVBloxWorld<Space>>(1.0)), parameters(parameters), N_(N), cpu_(cpu){
        // The truncation distance does not matter, as we only use the esdf anyway. 
    chomper.setParameters(parameters);
    if(cpu_){
        chomper.setDistanceFunction(
                [this](const Vector& pos){ // Distance function
                    return this->distanceToObstacle(pos);
        });
    } else {
        chomper.setDistanceFunction(
                [this](const Vector& pos){ // Distance function
                    return this->world_->distanceToObstacle(pos);
        });
    }
}

/**
 * Start planning run
 * @tparam Space
 * @param start Start state
 */
template<class Space>
void PlannerChomp<Space>::plan(const Vector &start, const Vector &goal) {
    trajectory_vector.clear();
    chomper.solveProblem(start, goal, this->N_, &trajectory);

    this->goal_reached = true;
    for(int i = 0; i < trajectory.trajectory.rows(); i++){
        Vector row = trajectory.trajectory.row(i);
        if(this->world_->collision(row)){
            this->goal_reached = false;
        }
        trajectory_vector.push_back(row);
    }
}



template<class Space>
double PlannerChomp<Space>::distanceToObstacle(const Vector &pos) {
    float voxel_size = this->world_->getEsdfLayer()->voxel_size();

    // Convert world coordinates to voxel coordinates
    // The min indices are important to figure out the correct conversion, and are the (voxel) indices of the first voxel. 
    // E.g. if the first voxel starts at position 10, with voxel size 0.2, then the min index is 50. 
    std::vector<int> voxel_pos = {
        std::floor(pos[0] / voxel_size) - this->min_indices_[0],
        std::floor(pos[1] / voxel_size) - this->min_indices_[1],
        std::floor(pos[2] / voxel_size) - this->min_indices_[2]
    };

    // We now need to make sure that we are within bounds. If we are outside of the array, we simply pick the nearest voxel
    // on the edge of the array.
    for(int i = 0; i < 3; i++){
        if(voxel_pos[i] < 0){
            voxel_pos[i] = 0;
        }
        if(voxel_pos[i] >= this->cpu_esdf_.shape()[i]){
            voxel_pos[i] = this->cpu_esdf_.shape()[i] - 1;
        }
    }
    nvblox::TsdfVoxel voxel = this->cpu_esdf_[voxel_pos[0]][voxel_pos[1]][voxel_pos[2]];
  return voxel.weight > 0.5 ? voxel.distance : -1.0;
}

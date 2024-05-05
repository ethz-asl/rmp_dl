#ifndef RMPCPP_PLANNER_PLANNER_CHOMP_H
#define RMPCPP_PLANNER_PLANNER_CHOMP_H

#include <boost/multi_array.hpp>
#include <iostream>
#include "nvblox/nvblox.h"
#include "rmpcpp/core/state.h"
#include "rmpcpp_planner/core/planner_base.h"
#include "comparison_planners/chomp/chomp_optimizer.h"
#include "rmpcpp_planner/core/world.h"

#include "rmpcpp_torch/conversions/nvblox_conversions.h"

template<class Space>
class PlannerChomp {
public:
    using Vector = Eigen::Matrix<double, Space::dim, 1>;
    explicit PlannerChomp(chomp::ChompParameters parameters, int N, bool cpu = true);

    void plan(const Vector& start, const Vector& goal);

    std::vector<Vector> getPath() {return trajectory_vector;};

    void setTsdf(const nvblox::TsdfLayer::Ptr& tsdfLayer) {
        this->world_->setTsdfLayer(tsdfLayer);
    };
    void setEsdf(const nvblox::TsdfLayer::Ptr& tsdfLayer) {
        this->world_->setEsdfLayer(tsdfLayer);
        
        std::vector<Eigen::Vector3i> indices = tsdfLayer->getAllBlockIndices();
        constexpr int voxelsPerSide = nvblox::VoxelBlock<nvblox::TsdfVoxel>::kVoxelsPerSide;
        
        const std::vector<int> max_indices = NVBloxConverter::getMaxIndices(indices, voxelsPerSide);
        const std::vector<int> min_indices = NVBloxConverter::getMinIndices(indices, voxelsPerSide);

        this->cpu_esdf_.resize((std::vector<int>{
            max_indices[0] - min_indices[0], 
            max_indices[1] - min_indices[1],
            max_indices[2] - min_indices[2]
            }));


        this->cpu_esdf_ = NVBloxConverter::toMultiArray<nvblox::TsdfVoxel>(tsdfLayer);

        // The min indices are important to figure out the correct conversion from world coordinates to voxel coordinates
        // for the cpu converted esdf
        this->min_indices_ = NVBloxConverter::getMinIndices(tsdfLayer->getAllBlockIndices(), 
                            nvblox::VoxelBlock<nvblox::TsdfVoxel>::kVoxelsPerSide);
    };

    double distanceToObstacle(const Vector& pos);
    
    Vector gradientToObstacle(const Vector& pos) {
        auto gradient = this->world_->gradientToObstacle(pos);
        return gradient;
    };

    bool success() {return this->goal_reached;};

private:
    chomp::ChompParameters parameters;
    int N_;
    chomp::ChompOptimizer chomper;
    chomp::ChompTrajectory trajectory;
    std::vector<Vector> trajectory_vector;

    bool goal_reached = false;

    std::unique_ptr<rmpcpp::NVBloxWorld<Space>> world_;

    boost::multi_array<nvblox::TsdfVoxel, 3> cpu_esdf_;
    std::vector<int> min_indices_;
    
    const bool cpu_;
};

// explicit instantiation
template class PlannerChomp<rmpcpp::Space<3>>;

#endif //RMPCPP_PLANNER_PLANNER_CHOMP_H

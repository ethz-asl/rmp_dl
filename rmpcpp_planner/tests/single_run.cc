#include <iostream>
#include "rmpcpp/core/space.h"
#include "rmpcpp/policies/simple_target_policy.h"
#include "rmpcpp_planner/core/parameters.h"
#include "rmpcpp_planner/policies/raycasting_CUDA.h"
#include "rmpcpp_planner/testing/settings.h"
#include "rmpcpp_planner/testing/tester.h"
#include "rmpcpp_planner/testing/parser.h"
#include "rmpcpp_planner/testing/worldgen.h"

int main(int argc, char* argv[]){

  using Space = rmpcpp::Space<3>;

  rmpcpp::WorldGenSettings settings;
  settings.seed = 1700089;

  settings.world_type = rmpcpp::SPHERES_BOX_WORLD;

  settings.world_limits = {
      {-0.2f, -0.2f, -0.2f}, {10.2f, 10.2f, 10.2f}};

  settings.voxel_size = 0.2f;
  settings.block_size = nvblox::VoxelBlock<bool>::kVoxelsPerSide * settings.voxel_size;

  settings.startpos = {1.0, 1.0, 1.0};
  settings.goal = {9.0, 9.0, 9.0};

  // In voxels, so this gets multiplied with voxel_size later on
  settings.voxel_truncation_distance_vox = 4.0f;

  rmpcpp::PlannerParameters params;
  params.dt = 0.04;
  params.max_length = 5000;
  // In voxels, so this gets multiplied with voxel_size later on
  params.truncation_distance_vox = 4.0f;
  params.voxel_size = 0.2f;
  params.terminate_upon_goal_reached = true;

  rmpcpp::WorldGen worldgen(settings);

  worldgen.generateRandomWorld(70);

  Eigen::Vector3d startv = worldgen.getStart();
  Eigen::Vector3d goalv = worldgen.getGoal();

  auto planner = std::make_unique<rmpcpp::PlannerRMP<rmpcpp::Space<3>>>(params);

  /**
   * We set both ESDF and TSDF. Note that the RMP planner does not actually use
   * the ESDF.
   */
  planner->setEsdf(worldgen.getEsdfLayer());
  planner->setTsdf(worldgen.getTsdfLayer());

  rmpcpp::TargetPolicyParameters target_parameters;


  /** Target Policy */
  std::shared_ptr<rmpcpp::SimpleTargetPolicy<Space>> target_policy =
      std::make_shared<rmpcpp::SimpleTargetPolicy<Space>>(
        goalv, Eigen::Matrix<double, Space::dim, Space::dim>::Identity(),
        target_parameters.alpha, target_parameters.beta, target_parameters.c_softmax
        );

  rmpcpp::RaycastingCudaPolicyParameters raycasting_cuda_policy_parameters;
  raycasting_cuda_policy_parameters.eta_rep = 88.0;
  raycasting_cuda_policy_parameters.eta_damp = 140.0;
  raycasting_cuda_policy_parameters.v_rep = 1.2;
  raycasting_cuda_policy_parameters.v_damp = 1.2;
  raycasting_cuda_policy_parameters.epsilon_damp = 0.1;
  raycasting_cuda_policy_parameters.c_softmax_obstacle = 0.2;
  raycasting_cuda_policy_parameters.r = 2.4;
  raycasting_cuda_policy_parameters.metric = true;
  raycasting_cuda_policy_parameters.N_sqrt = 32;
  raycasting_cuda_policy_parameters.max_steps = 100;
  raycasting_cuda_policy_parameters.surface_distance_epsilon_vox = 0.1;

  /** 'World Policy' */
  std::shared_ptr<rmpcpp::PolicyBase<Space>> world_policy;
  world_policy = std::make_shared<rmpcpp::RaycastingCudaPolicy<Space>>(
        raycasting_cuda_policy_parameters, worldgen.getTsdfLayer().get());
  

  planner->addPolicy(target_policy);
  planner->addPolicy(world_policy);

  rmpcpp::State<3> start(startv, {0.0, 0.0, 0.0});
  planner->setup(start, goalv);

  auto starttime = std::chrono::high_resolution_clock::now();
  planner->step(-1);
  auto endtime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      endtime - starttime);
  double duration_s = double(duration.count()) / 1E6;


  std::string success = planner->success() ? "Success: " : "Failure: ";
  std::cout << success << double(duration.count()) / 1000.0 << "ms"
            << std::endl;

}

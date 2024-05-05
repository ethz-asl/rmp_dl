#ifndef RMPCPP_PLANNER_PARAMETERS_H
#define RMPCPP_PLANNER_PARAMETERS_H

#include "Eigen/Dense"

namespace rmpcpp {

/**
 * Most of the default values here all get overridden by the parser class.
 */

enum PolicyType { SIMPLE_ESDF, RAYCASTING_CUDA };

struct TargetPolicyParameters {

  /** Target policy parameters */
  double alpha = 10.0;
  double beta = 15.0;
  double c_softmax = 0.2;
  double metric; // Metric multiplicative constant
};

struct EsdfPolicyParameters {
  /** Simple ESDF policy parameters. */
  double eta_rep = 22.0;   // Gets multiplied by a gain factor from the parser
  double eta_damp = 35.0;  // Gets multiplied by a gain factor from the parser
  double v_rep = 2.0;
  double v_damp = 2.0;
  double epsilon_damp = 0.1;
  double c_softmax_obstacle = 0.2;
  double r = 5.0;
};


struct RaycastingCudaPolicyParameters {
  float eta_rep = 22.0;   // Gets multiplied by a gain factor from the parser
  float eta_damp = 35.0;  // Gets multiplied by a gain factor from the parser
  float v_rep = 2.0;
  float v_damp = 2.0;
  float epsilon_damp = 0.1;
  float c_softmax_obstacle = 0.2;
  float r = 5.0;
  bool metric = true;

  int N_sqrt = 32;  // square root of number of rays. Has to be divisible by
                    // blocksize TODO: deal with non divisibility
  float surface_distance_epsilon_vox = 0.1f;
  int max_steps = 100;
  
  // In voxels, so this gets multiplied with voxel_size later on
  double truncation_distance_vox = 4.0f;
};


struct LidarNodeParameters {
  RaycastingCudaPolicyParameters raycastingCudaPolicyParameters;
  TargetPolicyParameters targetParameters;
};

struct PlannerParameters {
  double dt = 0.04;
  int max_length = 2000;
  // In voxels, so this gets multiplied with voxel_size later on
  double truncation_distance_vox = 4.0f;
  double voxel_size = 0.2f;

  bool terminate_upon_goal_reached = true;
};


struct ParametersRMP {
  PolicyType policy_type;

  PlannerParameters plannerParameters;
  TargetPolicyParameters targetPolicyParameters;
  RaycastingCudaPolicyParameters rayCastingCudaPolicyParameters;
  EsdfPolicyParameters esdfPolicyParameters;
};

}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_PARAMETERS_H

#include "rmpcpp_planner/testing/parser.h"

#include "rmpcpp_planner/core/parameters.h"
#include "rmpcpp_planner/testing/settings.h"

/**
 * Parser class to run experiments with different planner types (and their
 * parameters)
 * @param argc
 * @param argv
 */
Parser::Parser(int argc, char** argv) {
  po::options_description desc("Usage");
  /**
   * Big list of options
   */
  desc.add_options()("planner_type", po::value<int>()->default_value(0),
                     "Planner type. 0: RMP")(
      "obstacles", po::value<int>()->default_value(90), "Number of obstacles")(
      "n_runs", po::value<int>()->default_value(10), "Number of runs")(
      "seed", po::value<int>()->default_value(0),
      "Starting seed for the map random generator. Successive runs add seed + "
      "1")(
      "data_path",
      po::value<std::string>()->default_value("../eval/data/stats/default/"),
      "Path to save data files from this run")(
      "world_path",
      po::value<std::string>()->default_value("../eval/data/world/default/"),
      "Path to save world files from this run (ply format). Will not save if "
      "empty string is given.")(
      "world_load_path",
      po::value<std::string>()->default_value(
          "../eval/data/world/custom/maps/office.tsdf"),
      "Path to load the world file from.")(
      "policy_type", po::value<int>()->default_value(1),
      "World policy type. 0: Simple ESDF. 1: Raycast CUDA.")(
      "stats_only", po::value<int>()->default_value(0),
      "If set to 1, it will only output statistics (so no trajectory or world "
      "file)")("gain", po::value<float>()->default_value(4.0f),
               "Gain of obstacle policies.")(
      "metric", po::value<int>()->default_value(1),
      "Whether it uses a directionally stretched metric for obstacle "
      "policies.")(
      "N_sqrt", po::value<int>()->default_value(32),
      "Square root of number of rays cast. Must be divisible by blocksize")(
      "r", po::value<double>()->default_value(2.4),
      "Radius of CUDA raycasting and obstacle policy")(
      "trunc_dist", po::value<double>()->default_value(4.0),
      "Truncation distance of the TSDF in voxels (so gets multiplied with voxel_size later on)")(
      "v_rep_damp", po::value<float>()->default_value(1.2),
      "V parameter of the repulsive and damping obstacle policy terms")(
      "world_type", po::value<int>()->default_value(1),
      "World type: 0 for only spheres, 1 for a mix of spheres and cubes")(
      "terminate_upon_goal_reached", po::value<int>()->default_value(1),
      "Termination when goal is reached: 0 no, 1 yes.")(
      "dt", po::value<double>()->default_value(0.06), "Dt of RMP.");
  std::cout << "Options: \n" << desc << std::endl;
  po::store(po::parse_command_line(argc, argv, desc), opts_);
}

bool Parser::parse() {
  try {
    po::notify(opts_);
  } catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return false;
  }
  return true;
}

rmpcpp::TestSettings Parser::getSettings() {
  /** Settings */
  rmpcpp::TestSettings settings;
  settings.obstacles = opts_["obstacles"].as<int>();
  settings.n_runs = opts_["n_runs"].as<int>();
  settings.seed = opts_["seed"].as<int>();
  srand(settings.seed);
  settings.data_path = opts_["data_path"].as<std::string>();
  settings.world_save_path = opts_["world_path"].as<std::string>();
  settings.world_load_path = opts_["world_load_path"].as<std::string>();
  settings.stats_only = opts_["stats_only"].as<int>();
  settings.voxel_truncation_distance_vox = opts_["trunc_dist"].as<double>();
  settings.world_type =
      static_cast<rmpcpp::WorldType>(opts_["world_type"].as<int>());
  settings.planner_type =
      static_cast<rmpcpp::PlannerType>(opts_["planner_type"].as<int>());
  return settings;
}

rmpcpp::ParametersRMP Parser::getParameters() {
  switch (static_cast<rmpcpp::PlannerType>(opts_["planner_type"].as<int>())) {
    case rmpcpp::RMP:
      return getRMPParameters();
    default:
      throw std::runtime_error("Planner tpye not implemented in parser.cc");
  }
}

rmpcpp::ParametersRMP Parser::getRMPParameters() {
  /** Parameters */
  rmpcpp::PolicyType policy_type =
      static_cast<rmpcpp::PolicyType>(opts_["policy_type"].as<int>());
  rmpcpp::ParametersRMP parameters;
  parameters.policy_type = policy_type;

  parameters.plannerParameters.truncation_distance_vox = opts_["trunc_dist"].as<double>();
  parameters.plannerParameters.terminate_upon_goal_reached =
      static_cast<bool>(opts_["terminate_upon_goal_reached"].as<int>());
  parameters.plannerParameters.dt = opts_["dt"].as<double>();

  float gain = opts_["gain"].as<float>();
  parameters.rayCastingCudaPolicyParameters.eta_damp *= gain;
  parameters.rayCastingCudaPolicyParameters.eta_rep *= gain;
  parameters.esdfPolicyParameters.eta_damp *= gain;
  parameters.esdfPolicyParameters.eta_rep *= gain;

  float v_rep_damp = opts_["v_rep_damp"].as<float>();
  parameters.rayCastingCudaPolicyParameters.v_damp = v_rep_damp;
  parameters.rayCastingCudaPolicyParameters.v_rep = v_rep_damp;
  parameters.esdfPolicyParameters.v_damp = v_rep_damp;
  parameters.esdfPolicyParameters.v_rep = v_rep_damp;

  float r = opts_["r"].as<double>();
  parameters.rayCastingCudaPolicyParameters.r = r;
  parameters.esdfPolicyParameters.r = r;

  parameters.rayCastingCudaPolicyParameters.metric = bool(opts_["metric"].as<int>());
  parameters.rayCastingCudaPolicyParameters.N_sqrt = opts_["N_sqrt"].as<int>();
  parameters.rayCastingCudaPolicyParameters.truncation_distance_vox = opts_["trunc_dist"].as<double>();

  return parameters;
}

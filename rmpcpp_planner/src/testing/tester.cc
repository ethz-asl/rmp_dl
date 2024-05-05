#include "rmpcpp_planner/testing/tester.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>

#include "boost/format.hpp"
#include "boost/variant.hpp"
#include "rmpcpp/policies/simple_target_policy.h"
#include "rmpcpp_planner/core/parameters.h"
#include "rmpcpp_planner/core/planner_rmp.h"
#include "rmpcpp_planner/policies/raycasting_CUDA.h"
#include "rmpcpp_planner/policies/simple_ESDF.h"
#include "rmpcpp_planner/testing/settings.h"
#include "rmpcpp_planner/testing/worldgen.h"

Tester::Tester(const rmpcpp::ParametersRMP& parameters,
               const rmpcpp::TestSettings& settings)
    : worldgen_(rmpcpp::WorldGen(rmpcpp::WorldGenSettings(settings))),
      parameters_(parameters),
      settings_(settings) {}

/**
 * Does a single planner run
 */
void Tester::runSingle(const size_t run_index) {
  worldgen_.seed(settings_.seed);
  switch (settings_.world_type) {
    case rmpcpp::SPHERES_ONLY_WORLD:
    case rmpcpp::SPHERES_BOX_WORLD:
      worldgen_.generateRandomWorld(settings_.obstacles);
      break;
  }
  
  Eigen::Vector3d startv = worldgen_.getStart();
  Eigen::Vector3d goalv = worldgen_.getGoal();

  switch (settings_.planner_type) {
    case rmpcpp::RMP:
      planner_ = std::make_unique<rmpcpp::PlannerRMP<rmpcpp::Space<3>>>(
          parameters_.plannerParameters);
      break;
    default:
      throw std::runtime_error("Not implemented");
  }

  /**
   * We set both ESDF and TSDF. Note that the RMP planner does not actually use
   * the ESDF.
   */
  planner_->setEsdf(worldgen_.getEsdfLayer());
  planner_->setTsdf(worldgen_.getTsdfLayer());

  // Set the policies
  rmpcpp::ParametersRMP parameters = this->parameters_;

  /** Target Policy */
  std::shared_ptr<rmpcpp::SimpleTargetPolicy<Space>> target_policy =
      std::make_shared<rmpcpp::SimpleTargetPolicy<Space>>(
        goalv, Eigen::Matrix<double, Space::dim, Space::dim>::Identity(),
        parameters.targetPolicyParameters.alpha, parameters.targetPolicyParameters.beta, parameters.targetPolicyParameters.c_softmax
        );

  /** 'World Policy' */
  std::shared_ptr<rmpcpp::PolicyBase<Space>> world_policy;
  switch (this->parameters_.policy_type) {
    case rmpcpp::SIMPLE_ESDF:
      world_policy = std::make_shared<rmpcpp::SimpleEsdfPolicy<Space>>(
          this->parameters_.esdfPolicyParameters, worldgen_.getEsdfLayer().get());
      break;
    case rmpcpp::RAYCASTING_CUDA:
      world_policy = std::make_shared<rmpcpp::RaycastingCudaPolicy<Space>>(
          this->parameters_.rayCastingCudaPolicyParameters, worldgen_.getTsdfLayer().get());
      break;
  }

  planner_->addPolicy(target_policy);
  planner_->addPolicy(world_policy);

  rmpcpp::State<dim> start(startv, {0.0, 0.0, 0.0});
  planner_->setup(start, goalv);

  auto starttime = std::chrono::high_resolution_clock::now();
  planner_->step(-1);
  auto endtime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      endtime - starttime);
  double duration_s = double(duration.count()) / 1E6;

  updateStats(run_index, worldgen_.getDensity(), duration_s);

  std::string success = planner_->success() ? "Success: " : "Failure: ";
  std::cout << success << double(duration.count()) / 1000.0 << "ms"
            << std::endl;
}

void Tester::exportTrajectories(std::string path, const int i) {
  path.append("trajectory");

  boost::format index = boost::format("%03d") % std::to_string(i);
  path.append(index.str());
  std::string endpoints_path = path;
  path.append(".txt");
  std::ofstream file;
  file.open(path, std::ofstream::trunc);  // clear file contents with trunc
  planner_->getTrajectory()->writeToStream(file);
  file.close();

  /** Export start and goal */
  endpoints_path.append("endpoints.txt");
  Eigen::IOFormat format(Eigen::FullPrecision, Eigen::DontAlignCols, " ", " ",
                         "", "", " ", "");
  file.open(endpoints_path, std::ofstream::trunc);
  file << "sx sy sz ex ey ez" << std::endl;
  file << worldgen_.getStart().format(format)
       << worldgen_.getGoal().format(format) << std::endl;
  file.close();
}

void Tester::exportWorld(std::string path, const int i) {
  path.append("world");

  switch (settings_.world_type) {
    case rmpcpp::SPHERES_BOX_WORLD:
      path.append("sb");
      // fallthrough
    case rmpcpp::SPHERES_ONLY_WORLD:
      path.append((boost::format("%03d") % std::to_string(i)).str());
      break;
    case rmpcpp::CUSTOM_WORLD:
      /** Custom world stays the same, so only export the first time */
      if (i) {
        return;
      }
      break;
  }

  path.append(".ply");
  worldgen_.exportToPly(path);
}

void Tester::run() {
  statistics_.reset(getMapName());

  for (size_t run_index = 0; run_index < settings_.n_runs; run_index++) {
    runSingle(run_index);

    /** Export trajectories only if enabled */
    if (!settings_.stats_only) {
      exportTrajectories(settings_.data_path, run_index);
    }

    /** Export world only if enabled */
    if (!settings_.stats_only and !settings_.world_save_path.empty()) {
      exportWorld(settings_.world_save_path, run_index);
    }
    settings_.seed++;
  }
  exportStats(settings_.data_path);
}

/**
 * Export statistics to file
 * @param path
 */
void Tester::exportStats(std::string path) {
  std::ofstream f_stats;
  f_stats.open(path + "stats.txt",
               std::ofstream::trunc);  // clear file contents with trunc
  statistics_.writeSummary(f_stats);
  f_stats.close();

  std::ofstream f_stats_full;
  f_stats_full.open(path + "stats_full.txt");
  statistics_.writeLines(f_stats_full);
  f_stats_full.close();
}

std::string Tester::getMapName() {
  std::stringstream map_name;
  switch (settings_.world_type) {
    case rmpcpp::SPHERES_ONLY_WORLD:
    case rmpcpp::SPHERES_BOX_WORLD:
      map_name << settings_.obstacles;
      break;
    case rmpcpp::CUSTOM_WORLD:
      /** In case of a custom world, 'obstacles' is just the world name */
      map_name << settings_.world_load_path.substr(
          settings_.world_load_path.find_last_of("/\\") + 1);
      break;
  }
  return map_name.str();
}

void Tester::updateStats(int index, double map_density, double duration_s) {
  bool success = planner_->success();
  auto trajectory = planner_->getTrajectory();
  rmpcpp::RunStatistics::Line stat_line;

  stat_line.success = success;
  stat_line.index = index;
  stat_line.world_density = map_density;
  stat_line.time_sec = duration_s;

  if (trajectory) {
    stat_line.integration_steps = trajectory->getSegmentCount();
    stat_line.integration_steps_per_sec =
        trajectory->getSegmentCount() / duration_s;
    stat_line.length = trajectory->getLength();
    stat_line.smoothness = trajectory->getSmoothness();
  }
  statistics_.add(stat_line);
}

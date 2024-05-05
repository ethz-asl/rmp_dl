#include "pybind11/eigen.h"
#include "rmpcpp_planner/testing/settings.h"
#include "rmpcpp_planner/testing/worldgen.h"
#include "torch/extension.h"

using namespace pybind11::literals;  // For the _a shorthand notation

PYBIND11_MODULE(worldgenBindings, m) {
  py::enum_<rmpcpp::WorldType>(m, "WorldType")
      .value("SPHERES_ONLY_WORLD", rmpcpp::WorldType::SPHERES_ONLY_WORLD)
      .value("SPHERES_BOX_WORLD", rmpcpp::WorldType::SPHERES_BOX_WORLD);

  py::class_<rmpcpp::WorldGenSettings,
             std::shared_ptr<rmpcpp::WorldGenSettings>>(m,
                                                        "WorldGenSettingsCPP")
      // We create a new constructor that takes all members of the struct as
      // arguments
      .def(py::init([](int seed,
                       std::pair<Eigen::Vector3f, Eigen::Vector3f> world_limits,
                       float voxel_size, float voxel_truncation_distance_vox,
                       Eigen::Vector3d startpos, Eigen::Vector3d goal,
                       rmpcpp::WorldType world_type) {
             rmpcpp::WorldGenSettings instance;
             instance.seed = seed;
             instance.world_limits = world_limits;
             instance.voxel_size = voxel_size;
             instance.block_size = nvblox::VoxelBlock<bool>::kVoxelsPerSide * voxel_size;
             instance.voxel_truncation_distance_vox = voxel_truncation_distance_vox;
             instance.startpos = startpos;
             instance.goal = goal;
             instance.world_type = world_type;

             return instance;
           }),
           // Shorthand for exposing the arguments as kwargs in python, see:
           // https://pybind11.readthedocs.io/en/stable/basics.html#keyword-args
           "seed"_a, "world_limits"_a, "voxel_size"_a,
           "voxel_truncation_distance_vox"_a, "startpos"_a, "goal"_a,
           "world_type"_a);

  py::class_<rmpcpp::WorldGen>(m, "WorldGenCPP")
      .def(py::init<rmpcpp::WorldGenSettings>(), py::keep_alive<1, 2>())
      .def("getTsdfLayer", &rmpcpp::WorldGen::getTsdfLayer)
      .def("getEsdfLayer", &rmpcpp::WorldGen::getEsdfLayer)
      .def("getDensity", &rmpcpp::WorldGen::getDensity)
      .def("generateRandomWorld", &rmpcpp::WorldGen::generateRandomWorld,
           "n"_a,                        // Has no default
           "r"_a = 1.0, "r_std"_a = 0.2, "p_sphere"_a = 0.5, "margin"_a = 0.5  // Expose default arguments
           )
      .def("seed", &rmpcpp::WorldGen::seed)
      // To support libtorch versions that are compiled with the pre cxx11 ABI
      // we use c-style strings in the interface. E.g. all pip wheels use the pre
      // cxx11 ABI, see issue here:
      // https://github.com/pytorch/pytorch/issues/51039
      .def_static("exportToPly", &rmpcpp::WorldGen::exportToPlyC_STRING)
      .def_static("generate_esdf_from_tsdf", &rmpcpp::WorldGen::generateEsdfFromTsdf)
      .def("reset", &rmpcpp::WorldGen::reset);
}

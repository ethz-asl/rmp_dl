#include "pybind11/eigen.h"
#include "pybind11/stl.h"
#include "rmpcpp/core/state.h"
#include "rmpcpp_planner/core/planner_base.h"
#include "rmpcpp_planner/core/planner_rmp.h"
#include "rmpcpp_planner/core/parameters.h"
#include "rmpcpp_planner/core/trajectory_rmp.h"
#include "rmpcpp_planner/core/world.h"
#include "rmpcpp_planner/testing/settings.h"
#include "rmpcpp_planner/testing/worldgen.h"
#include "torch/extension.h"

using namespace pybind11::literals;

PYBIND11_MODULE(rmpPlannerBindings, m){

    py::class_<rmpcpp::PlannerParameters, 
        std::shared_ptr<rmpcpp::PlannerParameters>>(m, "PlannerParametersCpp")
        .def(
            py::init(
            [](double dt, int max_length, double truncation_distance_vox, 
            bool terminate_upon_goal_reached
            ) {
                rmpcpp::PlannerParameters instance;
                instance.dt = dt;
                instance.max_length = max_length;
                instance.truncation_distance_vox = truncation_distance_vox;
                instance.terminate_upon_goal_reached = terminate_upon_goal_reached;
                return instance;
            }),
            // Shorthand for exposing the arguments as kwargs in python, see:
            // https://pybind11.readthedocs.io/en/stable/basics.html#keyword-args
            "dt"_a, "max_length"_a, "truncation_distance_vox"_a, "terminate_upon_goal_reached"_a
        );

    py::class_<rmpcpp::PlannerRMP<rmpcpp::Space<3>>>(m, "PlannerRmpCpp")
        .def(py::init<rmpcpp::PlannerParameters& >(), 
            py::keep_alive<1, 2>()) // Otherwise the PlannerParameters object is deleted
        .def("setup",  // We use a lambda to initialize the start_velocity as 0 by default if left out
            [](rmpcpp::PlannerRMP<rmpcpp::Space<3>>& instance, 
                Eigen::Vector3d start, Eigen::Vector3d goal, 
                Eigen::Vector3d start_velocity) {
                instance.setup(rmpcpp::State<3>(start, start_velocity), goal);
            },
            "start"_a, "goal"_a, "start_velocity"_a = Eigen::Vector3d::Zero()
        )
        .def("step", &rmpcpp::PlannerRMP<rmpcpp::Space<3>>::step, 
            "steps"_a)
        .def("add_policy", &rmpcpp::PlannerRMP<rmpcpp::Space<3>>::addPolicy, 
            py::keep_alive<1, 2>()) // We need this to keep python managed objects alive, e.g. in case the policy is a derived python class
        .def("get_trajectory", 
            [](rmpcpp::PlannerRMP<rmpcpp::Space<3>>& instance) {
                const rmpcpp::TrajectoryRMP<rmpcpp::Space<3>>& trajectory = *instance.getTrajectory();
                
                size_t size = trajectory.getSize();
                std::vector<Eigen::Vector3d> positions; positions.reserve(size);
                std::vector<Eigen::Vector3d> velocities; velocities.reserve(size);
                std::vector<Eigen::Vector3d> accelerations; accelerations.reserve(size);
                
                for(int i = 0; i < size; i++){
                    rmpcpp::TrajectoryPointRMP<rmpcpp::Space<3>> point = trajectory[i];
                    positions.push_back(point.position);
                    velocities.push_back(point.velocity);
                    accelerations.push_back(point.acceleration);
                }
                return py::make_tuple(positions, velocities, accelerations);
            })
        .def("set_tsdf", &rmpcpp::PlannerBase<rmpcpp::Space<3>>::setTsdf)
        .def("set_esdf", &rmpcpp::PlannerBase<rmpcpp::Space<3>>::setEsdf)
        .def("success", &rmpcpp::PlannerBase<rmpcpp::Space<3>>::success)
        .def("collided", &rmpcpp::PlannerBase<rmpcpp::Space<3>>::collided)
        .def("diverged", &rmpcpp::PlannerBase<rmpcpp::Space<3>>::diverged)
        .def("get_pos", &rmpcpp::PlannerRMP<rmpcpp::Space<3>>::getPos)
        .def("get_vel", &rmpcpp::PlannerRMP<rmpcpp::Space<3>>::getVel)
        .def("get_acc", &rmpcpp::PlannerRMP<rmpcpp::Space<3>>::getAcc)
        .def("get_previous_pos", &rmpcpp::PlannerRMP<rmpcpp::Space<3>>::getPreviousPos)
        .def("get_previous_vel", &rmpcpp::PlannerRMP<rmpcpp::Space<3>>::getPreviousVel)
        .def("get_previous_acc", &rmpcpp::PlannerRMP<rmpcpp::Space<3>>::getPreviousAcc)
        ;

    using Vector = rmpcpp::NVBloxWorld<rmpcpp::Space<3>>::Vector;
    py::class_<rmpcpp::NVBloxWorld<rmpcpp::Space<3>>>(m, "NVBloxWorld")
        .def_static("check_motion", 
            py::overload_cast<const Vector&,  const Vector&, nvblox::TsdfLayer::Ptr>(&rmpcpp::NVBloxWorld<rmpcpp::Space<3>>::checkMotion))
        .def_static("collision", 
            py::overload_cast<const Vector&, nvblox::TsdfLayer::Ptr>(&rmpcpp::NVBloxWorld<rmpcpp::Space<3>>::collision))
        .def_static("collision_with_unobserved_invalid", 
            py::overload_cast<const Vector&, nvblox::TsdfLayer::Ptr, const float>(&rmpcpp::NVBloxWorld<rmpcpp::Space<3>>::collisionWithUnobservedInvalid), 
            "pos"_a, "layer"_a, "margin"_a=0.0f);
}


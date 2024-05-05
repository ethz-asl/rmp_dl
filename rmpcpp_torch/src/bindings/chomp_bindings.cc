#include "comparison_planners/chomp/chomp_optimizer.h"
#include "pybind11/eigen.h"
#include "torch/extension.h"

#include "comparison_planners/chomp/planner_chomp.h"

using namespace pybind11::literals;

PYBIND11_MODULE(chompBindings, m) {  
    py::class_<chomp::ChompParameters>(m, "ChompParameters")
        .def(py::init<>())
        
        .def_readwrite("D", &chomp::ChompParameters::D)
        .def_readwrite("w_smooth", &chomp::ChompParameters::w_smooth)
        .def_readwrite("w_collision", &chomp::ChompParameters::w_collision)
        .def_readwrite("epsilon", &chomp::ChompParameters::epsilon)
        .def_readwrite("lmbda", &chomp::ChompParameters::lambda)
        .def_readwrite("rel_tol", &chomp::ChompParameters::rel_tol)
        .def_readwrite("max_iter", &chomp::ChompParameters::max_iter)
        .def_readwrite("decrease_step_size", &chomp::ChompParameters::decrease_step_size)
        .def_readwrite("map_resolution", &chomp::ChompParameters::map_resolution)
        .def_readwrite("verbose", &chomp::ChompParameters::verbose);

    py::class_<PlannerChomp<rmpcpp::Space<3>>>(m, "PlannerChomp")
        .def(py::init<chomp::ChompParameters, int, bool>(), 
            "parameters"_a, "N"_a, "cpu"_a = true)
        .def("plan", &PlannerChomp<rmpcpp::Space<3>>::plan, 
            "start"_a, "goal"_a)
        // .def("plan", [](PlannerChomp<rmpcpp::Space<3>>& instance, 
        //         Eigen::Vector3d start, Eigen::Vector3d goal) {
        //         instance.plan(rmpcpp::State<3>(start), rmpcpp::State<3>(goal));
        //     },
        //     "start"_a, "goal"_a)
        .def("get_path", &PlannerChomp<rmpcpp::Space<3>>::getPath)
        .def("set_tsdf", &PlannerChomp<rmpcpp::Space<3>>::setTsdf)
        .def("set_esdf", &PlannerChomp<rmpcpp::Space<3>>::setEsdf)
        .def("distance_to_obstacle", &PlannerChomp<rmpcpp::Space<3>>::distanceToObstacle)
        .def("gradient_to_obstacle", &PlannerChomp<rmpcpp::Space<3>>::gradientToObstacle)
        .def("success", &PlannerChomp<rmpcpp::Space<3>>::success)
        ;
}

#include "pybind11/eigen.h"
#include "torch/extension.h"

#include "comparison_planners/rrt/parameters_rrt.h"
#include "comparison_planners/rrt/planner_rrt.h"

using namespace pybind11::literals;

PYBIND11_MODULE(rrtBindings, m) {  
    py::class_<RRT::RRTParameters>(m, "RRTParameters")
        .def(py::init<>())

        .def_readwrite("type", &RRT::RRTParameters::type)
        .def_readwrite("stateValidityCheckerResolution", &RRT::RRTParameters::stateValidityCheckerResolution_m)
        .def_readwrite("range", &RRT::RRTParameters::range)
        .def_readwrite("margin_to_obstacles", &RRT::RRTParameters::margin_to_obstacles)
        ;

    py::class_<PlannerRRT<rmpcpp::Space<3>>>(m, "PlannerRRT")
        .def(py::init<RRT::RRTParameters&>(), 
            "parameters"_a=RRT::RRTParameters())
        .def("plan", &PlannerRRT<rmpcpp::Space<3>>::plan, 
            "start"_a, "goal"_a, "time"_a)
        .def("get_path", &PlannerRRT<rmpcpp::Space<3>>::getPath)
        .def("set_esdf", &PlannerRRT<rmpcpp::Space<3>>::setEsdf)
        .def("success", &PlannerRRT<rmpcpp::Space<3>>::success)
        ;
}


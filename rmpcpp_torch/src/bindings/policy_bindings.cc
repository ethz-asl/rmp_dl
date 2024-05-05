#include <stdexcept>
#include "nvblox/nvblox.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
#include "rmpcpp/core/policy_base.h"
#include "rmpcpp/core/policy_value.h"
#include "rmpcpp/policies/simple_target_policy.h"
#include "rmpcpp_planner/core/parameters.h"
#include "rmpcpp_planner/policies/raycasting_CUDA.h"
#include "rmpcpp_planner/policies/simple_ESDF.h"
#include "torch/extension.h"

using namespace pybind11::literals;
using Space = rmpcpp::Space<3>;


/**
 * See https://pybind11.readthedocs.io/en/stable/advanced/classes.html#combining-virtual-functions-and-inheritance 
 * for documentation on the templated trampolines defined below. 
 */

/**
 * @brief Helper 'trampoline' class such that we can create python policies that
 * inherit from policybase
 */
template<class PolicyBaseTemplate = rmpcpp::PolicyBase<Space>>
class PyPolicyBase : public PolicyBaseTemplate {
public:
    using PolicyBaseTemplate::PolicyBase;
    using PValue = typename PolicyBaseTemplate::PValue;
    using PState = typename PolicyBaseTemplate::PState;

    PValue evaluateAt(const PState& state) override {
        PYBIND11_OVERRIDE_PURE_NAME(
            PValue, // return type
            PolicyBaseTemplate, // parent class
            "evaluate_at", // Name of function in python
            evaluateAt, // Name of function in c++
            state // Argument
        );
    }
};

template<class RaycastingCudaPolicyTemplate = rmpcpp::RaycastingCudaPolicy<Space>>
class PyRaycastingCudaPolicy : public RaycastingCudaPolicyTemplate { 
    using RaycastingCudaPolicyTemplate::RaycastingCudaPolicy;
    using PValue = typename RaycastingCudaPolicyTemplate::PValue;
    using PState = typename RaycastingCudaPolicyTemplate::PState;
    
    PValue evaluateAt(const PState& state) override {
        PYBIND11_OVERRIDE_NAME( // Note the lack of _PURE_ in this case
            PValue, // return type
            RaycastingCudaPolicyTemplate, // parent class
            "evaluate_at", // Name of function in python
            evaluateAt, // Name of function in c++
            state // Argument
        );
    }
};

PYBIND11_MODULE(policyBindings, m){
    py::class_<rmpcpp::PolicyValue<3>>(m, "PolicyValue")
        .def(py::init<rmpcpp::PolicyValue<3>::Vector, rmpcpp::PolicyValue<3>::Matrix>())
        .def_readonly("A_", &rmpcpp::PolicyValue<3>::A_)
        .def_readonly("f_", &rmpcpp::PolicyValue<3>::f_)
        .def_static("sum", &rmpcpp::PolicyValue<3>::sum);

    py::class_<rmpcpp::State<3>>(m, "State")
        .def(py::init<rmpcpp::State<3>::Vector, rmpcpp::State<3>::Vector>())
        .def_readwrite("pos", &rmpcpp::State<3>::pos_)
        .def_readwrite("vel", &rmpcpp::State<3>::vel_);

    py::class_<
        rmpcpp::PolicyBase<Space>,
        PyPolicyBase<>, // Trampoline 
        std::shared_ptr<rmpcpp::PolicyBase<Space>> // Holder type
        >(m, "PolicyBase")
        .def(py::init<>())
        .def("evaluate_at", &rmpcpp::PolicyBase<Space>::evaluateAt);

    py::class_<
        rmpcpp::RaycastingCudaPolicy<Space>, 
        rmpcpp::PolicyBase<Space>, // Inherits from 
        PyRaycastingCudaPolicy<>, // Trampoline
        std::shared_ptr<rmpcpp::RaycastingCudaPolicy<Space>> // Holder type
        >(m, "RaycastingCudaPolicy")
        .def(py::init(
            [](nvblox::TsdfLayer* tsdf, 
                double eta_rep, double eta_damp, double v_rep, double v_damp,
                double epsilon_damp, double c_softmax_obstacle, double r,
                bool metric, int N_sqrt, double surface_distance_epsilon_vox, int max_steps, 
                double truncation_distance_vox, double metric_scale, double force_scale
                // TODO: Pass on metric scale and force scale
            ) {
                rmpcpp::RaycastingCudaPolicyParameters params;
                params.eta_rep = eta_rep;
                params.eta_damp = eta_damp;
                params.v_rep = v_rep;
                params.v_damp = v_damp;
                params.epsilon_damp = epsilon_damp;
                params.c_softmax_obstacle = c_softmax_obstacle;
                params.r = r;
                params.metric = metric;
                params.N_sqrt = N_sqrt;
                params.surface_distance_epsilon_vox = surface_distance_epsilon_vox;
                params.max_steps = max_steps;
                params.truncation_distance_vox = truncation_distance_vox;
                return std::make_shared<rmpcpp::RaycastingCudaPolicy<Space>>(params, tsdf);
            }), py::keep_alive<1, 2>(), // Keep tsdf alive as long as the policy is alive
            "tsdf"_a, "eta_rep"_a, "eta_damp"_a, "v_rep"_a, "v_damp"_a,
            "epsilon_damp"_a, "c_softmax_obstacle"_a, "r"_a, "metric"_a, "N_sqrt"_a,
            "surface_distance_epsilon_vox"_a, "max_steps"_a, "truncation_distance_vox"_a, 
            "metric_scale"_a=1024.0, "force_scale"_a=1.0
            )
        .def("evaluate_at", &rmpcpp::RaycastingCudaPolicy<Space>::evaluateAt);

    py::class_<
        rmpcpp::SimpleEsdfPolicy<Space>, 
        rmpcpp::PolicyBase<Space>, // Inherits from 
        std::shared_ptr<rmpcpp::SimpleEsdfPolicy<Space>> // Holder type
        >(
        m, "SimpleEsdfPolicy")
        .def(py::init(
            [](nvblox::TsdfLayer* esdf,
                double eta_rep, double eta_damp, double v_rep, double v_damp,
                double epsilon_damp, double c_softmax_obstacle, double r
            ) {
                throw std::runtime_error("Need to fix the esdf policy to take a tsdf layer with high truncation distance instead.\
                    This is because the esdf layer is less accurate and uses voxel distance instead of distance in meters. ");
                rmpcpp::EsdfPolicyParameters params;
                params.eta_rep = eta_rep;
                params.eta_damp = eta_damp;
                params.v_rep = v_rep;
                params.v_damp = v_damp;
                params.epsilon_damp = epsilon_damp;
                params.c_softmax_obstacle = c_softmax_obstacle;
                params.r = r;
                return std::make_shared<rmpcpp::SimpleEsdfPolicy<Space>>(params, esdf);
            }),          
            // Shorthand for exposing the arguments as kwargs in python, see:
            // https://pybind11.readthedocs.io/en/stable/basics.html#keyword-args
            "esdf"_a, "eta_rep"_a, "eta_damp"_a, "v_rep"_a, "v_damp"_a, "epsilon_damp"_a, "c_softmax_obstacle"_a, "r"_a
        );
    py::class_<
        rmpcpp::SimpleTargetPolicy<Space>, 
        rmpcpp::PolicyBase<Space>, // Inherits from
        std::shared_ptr<rmpcpp::SimpleTargetPolicy<Space>> // Holder type
        >(
        m, "SimpleTargetPolicy")
        .def(py::init<Eigen::Vector3d, double, double, double, double>(),
            // Shorthand for exposing the arguments as kwargs in python, see:
            // https://pybind11.readthedocs.io/en/stable/basics.html#keyword-args
            "target"_a, "alpha"_a, "beta"_a, "c_softmax"_a, "metric"_a);
}
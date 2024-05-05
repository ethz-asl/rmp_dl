#include "torch/extension.h"
#include "rmpcpp_torch/kernels/get_rays.h"

#include <pybind11/eigen.h>

using namespace pybind11::literals;

PYBIND11_MODULE(kernelBindings, m) {
    m.def("get_rays", &getRays,
        "raycasted_depth"_a,
        "origin"_a, 
        "layer"_a,
        "N_sqrt"_a, 
        "truncation_distance_vox"_a, 
        "maximum_steps"_a,
        "maximum_ray_length"_a, 
        "surface_distance_epsilon_vox"_a
        );
}


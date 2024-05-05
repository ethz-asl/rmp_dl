#include <map/common_names.h>
#include <pybind11/detail/common.h>
#include "pybind11/eigen.h"
#include "rmpcpp_planner/testing/settings.h"
#include "rmpcpp_planner/testing/worldgen.h"
#include "torch/extension.h"

#include "nvblox/nvblox.h"
#include "nvblox/executables/fuser.h"
#include "nvblox/datasets/3dmatch.h" 

using namespace pybind11::literals;  // For the _a shorthand notation

// Forward declare the c_str wrapper
std::unique_ptr<nvblox::Fuser> createFuser(const char* base_path_c_str, 
                          const char* timing_output_path, 
                          const char* map_output_path, 
                          const char* mesh_output_path);

void set_voxel_size_(const float voxel_size, nvblox::Fuser* fuser);
void set_truncation_distance_vox_(const float truncation_distance_vox, nvblox::Fuser* fuser);

class Fuse3dMatchWorldgen {
  public:
    Fuse3dMatchWorldgen(const char* base_path_c_str, const char* timing_output_path, const char* map_output_path, const char* mesh_output_path) {
      this->fuser = createFuser(base_path_c_str, timing_output_path, map_output_path, mesh_output_path);
    }

    void run(){
      this->fuser->run();
    }

    nvblox::TsdfLayer::Ptr get_tsdf_layer() {
      return std::make_shared<nvblox::TsdfLayer>(this->fuser->mapper().tsdf_layer());
    }

    void set_voxel_size(const float voxel_size) {
      set_voxel_size_(voxel_size, this->fuser.get());
    }

    void set_truncation_distance_vox(const float truncation_distance_vox) {
      set_truncation_distance_vox_(truncation_distance_vox, this->fuser.get());
    }

  private: 
    std::unique_ptr<nvblox::Fuser> fuser;
};

PYBIND11_MODULE(fuse3DMatchBindings, m) {
  py::class_<Fuse3dMatchWorldgen>(m, "Fuse3DMatchWorldgen")
    .def(py::init<const char*, const char*, const char*, const char*>(), 
         "base_path"_a, "timing_output_path"_a, "map_output_path"_a, "mesh_output_path"_a)
    .def("run", &Fuse3dMatchWorldgen::run)
    .def("get_tsdf_layer", &Fuse3dMatchWorldgen::get_tsdf_layer)
    .def("set_voxel_size", &Fuse3dMatchWorldgen::set_voxel_size)
    .def("set_truncation_distance_vox", &Fuse3dMatchWorldgen::set_truncation_distance_vox);
};

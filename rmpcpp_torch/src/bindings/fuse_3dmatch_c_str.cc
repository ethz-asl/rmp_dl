

#include "nvblox/nvblox.h"
#include "nvblox/executables/fuser.h"
#include "nvblox/datasets/3dmatch.h"

// See comments in nvblox_tunnel.cc for why this is necessary (in this case the Fuse3DMatch constructor contains a std::string) 

std::unique_ptr<nvblox::Fuser> createFuser(const char* base_path_c_str, 
                                           const char* timing_output_path, 
                                           const char* map_output_path,
                                           const char* mesh_output_path) {
    std::unique_ptr<nvblox::Fuser> fuser = nvblox::datasets::threedmatch::createFuser(std::string(base_path_c_str), 1);
    fuser->timing_output_path_ = std::string(timing_output_path);
    fuser->map_output_path_ = std::string(map_output_path);
    fuser->mesh_output_path_ = std::string(mesh_output_path);
    return fuser;
}

// Also run into linker errors if these are not defined here
void set_voxel_size_(const float voxel_size, nvblox::Fuser* fuser){
    fuser->setVoxelSize(voxel_size);
}

void set_truncation_distance_vox_(const float truncation_distance_vox, nvblox::Fuser* fuser){
    fuser->mapper().tsdf_integrator().truncation_distance_vox(truncation_distance_vox);
}


#include "nvblox/nvblox.h"


/**
 * We can't bind the generateSdfFromScene method on the Scene class, 
 * as the linker needs to link against symbols that contain std::string
 * (specifically, the nvblox::timing::TimerNvtx::TimerNvtx constructor).
 * As a result it will be incompatible with pre-cxx11 ABI compiled libraries, such as pytorch 
 * installed via pip, which contains the pybind library that we use, so we have to also link against 
 * that library. We can't link against 2 libraries with different ABI's so we define this 'tunnel'
 * library, which hides the symbols containing std::string. 
 */
void generateTsdfFromScene(nvblox::primitives::Scene& scene, float max_dist, nvblox::TsdfLayer* layer) {
    scene.generateLayerFromScene(max_dist, layer);
}

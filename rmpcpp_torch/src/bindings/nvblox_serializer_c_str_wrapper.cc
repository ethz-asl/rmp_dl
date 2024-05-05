#include <map/layer_cake.h>
#include "nvblox/nvblox.h"
#include "nvblox/serialization/internal/serializer.h"

nvblox::TsdfLayer::Ptr loadTsdfFromFile_cstr(const char* filename) {
    nvblox::LayerCake layerCake = nvblox::io::loadLayerCakeFromFile(std::string(filename), nvblox::MemoryType::kUnified);
    if(layerCake.empty()){
        return nullptr; // This is converted to None in python
    }   
    return std::make_shared<nvblox::TsdfLayer>(layerCake.get<nvblox::TsdfLayer>());
}

bool writeLayerCakeToFile_cstr(const char* filename, nvblox::TsdfLayer::Ptr layer) {
    nvblox::LayerCake layerCake = nvblox::LayerCake(layer->voxel_size());
    nvblox::TsdfLayer* tsdf_ptr = layerCake.add<nvblox::TsdfLayer>(nvblox::MemoryType::kUnified);
    if(tsdf_ptr == nullptr){
        return false;
    }
    *tsdf_ptr = *layer;

    return nvblox::io::writeLayerCakeToFile(std::string(filename), layerCake);
}


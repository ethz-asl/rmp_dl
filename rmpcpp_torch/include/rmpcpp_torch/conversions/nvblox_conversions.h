
#include <map/layer.h>
#include "nvblox/nvblox.h"

#include <boost/multi_array.hpp>
#include <limits>

#include <math.h>

class NVBloxConverter{
public:
    template<typename T>
    using Array3d = boost::multi_array<T, 3>;
    

    /** Same as below but with identity conversion */
    template<typename VoxelType>
    static Array3d<VoxelType> toMultiArray(const typename nvblox::VoxelBlockLayer<VoxelType>::Ptr layer){
        return toMultiArray<VoxelType, VoxelType>(layer,
                                                [](VoxelType v) { return v; } /** identity lambda */ );
    }

    /**
     * Convert an nvblox layer to 3d CPU vectors. The conversion function can be passed to convert voxels
     * to any arbitrary output type
     * @tparam VoxelType Input voxel type
     * @tparam OutputType Output type
     * @param layer Voxel layer of $VoxelType
     * @param conversionFunction Function that converts from $VoxelType to $OutputType
     * @return
     */
    template<typename VoxelType, typename OutputType>
    static Array3d<OutputType> toMultiArray(
            const typename nvblox::VoxelBlockLayer<VoxelType>::Ptr layer,
            const std::function<OutputType(VoxelType)> conversionFunction){
        const std::vector<Eigen::Vector3i> indices = layer->getAllBlockIndices();
    
        constexpr int voxelsPerSide = nvblox::VoxelBlock<VoxelType>::kVoxelsPerSide;

        const std::vector<int> max_indices = getMaxIndices(indices, voxelsPerSide);
        const std::vector<int> min_indices = getMinIndices(indices, voxelsPerSide);

        Array3d<OutputType> result(std::vector<int>{
            max_indices[0] - min_indices[0], 
            max_indices[1] - min_indices[1],
            max_indices[2] - min_indices[2]
            });

        for(auto& index: indices){
            typename nvblox::VoxelBlock<VoxelType>::Ptr block = layer->getBlockAtIndex(index);
            for(int i = 0; i < voxelsPerSide; i++){
                for(int j = 0; j < voxelsPerSide; j++) {
                    for (int k = 0; k < voxelsPerSide; k++) {
                        const int ix = index[0] * voxelsPerSide + i - min_indices[0];
                        const int iy = index[1] * voxelsPerSide + j - min_indices[1];
                        const int iz = index[2] * voxelsPerSide + k - min_indices[2];
                        result[ix][iy][iz] =
                                conversionFunction(block->voxels[i][j][k]);
                    }
                }
            }
        }

        return result;
    }


    /**
     * @brief Get the minimum indices
     * 
     * @param indices 
     * @return std::vector<int> 
     */
    static std::vector<int> getMinIndices(const std::vector<Eigen::Vector3i>& indices, const int voxelsPerSide) {
        std::vector<int> min_indices = std::vector({std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max()});
        for(auto& index: indices){
          min_indices[0] = std::min(min_indices[0], index[0] * voxelsPerSide);
          min_indices[1] = std::min(min_indices[1], index[1] * voxelsPerSide);
          min_indices[2] = std::min(min_indices[2], index[2] * voxelsPerSide);
        }
        return min_indices;
    }


    /**
     * @brief Get the maximum indices
     * 
     * @param indices 
     * @return std::vector<int> 
     */
    static std::vector<int> getMaxIndices(const std::vector<Eigen::Vector3i>& indices, const int voxelsPerSide) {
        std::vector<int> max_indices = std::vector({std::numeric_limits<int>::min(), std::numeric_limits<int>::min(), std::numeric_limits<int>::min()});
        for(auto& index: indices){
          max_indices[0] = std::max(max_indices[0], (index[0] + 1) * voxelsPerSide);
          max_indices[1] = std::max(max_indices[1], (index[1] + 1) * voxelsPerSide);
          max_indices[2] = std::max(max_indices[2], (index[2] + 1) * voxelsPerSide);
        }
        return max_indices;
    }



    static void convertEsdfToTsdf(
        const typename nvblox::EsdfLayer::Ptr input, 
        typename nvblox::TsdfLayer::Ptr output){
        const float voxel_size = input->voxel_size();
        convertVoxelLayer<nvblox::EsdfVoxel, nvblox::TsdfVoxel>(input, output, 
            [voxel_size](const nvblox::EsdfVoxel& voxel){
                nvblox::TsdfVoxel tsdf_voxel;
                // ESDF uses squared voxel distance. 
                // We convert to meters by taking the square root and multiplying with voxel size.
                auto d = sqrt(voxel.squared_distance_vox) * voxel_size;
                tsdf_voxel.distance = voxel.is_inside ? -d : d;
                tsdf_voxel.weight = voxel.observed ? 1. : 0.;
                
                return tsdf_voxel;
            });
    }

    template<typename InputVoxelType, typename OutputVoxelType>
    static void convertVoxelLayer(
        const typename nvblox::VoxelBlockLayer<InputVoxelType>::Ptr input, 
        typename nvblox::VoxelBlockLayer<OutputVoxelType>::Ptr output, 
        const std::function<OutputVoxelType(InputVoxelType)> conversionFunction){

        const std::vector<Eigen::Vector3i> indices = input->getAllBlockIndices();
        constexpr int voxelsPerSide = nvblox::VoxelBlock<InputVoxelType>::kVoxelsPerSide;

        for(auto& index: indices){
            typename nvblox::VoxelBlock<InputVoxelType>::Ptr block = input->getBlockAtIndex(index);
            typename nvblox::VoxelBlock<OutputVoxelType>::Ptr newBlock = output->allocateBlockAtIndex(index);
            for(int i = 0; i < voxelsPerSide; i++){
                for(int j = 0; j < voxelsPerSide; j++) {
                    for (int k = 0; k < voxelsPerSide; k++) {
                        newBlock->voxels[i][j][k] = conversionFunction(block->voxels[i][j][k]);
                    }
                }
            }
        }

    }

};


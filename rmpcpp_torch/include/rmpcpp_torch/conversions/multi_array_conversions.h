#ifndef CATKIN_PYBIND_TEST_MULTI_ARRAY_CONVERSIONS_H
#define CATKIN_PYBIND_TEST_MULTI_ARRAY_CONVERSIONS_H


#include <boost/multi_array.hpp>

class MultiArrayConversions{
 public:
  template<typename NumType, int dims>
  static std::vector<NumType> toVector(boost::multi_array<NumType, dims> arr){
    auto shape = arr.shape();
    int length = 1;
    for(int i = 0; i < dims; i++){
      length *= shape[i];
    }

    std::vector<NumType> result;
    result.resize(length);

    for(int i = 0; i < length; i++){
      result[i] = arr.data()[i];
    }

    return result;
  }
};

#endif  // CATKIN_PYBIND_TEST_MULTI_ARRAY_CONVERSIONS_H

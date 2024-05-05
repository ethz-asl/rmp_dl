#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>

#include "nvblox/nvblox.h"
#include "torch/extension.h"

using namespace pybind11::literals;

// Forward declare tunnel. See nvblox_tunnel.cc
void generateTsdfFromScene(nvblox::primitives::Scene&, float, nvblox::TsdfLayer*);


PYBIND11_MODULE(nvbloxSceneBindings, m) {
  py::class_<nvblox::primitives::Primitive>(m, "NvbloxPrimitive")
    .def("get_distance_to_point", &nvblox::primitives::Primitive::getDistanceToPoint);

  py::class_<nvblox::primitives::Sphere, 
    nvblox::primitives::Primitive>(m, "NvbloxSphere")
      .def(py::init<Eigen::Vector3f, float>(), 
           "center"_a, "radius"_a);

  py::class_<nvblox::primitives::Cube,
    nvblox::primitives::Primitive>(m, "NvbloxCube")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f>(), 
           "center"_a, "size"_a);

  py::class_<nvblox::primitives::Scene>(m, "NvbloxScene")
      .def(py::init<>())
      .def("clear", &nvblox::primitives::Scene::clear)

      .def("add_primitive",
           [](nvblox::primitives::Scene& instance,
              nvblox::primitives::Primitive& primitive) {
             // We need to copy the primitive, as we are not able to pass a
             // python managed object as a unique_ptr
             switch (primitive.getType()) {
               case nvblox::primitives::Primitive::Type::kPlane:
                 instance.addPrimitive(
                     std::move(std::make_unique<nvblox::primitives::Plane>(
                         static_cast<nvblox::primitives::Plane&>(primitive))));
                 break;
               case nvblox::primitives::Primitive::Type::kCube:
                 instance.addPrimitive(
                     std::move(std::make_unique<nvblox::primitives::Cube>(
                         static_cast<nvblox::primitives::Cube&>(primitive))));
                 break;
               case nvblox::primitives::Primitive::Type::kSphere:
                 instance.addPrimitive(
                     std::move(std::make_unique<nvblox::primitives::Sphere>(
                         static_cast<nvblox::primitives::Sphere&>(primitive))));
                 break;
               case nvblox::primitives::Primitive::Type::kCylinder:
                 instance.addPrimitive(
                     std::move(std::make_unique<nvblox::primitives::Cylinder>(
                         static_cast<nvblox::primitives::Cylinder&>(
                             primitive))));
                 break;
               default:
                 throw std::runtime_error("Unknown primitive type");
             }
           })
      .def("add_ground_level", &nvblox::primitives::Scene::addGroundLevel)
      .def("add_ceiling", &nvblox::primitives::Scene::addCeiling)
      .def("add_plane_boundaries",
           &nvblox::primitives::Scene::addPlaneBoundaries)
      .def("set_aabb", [](nvblox::primitives::Scene& instance,
                          Eigen::Vector3f min, Eigen::Vector3f max) {
        instance.aabb() = Eigen::AlignedBox3f(min, max);
      }) 
      /** CAN'T DO THIS, WILL RUN INTO LINKER ERRORS DUE TO CXX11ABI MIXING
      .def("generate_tsdf_from_scene",
           &nvblox::primitives::Scene::generateSdfFromScene<nvblox::TsdfVoxel>)
           */
      ;

  m.def("generate_tsdf_from_scene", &generateTsdfFromScene);
}
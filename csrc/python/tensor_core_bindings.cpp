#include "bindings.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../tensor_core/tensor.hpp"

namespace py = pybind11;

void init_tensor_module(py::module &mt) {
    // mt refers to the tensor submodule of the tinytorch module

    py::class_<tinytorch::Tensor>(mt, "Tensor")

        // init <-> constructor
        // NOTE: this is called from the python frontend which handles the python lists
        .def(py::init([](py::list data, std::vector<size_t> shape, tinytorch::Dtype dtype) {

            DISPATCH_DTYPE_INIT(dtype, data, shape);

            }),
            py::arg("data"),
            py::arg("shape"),
            py::arg("dtype") = tinytorch::Dtype::Float32,

            "Create tinytorch tensors from arbitrary python lists.")

        .def_property_readonly("shape", &tinytorch::Tensor::get_shape) // we do not want external modification of shape

        .def("size", [](const tinytorch::Tensor &t) { return t.get_shape(); })

        .def("size", [](const tinytorch::Tensor &t, int dim) {
            const auto& shape = t.get_shape();
            // python style negative indexing
            if (dim < 0) {
                dim += shape.size();
            }
            // size check
            if (dim < 0 || dim >= static_cast<int>(shape.size())) {
                throw std::out_of_range("Dimension out of range");
            }
            
            return shape[dim];
        }, py::arg("dim"))

        .def("__repr__", [](const tinytorch::Tensor &t) {
            return "<Tensor>";  // TODO: to be done properly
        });
}
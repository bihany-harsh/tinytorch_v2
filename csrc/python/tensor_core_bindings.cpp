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

        .def("__repr__", [](const tinytorch::Tensor &t) {
            return "<Tensor>";  // TODO: to be done properly
        });
}
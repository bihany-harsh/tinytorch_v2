#include "bindings.hpp"
#include "../tensor_core/tensor.hpp"

// mtt <- module tintorch
PYBIND11_MODULE(_core, mtt, py::mod_gil_not_used()) {

    py::enum_<tinytorch::Dtype>(mtt, "Dtype")
        .value("Float32", tinytorch::Dtype::Float32)
        .value("Float64", tinytorch::Dtype::Float64)
        .value("Int32", tinytorch::Dtype::Int32)
        .value("Int64", tinytorch::Dtype::Int64)
        .value("Bool", tinytorch::Dtype::Bool)
        .export_values();

    py::module tensor_module = mtt.def_submodule("tensor", "Tensor manipulations");

    init_tensor_module(tensor_module);
}
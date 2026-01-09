#include "tensor.hpp"
using namespace tinytorch;

void Tensor::set_stride() {
    stride.resize(shape.size());
    if (shape.empty()) return;

    stride[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}

void Tensor::fill_zeros() {
    if (!storage) {
        throw std::runtime_error("Storage not initialized (tensor.cpp)");
    }
    std::memset(storage->data_ptr(), 0, elem_count * dtype_size(dtype));
}
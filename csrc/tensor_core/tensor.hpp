#pragma once
#include "storage.hpp"
#include "dtype.hpp"
#include <memory>
#include <vector>
#include <iostream>
#include <cstring>

// TODO: 
// 0. pybind setup so that python can process lists and provide raw data to cpp
// 1. copy, move constructors to be created
// 2. proper destructor
// 3. test build
// 4. __repr__ completion

namespace tinytorch {
    // TENSOR CLASS
    
    class Tensor {
        std::shared_ptr<Storage> storage;
        Dtype dtype;
        size_t elem_count;
        std::vector<size_t> shape;
        std::vector<size_t> stride;

        void set_stride();
        void fill_zeros();

        template <typename T>
        void fill_data(std::vector<T> v) {
            if (v.size() != elem_count) {
                throw std::runtime_error("Data size mismatch (tensor.cpp)");
            }
            
            void* data_ptr = storage->data_ptr();
            
            switch (dtype) {
                case Dtype::Float32: {
                    f32* ptr = static_cast<f32*>(data_ptr);
                    for (size_t i = 0; i < elem_count; i++) {
                        ptr[i] = static_cast<f32>(v[i]);
                    }
                    break;
                }
                case Dtype::Float64: {
                    f64* ptr = static_cast<f64*>(data_ptr);
                    for (size_t i = 0; i < elem_count; i++) {
                        ptr[i] = static_cast<f64>(v[i]);
                    }
                    break;
                }
                case Dtype::Int32: {
                    i32* ptr = static_cast<i32*>(data_ptr);
                    for (size_t i = 0; i < elem_count; i++) {
                        ptr[i] = static_cast<i32>(v[i]);
                    }
                    break;
                }
                case Dtype::Int64: {
                    i64* ptr = static_cast<i64*>(data_ptr);
                    for (size_t i = 0; i < elem_count; i++) {
                        ptr[i] = static_cast<i64>(v[i]);
                    }
                    break;
                }
                case Dtype::Bool: {
                    bool* ptr = static_cast<bool*>(data_ptr);
                    for (size_t i = 0; i < elem_count; i++) {
                        ptr[i] = static_cast<bool>(v[i]);
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported dtype (tensor.cpp)");
            }
        }

        static size_t compute_elem_count(const std::vector<size_t>& shape) {
            size_t count = 1;
            for (size_t s: shape) count *= s;
            return count;
        }

    public:
        Tensor(Dtype) {
            throw std::runtime_error("Empty tensor's not allowed");
        };

        Tensor() : Tensor(Dtype::Float32) {};

        template <typename T>
        Tensor(
            std::vector<T>& init_v,
            std::vector<size_t>& shape,
            Dtype dtype = Dtype::Float32
        ): dtype(dtype) {
            elem_count = init_v.size();
            this->shape = shape;
            size_t computed_count = compute_elem_count(shape);
            if (computed_count != elem_count) {
               throw std::runtime_error("Shape and data size mismatch (tensor.hpp)");
            }

            size_t data_size = elem_count * dtype_size(dtype);
            storage = std::make_shared<Storage>(data_size);
            
            set_stride();
            fill_data(data_v);
        }

        // DESTRUCTOR
        ~Tensor() {
            std::cout << "destoying the tensor" << std::endl;
            for(size_t i = 0; i < elem_count; i++) {
                std::cout << *((double*)storage->data_ptr() + i) << " ";
            }
            std::cout << std::endl;
            std::cout << "shape: ";
            for(const auto& el: shape) std::cout << el << " ";
            std::cout << std::endl;
            std::cout << "stride: ";
            for(const auto& el: stride) std::cout << el << " ";
            std::cout << std::endl;
        }
    };
}
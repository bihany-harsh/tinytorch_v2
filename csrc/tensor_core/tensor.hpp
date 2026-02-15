#pragma once
#include "storage.hpp"
#include "dtype.hpp"
#include <memory>
#include <vector>
#include <iostream>
#include <cstring>

// dispatch macro
#define DISPATCH_DTYPE_INIT(dtype, DATA, SHAPE)                                     \
switch(dtype) {                                                                     \
        case tinytorch::Dtype::Float32: {                                           \
            std::vector<float> v = DATA.cast<std::vector<float>>();                 \
            return new tinytorch::Tensor(v, SHAPE, dtype);                          \
        }                                                                           \
        case tinytorch::Dtype::Float64: {                                           \
            std::vector<double> v = DATA.cast<std::vector<double>>();               \
            return new tinytorch::Tensor(v, SHAPE, dtype);                          \
        }                                                                           \
        case tinytorch::Dtype::Int32: {                                             \
            std::vector<long long> v_tmp = DATA.cast<std::vector<long long>>();     \
            std::vector<tinytorch::i32> v(v_tmp.begin(), v_tmp.end());             \
            return new tinytorch::Tensor(v, SHAPE, dtype);                          \
        }                                                                           \
        case tinytorch::Dtype::Int64: {                                             \
            std::vector<long long> v_tmp = DATA.cast<std::vector<long long>>();     \
            std::vector<tinytorch::i64> v(v_tmp.begin(), v_tmp.end());             \
            return new tinytorch::Tensor(v, SHAPE, dtype);                          \
        }                                                                           \
        case tinytorch::Dtype::Bool: {                                              \
            std::vector<bool> v = DATA.cast<std::vector<bool>>();                   \
            return new tinytorch::Tensor(v, SHAPE, dtype);                          \
        }                                                                           \
        default:                                                                    \
            throw std::runtime_error("Unsupported dtype.") ;                        \
    }                                                                               

// TODO: 
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
            const std::vector<T>& init_v,
            const std::vector<size_t>& shape,
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
            fill_data(init_v);
        }

        // DESTRUCTOR
        ~Tensor() {
            std::cout << "destroying the tensor" << std::endl;
            
            void* data_ptr = storage->data_ptr();
            
            switch (dtype) {
                case Dtype::Float32: {
                    f32* ptr = static_cast<f32*>(data_ptr);
                    for (size_t i = 0; i < elem_count; i++) {
                        std::cout << ptr[i] << " ";
                    }
                    break;
                }
                case Dtype::Float64: {
                    f64* ptr = static_cast<f64*>(data_ptr);
                    for (size_t i = 0; i < elem_count; i++) {
                        std::cout << ptr[i] << " ";
                    }
                    break;
                }
                case Dtype::Int32: {
                    i32* ptr = static_cast<i32*>(data_ptr);
                    for (size_t i = 0; i < elem_count; i++) {
                        std::cout << ptr[i] << " ";
                    }
                    break;
                }
                case Dtype::Int64: {
                    i64* ptr = static_cast<i64*>(data_ptr);
                    for (size_t i = 0; i < elem_count; i++) {
                        std::cout << ptr[i] << " ";
                    }
                    break;
                }
                case Dtype::Bool: {
                    bool* ptr = static_cast<bool*>(data_ptr);
                    for (size_t i = 0; i < elem_count; i++) {
                        std::cout << (ptr[i] ? "true" : "false") << " ";
                    }
                    break;
                }
                default:
                    std::cout << "unknown dtype";
            }
            
            std::cout << std::endl;
            std::cout << "shape: ";
            for(const auto& el: shape) std::cout << el << " ";
            std::cout << std::endl;
            std::cout << "stride: ";
            for(const auto& el: stride) std::cout << el << " ";
            std::cout << std::endl;
            std::cout << "dtype: ";
            if (dtype == tinytorch::Dtype::Float32) std::cout << "dtype is Float32" << std::endl;
        }
    };
}
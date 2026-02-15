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
// returning pointer instead of object suggested by Claude Sonnet-4.5 (otherwise there is a temporary object which is deleted shortly and an object that the python wrapper carries (after probably calling the copy(default) constructor and hence two (instead of one) instances of the same tensor. Pybind11 can handle returning both pointers and objects directly. Hence I think it is almost always preferable to return pointers within py::init (?)))

// TODO: 
// 1.5 testing the copy/move constructors and copy/move assignments
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

        const std::vector<size_t>& get_shape() const;

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

        // default copy constructor makes adds a reference of the shared_ptr (Storage) and apparently makes a (hard) copy of the vector member objects!
        // so we might need the copy constructor copy assignment operator but still

        // copy constructor
        Tensor(const Tensor& other) : 
            storage(other.storage), // shared_ptr copy and hence increases the ref-count
            dtype(other.dtype),
            elem_count(other.elem_count),
            shape(other.shape), // should be a deepcopy (default copy constructor behaviour of vector) and something which is not too bad
            stride(other.stride) {}

        
        // move constructor
        Tensor(Tensor &&other) noexcept: 
            storage(std::move(other.storage)),
            dtype(other.dtype),
            elem_count(elem_count),
            shape(std::move(other.shape)),
            stride(std::move(other.stride)) {}

        // copy assignment
        Tensor& operator=(const Tensor& other) {
            if (this != &other) {
                storage = other.storage; // again should be (?) shared_ptr copy which is expected/required behavior
                dtype = other.dtype;
                elem_count = other.elem_count;
                shape = other.shape;
                stride = other.stride;
            }

            return *this;
        }

        // move assignment 
        Tensor& operator=(Tensor&& other) noexcept {
            if (this != &other) {
                storage = std::move(other.storage);
                dtype = other.dtype;
                elem_count = other.elem_count;
                shape = std::move(other.shape);
                stride = std::move(other.stride);
            }

            return *this;
        }


        // DESTRUCTOR
        // ~Tensor() {
        //     // std::cout << "destroying the tensor" << std::endl;
            
        //     // void* data_ptr = storage->data_ptr();
            
        //     // switch (dtype) {
        //     //     case Dtype::Float32: {
        //     //         f32* ptr = static_cast<f32*>(data_ptr);
        //     //         for (size_t i = 0; i < elem_count; i++) {
        //     //             std::cout << ptr[i] << " ";
        //     //         }
        //     //         break;
        //     //     }
        //     //     case Dtype::Float64: {
        //     //         f64* ptr = static_cast<f64*>(data_ptr);
        //     //         for (size_t i = 0; i < elem_count; i++) {
        //     //             std::cout << ptr[i] << " ";
        //     //         }
        //     //         break;
        //     //     }
        //     //     case Dtype::Int32: {
        //     //         i32* ptr = static_cast<i32*>(data_ptr);
        //     //         for (size_t i = 0; i < elem_count; i++) {
        //     //             std::cout << ptr[i] << " ";
        //     //         }
        //     //         break;
        //     //     }
        //     //     case Dtype::Int64: {
        //     //         i64* ptr = static_cast<i64*>(data_ptr);
        //     //         for (size_t i = 0; i < elem_count; i++) {
        //     //             std::cout << ptr[i] << " ";
        //     //         }
        //     //         break;
        //     //     }
        //     //     case Dtype::Bool: {
        //     //         bool* ptr = static_cast<bool*>(data_ptr);
        //     //         for (size_t i = 0; i < elem_count; i++) {
        //     //             std::cout << (ptr[i] ? "true" : "false") << " ";
        //     //         }
        //     //         break;
        //     //     }
        //     //     default:
        //     //         std::cout << "unknown dtype";
        //     // }
            
        //     // std::cout << std::endl;
        //     // std::cout << "shape: ";
        //     // for(const auto& el: shape) std::cout << el << " ";
        //     // std::cout << std::endl;
        //     // std::cout << "stride: ";
        //     // for(const auto& el: stride) std::cout << el << " ";
        //     // std::cout << std::endl;
        //     // std::cout << "dtype: ";
        //     // if (dtype == tinytorch::Dtype::Float32) std::cout << "dtype is Float32" << std::endl;
        // }
    };
}
#pragma once
#include "storage.hpp"
#include <cstdint>
#include <stdexcept>

namespace tinytorch {

    typedef int32_t i32;
    typedef int64_t i64;
    typedef float f32;
    typedef double f64;

    enum class Dtype {
        Float32,
        Float64,
        Int32,
        Int64,
        Bool
    };

    // this function is `inline`d because the compiler throws a duplicate symbol error. Alternative is to keep the method in a .cpp file
    inline size_t dtype_size(Dtype dtype) {
        switch (dtype)
        {
        case Dtype::Float32: return sizeof(f32);
        case Dtype::Float64: return sizeof(f64);
        case Dtype::Int32: return sizeof(i32);
        case Dtype::Int64: return sizeof(i64);
        case Dtype::Bool: return sizeof(bool);
        default:
            throw std::runtime_error("invalid datatype (dtype.hpp)");
        }
    }
}
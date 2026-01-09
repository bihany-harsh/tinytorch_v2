#pragma once
#include <cstddef>
#include <cstdlib>

namespace tinytorch {

    class Storage {
        void* data;
        size_t data_size;
    
    public:
        Storage(size_t data_size){
            this->data_size = data_size;
            data = malloc(data_size);
        };
        ~Storage() {
            free(data);
        }
        void* data_ptr() { return data; }
    };
    
}
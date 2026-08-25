#pragma once

#include <cstring>
#include <memory>
#include <string>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "shared_memory.hpp"

namespace deep_ep::fabric_memory {

namespace py = pybind11;

class FabricAllocation : public std::enable_shared_from_this<FabricAllocation> {
public:
    static std::shared_ptr<FabricAllocation> allocate(const int64_t num_bytes) {
        EP_HOST_ASSERT(num_bytes > 0);
        auto allocation = std::shared_ptr<FabricAllocation>(new FabricAllocation(true));
        allocation->allocator.malloc(&allocation->ptr, static_cast<size_t>(num_bytes));
        allocation->refresh_size();
        return allocation;
    }

    static std::shared_ptr<FabricAllocation> import_handle(const py::bytes& bytes) {
        const std::string serialized = bytes;
        EP_HOST_ASSERT(serialized.size() == sizeof(shared_memory::MemHandle));
        auto allocation = std::shared_ptr<FabricAllocation>(new FabricAllocation(false));
        shared_memory::MemHandle handle = {};
        std::memcpy(&handle, serialized.data(), sizeof(handle));
        allocation->allocator.open_mem_handle(&allocation->ptr, &handle);
        allocation->num_bytes = handle.size;
        return allocation;
    }

    ~FabricAllocation() {
        if (ptr == nullptr)
            return;
        if (is_owner)
            allocator.free(ptr);
        else
            allocator.close_mem_handle(ptr);
    }

    torch::Tensor tensor() {
        auto self = shared_from_this();
        return torch::from_blob(
            ptr,
            {static_cast<int64_t>(num_bytes)},
            [self = std::move(self)](void*) mutable { self.reset(); },
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    }

    py::bytes export_handle() const {
        EP_HOST_ASSERT(is_owner and ptr != nullptr);
        shared_memory::MemHandle handle = {};
        allocator.get_mem_handle(&handle, ptr);
        return py::bytes(reinterpret_cast<const char*>(&handle), sizeof(handle));
    }

    int64_t size() const {
        return static_cast<int64_t>(num_bytes);
    }

    int64_t data_ptr() const {
        return reinterpret_cast<int64_t>(ptr);
    }

private:
    explicit FabricAllocation(const bool is_owner)
        : allocator(true), is_owner(is_owner) {}

    void refresh_size() {
        num_bytes = allocator.get_size(ptr);
    }

    shared_memory::SharedMemoryAllocator allocator;
    bool is_owner;
    void* ptr = nullptr;
    size_t num_bytes = 0;
};

static void register_apis(py::module_& m) {
    py::class_<FabricAllocation, std::shared_ptr<FabricAllocation>>(m, "FabricAllocation")
        .def_static("allocate", &FabricAllocation::allocate)
        .def_static("import_handle", &FabricAllocation::import_handle)
        .def("tensor", &FabricAllocation::tensor)
        .def("export_handle", &FabricAllocation::export_handle)
        .def_property_readonly("size", &FabricAllocation::size)
        .def_property_readonly("data_ptr", &FabricAllocation::data_ptr);
}

}  // namespace deep_ep::fabric_memory

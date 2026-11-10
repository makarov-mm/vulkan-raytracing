#pragma once
// Vulkan headers, error checking and RAII ownership of Vulkan handles.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <concepts>
#include <cstdint>
#include <memory>
#include <source_location>
#include <string_view>
#include <utility>

// Throws std::runtime_error naming the failing call site unless `result` is VK_SUCCESS.
void check(VkResult result, std::source_location location = std::source_location::current());

[[nodiscard]] std::string_view toString(VkResult result);

template <std::unsigned_integral T>
constexpr T alignUp(T value, T alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

constexpr bool operator==(VkExtent2D a, VkExtent2D b) { return a.width == b.width && a.height == b.height; }

// Entry points of VK_KHR_acceleration_structure and VK_KHR_ray_tracing_pipeline.
// The loader library does not export them; VulkanContext loads them once.
struct RayTracingFunctions {
    PFN_vkCreateAccelerationStructureKHR createAccelerationStructure = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroyAccelerationStructure = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR getAccelerationStructureBuildSizes = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getAccelerationStructureDeviceAddress = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildAccelerationStructures = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR createRayTracingPipelines = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR getRayTracingShaderGroupHandles = nullptr;
    PFN_vkCmdTraceRaysKHR cmdTraceRays = nullptr;
};
inline RayTracingFunctions vkrt;

// ---- Ownership ---------------------------------------------------------------

// Owns a handle created from `parent`; `Destroy(parent, handle, nullptr)` releases it.
template <typename Parent, typename Handle, auto Destroy>
class Unique {
public:
    Unique() = default;
    Unique(Parent parent, Handle handle) noexcept : parent_{parent}, handle_{handle} {}
    Unique(Unique&& other) noexcept
        : parent_{std::exchange(other.parent_, VK_NULL_HANDLE)},
          handle_{std::exchange(other.handle_, VK_NULL_HANDLE)} {}
    Unique& operator=(Unique&& other) noexcept {
        if (this != &other) {
            reset();
            parent_ = std::exchange(other.parent_, VK_NULL_HANDLE);
            handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
        }
        return *this;
    }
    Unique(const Unique&) = delete;
    Unique& operator=(const Unique&) = delete;
    ~Unique() { reset(); }

    void reset() noexcept {
        if (handle_ != VK_NULL_HANDLE) Destroy(parent_, handle_, nullptr);
        handle_ = VK_NULL_HANDLE;
    }

    [[nodiscard]] Handle get() const noexcept { return handle_; }
    [[nodiscard]] const Handle* address() const noexcept { return &handle_; }
    operator Handle() const noexcept { return handle_; }  // pass straight to Vulkan calls

private:
    Parent parent_ = VK_NULL_HANDLE;
    Handle handle_ = VK_NULL_HANDLE;
};

void destroyDebugMessenger(VkInstance, VkDebugUtilsMessengerEXT, const VkAllocationCallbacks*);
inline void destroyAccelerationStructure(VkDevice device, VkAccelerationStructureKHR handle,
                                         const VkAllocationCallbacks* allocator) {
    vkrt.destroyAccelerationStructure(device, handle, allocator);
}

struct InstanceDeleter {
    void operator()(VkInstance instance) const noexcept { vkDestroyInstance(instance, nullptr); }
};
struct DeviceDeleter {
    void operator()(VkDevice device) const noexcept { vkDestroyDevice(device, nullptr); }
};
using UniqueInstance = std::unique_ptr<VkInstance_T, InstanceDeleter>;
using UniqueDevice = std::unique_ptr<VkDevice_T, DeviceDeleter>;
using UniqueDebugMessenger = Unique<VkInstance, VkDebugUtilsMessengerEXT, destroyDebugMessenger>;
using UniqueSurface = Unique<VkInstance, VkSurfaceKHR, vkDestroySurfaceKHR>;
using UniqueSwapchain = Unique<VkDevice, VkSwapchainKHR, vkDestroySwapchainKHR>;
using UniqueBuffer = Unique<VkDevice, VkBuffer, vkDestroyBuffer>;
using UniqueImage = Unique<VkDevice, VkImage, vkDestroyImage>;
using UniqueImageView = Unique<VkDevice, VkImageView, vkDestroyImageView>;
using UniqueMemory = Unique<VkDevice, VkDeviceMemory, vkFreeMemory>;
using UniqueSemaphore = Unique<VkDevice, VkSemaphore, vkDestroySemaphore>;
using UniqueFence = Unique<VkDevice, VkFence, vkDestroyFence>;
using UniqueCommandPool = Unique<VkDevice, VkCommandPool, vkDestroyCommandPool>;
using UniqueDescriptorSetLayout = Unique<VkDevice, VkDescriptorSetLayout, vkDestroyDescriptorSetLayout>;
using UniqueDescriptorPool = Unique<VkDevice, VkDescriptorPool, vkDestroyDescriptorPool>;
using UniquePipelineLayout = Unique<VkDevice, VkPipelineLayout, vkDestroyPipelineLayout>;
using UniquePipeline = Unique<VkDevice, VkPipeline, vkDestroyPipeline>;
using UniqueShaderModule = Unique<VkDevice, VkShaderModule, vkDestroyShaderModule>;
using UniqueAccelerationStructure = Unique<VkDevice, VkAccelerationStructureKHR, destroyAccelerationStructure>;

// A buffer with its own memory. Destruction runs bottom-up (buffer, then
// memory); move assignment releases the old objects in the same order.
struct Buffer {
    UniqueMemory memory;
    UniqueBuffer buffer;
    VkDeviceAddress address = 0;
    VkDeviceSize size = 0;
    void* mapped = nullptr;  // persistently mapped if host-visible

    Buffer() = default;
    Buffer(Buffer&&) noexcept = default;
    Buffer& operator=(Buffer&& other) noexcept {
        buffer = std::move(other.buffer);
        memory = std::move(other.memory);
        address = std::exchange(other.address, 0);
        size = std::exchange(other.size, 0);
        mapped = std::exchange(other.mapped, nullptr);
        return *this;
    }
};

// A 2D image with its own memory and a view; same destruction order rules as Buffer.
struct Image {
    UniqueMemory memory;
    UniqueImage image;
    UniqueImageView view;

    Image() = default;
    Image(Image&&) noexcept = default;
    Image& operator=(Image&& other) noexcept {
        view = std::move(other.view);
        image = std::move(other.image);
        memory = std::move(other.memory);
        return *this;
    }
};

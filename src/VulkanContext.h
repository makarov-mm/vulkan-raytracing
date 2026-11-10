#pragma once
// Instance, surface, GPU selection, logical device and resource helpers.

#include <functional>
#include <string>

#include "Vulkan.h"

class VulkanContext {
public:
    VulkanContext(HINSTANCE instance, HWND window);
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    [[nodiscard]] VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    [[nodiscard]] VkDevice device() const { return device_.get(); }
    [[nodiscard]] VkSurfaceKHR surface() const { return surface_; }
    [[nodiscard]] VkQueue queue() const { return queue_; }
    [[nodiscard]] std::uint32_t queueFamily() const { return queueFamily_; }
    [[nodiscard]] const std::string& deviceName() const { return deviceName_; }
    [[nodiscard]] const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rayTracingProperties() const {
        return rayTracingProperties_;
    }
    [[nodiscard]] const VkPhysicalDeviceAccelerationStructurePropertiesKHR& accelerationStructureProperties() const {
        return accelerationStructureProperties_;
    }

    // Buffers always get a device address: ray tracing needs one for nearly every buffer.
    [[nodiscard]] Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                      VkMemoryPropertyFlags properties) const;
    // A device-local 2D image with a colour view.
    [[nodiscard]] Image createImage(VkExtent2D extent, VkFormat format, VkImageUsageFlags usage) const;

    // Records commands with `record`, submits them and waits until they finish.
    void submitAndWait(const std::function<void(VkCommandBuffer)>& record) const;

private:
    void createInstance();
    void pickPhysicalDevice();
    void createDevice();
    [[nodiscard]] UniqueMemory allocate(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags properties,
                                        bool deviceAddress) const;

    UniqueInstance instance_;
    UniqueDebugMessenger debugMessenger_;
    UniqueSurface surface_;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    std::string deviceName_;
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties_{};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties_{};
    std::uint32_t queueFamily_ = 0;
    UniqueDevice device_;
    VkQueue queue_ = VK_NULL_HANDLE;
    UniqueCommandPool commandPool_;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;  // for submitAndWait()
};

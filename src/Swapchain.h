#pragma once
// Swapchain plus one "render finished" semaphore per image.

#include <vector>

#include "Vulkan.h"

class VulkanContext;

class Swapchain {
public:
    explicit Swapchain(const VulkanContext& context);

    // (Re)creates the swapchain for a window of the given client size. The
    // device must be idle. Returns false while the window has no area.
    bool recreate(VkExtent2D windowSize);

    [[nodiscard]] VkSwapchainKHR handle() const { return swapchain_; }
    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] VkImage image(std::uint32_t index) const { return images_[index]; }
    // Signalled when rendering into image `index` finishes; waited on by present.
    // Per image rather than per frame: presentation may still hold the semaphore
    // when the next frame starts.
    [[nodiscard]] VkSemaphore renderFinished(std::uint32_t index) const { return renderFinished_[index]; }

private:
    const VulkanContext& context_;
    VkSurfaceFormatKHR format_{};
    VkExtent2D extent_{};
    UniqueSwapchain swapchain_;
    std::vector<VkImage> images_;
    std::vector<UniqueSemaphore> renderFinished_;
};

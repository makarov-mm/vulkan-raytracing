#include "Swapchain.h"

#include <algorithm>
#include <bit>
#include <print>
#include <stdexcept>

#include "VulkanContext.h"

namespace {

// The ray tracer writes gamma-encoded rgba8 and blits it, so the swapchain
// must be UNORM: an sRGB format would encode the colours a second time.
VkSurfaceFormatKHR chooseFormat(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &count, formats.data());

    const auto blittable = [&](VkFormat format) {
        VkFormatProperties properties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
    };
    for (VkFormat preferred : {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_A8B8G8R8_UNORM_PACK32}) {
        for (const VkSurfaceFormatKHR& format : formats) {
            if (format.format == preferred && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
                blittable(format.format)) {
                return format;
            }
        }
    }
    for (const VkSurfaceFormatKHR& format : formats) {
        if (blittable(format.format)) {
            std::println(stderr, "No 8-bit UNORM swapchain format; colours may look off.");
            return format;
        }
    }
    throw std::runtime_error("The window surface has no format that can be a blit target");
}

} // namespace

Swapchain::Swapchain(const VulkanContext& context)
    : context_{context}, format_{chooseFormat(context.physicalDevice(), context.surface())} {}

bool Swapchain::recreate(VkExtent2D windowSize) {
    VkSurfaceCapabilitiesKHR capabilities;
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context_.physicalDevice(), context_.surface(), &capabilities));

    // On Windows currentExtent is the client size (0 x 0 while minimised);
    // 0xFFFFFFFF would mean that the swapchain decides.
    VkExtent2D extent = capabilities.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent = {std::clamp(windowSize.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                  std::clamp(windowSize.height, capabilities.minImageExtent.height,
                             capabilities.maxImageExtent.height)};
    }
    if (extent.width == 0 || extent.height == 0) return false;
    if (!(capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        throw std::runtime_error("Swapchain images cannot be blit targets on this surface");
    }

    std::uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) imageCount = std::min(imageCount, capabilities.maxImageCount);

    const auto alphaModes = capabilities.supportedCompositeAlpha;
    const VkSwapchainCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = context_.surface(),
        .minImageCount = imageCount,
        .imageFormat = format_.format,
        .imageColorSpace = format_.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = (alphaModes & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                              ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                              : static_cast<VkCompositeAlphaFlagBitsKHR>(std::bit_floor(alphaModes)),
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,  // vsync, always available
        .clipped = VK_TRUE,
        .oldSwapchain = swapchain_,
    };
    const VkDevice device = context_.device();
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    check(vkCreateSwapchainKHR(device, &info, nullptr, &swapchain));
    swapchain_ = {device, swapchain};  // releases the previous swapchain
    extent_ = extent;

    std::uint32_t count = 0;
    check(vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr));
    images_.resize(count);
    check(vkGetSwapchainImagesKHR(device, swapchain, &count, images_.data()));

    renderFinished_.clear();
    const VkSemaphoreCreateInfo semaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (std::uint32_t i = 0; i < count; ++i) {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        check(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore));
        renderFinished_.emplace_back(device, semaphore);
    }
    return true;
}

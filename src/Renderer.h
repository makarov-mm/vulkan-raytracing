#pragma once
// Hardware ray tracer: acceleration structures, ray-tracing pipeline, shader
// binding table and the per-frame trace -> accumulate -> blit -> present loop.

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "../shaders/shared.h"
#include "Camera.h"
#include "Swapchain.h"
#include "Vulkan.h"

class VulkanContext;
struct Scene;

struct RenderSettings {
    std::uint32_t samplesPerFrame = 6;   // a multiple of 3 gives every colour channel the same refraction samples per frame
    std::uint32_t maxBounces = 8;        // glass needs several
    std::uint32_t targetSamples = 4096;  // per pixel; rendering pauses there until the view changes
};

class Renderer {
public:
    Renderer(const VulkanContext& context, const Scene& scene, const RenderSettings& settings,
             const std::filesystem::path& shaderDirectory);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Traces one frame, adds it to the running average and presents it.
    // Returns false if nothing was presented (no window area, or the swapchain
    // was out of date and gets rebuilt on the next call).
    bool render(const CameraView& view, VkExtent2D windowSize);

    void resetAccumulation() { accumulatedFrames_ = 0; }
    [[nodiscard]] std::uint32_t samplesPerPixel() const { return accumulatedFrames_ * settings_.samplesPerFrame; }
    [[nodiscard]] bool converged() const { return samplesPerPixel() >= settings_.targetSamples; }

private:
    static constexpr std::uint32_t kFramesInFlight = 2;

    struct Frame {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        UniqueSemaphore imageAcquired;
        UniqueFence done;
    };

    void createPipeline(const std::filesystem::path& shaderDirectory);
    void createSceneResources(const Scene& scene);
    void createDescriptorSet();
    void createFrames();
    bool rebuildSwapchain(VkExtent2D windowSize);
    void createImages(VkExtent2D extent);
    void record(VkCommandBuffer commandBuffer, std::uint32_t imageIndex, const CameraView& view) const;

    const VulkanContext& context_;
    RenderSettings settings_;
    gpu::vec4 light_{};

    // Scene: shared vertex/index buffers, one BLAS per mesh, one TLAS of instances.
    Buffer vertices_;
    Buffer indices_;
    Buffer instances_;
    std::vector<Buffer> blasStorage_;
    std::vector<UniqueAccelerationStructure> blas_;
    Buffer tlasStorage_;
    UniqueAccelerationStructure tlas_;

    // Pipeline and shader binding table.
    UniqueDescriptorSetLayout descriptorSetLayout_;
    UniquePipelineLayout pipelineLayout_;
    UniquePipeline pipeline_;
    Buffer shaderBindingTable_;
    VkStridedDeviceAddressRegionKHR raygenRegion_{};
    VkStridedDeviceAddressRegionKHR missRegion_{};
    VkStridedDeviceAddressRegionKHR hitRegion_{};
    VkStridedDeviceAddressRegionKHR callableRegion_{};
    UniqueDescriptorPool descriptorPool_;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;

    // Presentation.
    Swapchain swapchain_;
    bool swapchainValid_ = false;
    VkExtent2D windowSize_{};
    Image output_;        // rgba8, gamma-encoded; blitted to the swapchain
    Image accumulation_;  // rgba32f running average, always in GENERAL layout
    VkExtent2D imageExtent_{};

    UniqueCommandPool commandPool_;
    std::array<Frame, kFramesInFlight> frames_;
    std::uint64_t frameIndex_ = 0;
    std::uint32_t accumulatedFrames_ = 0;
};

#include "VulkanContext.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <format>
#include <optional>
#include <print>
#include <stdexcept>
#include <vector>

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
#ifdef NDEBUG
constexpr bool kWantValidation = false;
#else
constexpr bool kWantValidation = true;
#endif

constexpr std::array kDeviceExtensions{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,  // required by acceleration structures
};

// Called from inside the Vulkan loader: must not throw, hence fprintf.
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    std::fprintf(stderr, "[Vulkan] %s\n", data->pMessage);
    return VK_FALSE;
}

bool hasInstanceLayer(std::string_view name) {
    std::uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    return std::ranges::any_of(layers, [&](const VkLayerProperties& layer) { return name == layer.layerName; });
}

bool hasDeviceExtensions(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, available.data());
    return std::ranges::all_of(kDeviceExtensions, [&](std::string_view required) {
        return std::ranges::any_of(available,
                                   [&](const VkExtensionProperties& e) { return required == e.extensionName; });
    });
}

bool hasDeviceFeatures(VkPhysicalDevice device) {
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracing{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, .pNext = &rayTracing};
    VkPhysicalDeviceVulkan13Features vulkan13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                              .pNext = &accelerationStructure};
    VkPhysicalDeviceVulkan12Features vulkan12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                              .pNext = &vulkan13};
    VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &vulkan12};
    vkGetPhysicalDeviceFeatures2(device, &features);
    return vulkan12.bufferDeviceAddress && vulkan13.synchronization2 && accelerationStructure.accelerationStructure &&
           rayTracing.rayTracingPipeline;
}

// A queue family that can blit (graphics) and present to the surface.
std::optional<std::uint32_t> findQueueFamily(VkPhysicalDevice device, VkSurfaceKHR surface) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
    for (std::uint32_t i = 0; i < count; ++i) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present);
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) return i;
    }
    return std::nullopt;
}

template <typename Function>
void load(VkDevice device, Function& function, const char* name) {
    function = reinterpret_cast<Function>(vkGetDeviceProcAddr(device, name));
    if (!function) throw std::runtime_error(std::format("{} is not available", name));
}

} // namespace

VulkanContext::VulkanContext(HINSTANCE instance, HWND window) {
    createInstance();

    const VkWin32SurfaceCreateInfoKHR surfaceInfo{
        .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR, .hinstance = instance, .hwnd = window};
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    check(vkCreateWin32SurfaceKHR(instance_.get(), &surfaceInfo, nullptr, &surface));
    surface_ = {instance_.get(), surface};

    pickPhysicalDevice();
    createDevice();

    const VkCommandPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                           .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                           .queueFamilyIndex = queueFamily_};
    VkCommandPool pool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(device(), &poolInfo, nullptr, &pool));
    commandPool_ = {device(), pool};

    const VkCommandBufferAllocateInfo allocateInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                   .commandPool = pool,
                                                   .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                   .commandBufferCount = 1};
    check(vkAllocateCommandBuffers(device(), &allocateInfo, &commandBuffer_));
}

void VulkanContext::createInstance() {
    const bool validation = kWantValidation && hasInstanceLayer(kValidationLayer);
    if (kWantValidation && !validation) {
        std::println(stderr, "{} is not installed; running without validation.", kValidationLayer);
    }

    std::vector<const char*> extensions{VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    if (validation) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    const VkDebugUtilsMessengerCreateInfoEXT messengerInfo{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
    };
    const VkApplicationInfo appInfo{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                    .pApplicationName = "Vulkan Hardware Ray Tracing",
                                    .apiVersion = VK_API_VERSION_1_3};
    const VkInstanceCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = validation ? &messengerInfo : nullptr,  // also covers vkCreateInstance itself
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = validation ? 1u : 0u,
        .ppEnabledLayerNames = &kValidationLayer,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
    VkInstance instance = VK_NULL_HANDLE;
    check(vkCreateInstance(&info, nullptr, &instance));
    instance_.reset(instance);

    if (validation) {
        const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
        check(create(instance, &messengerInfo, nullptr, &messenger));
        debugMessenger_ = {instance, messenger};
    }
}

void VulkanContext::pickPhysicalDevice() {
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_.get(), &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_.get(), &count, devices.data());

    // Prefer a discrete GPU: laptops often list the integrated one first.
    int bestScore = -1;
    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        const auto family = findQueueFamily(device, surface_);
        if (properties.apiVersion < VK_API_VERSION_1_3 || !family || !hasDeviceExtensions(device) ||
            !hasDeviceFeatures(device)) {
            continue;
        }
        const int score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 2
                          : properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1
                                                                                            : 0;
        if (score > bestScore) {
            bestScore = score;
            physicalDevice_ = device;
            queueFamily_ = *family;
        }
    }
    if (physicalDevice_ == VK_NULL_HANDLE) {
        throw std::runtime_error("No GPU with Vulkan 1.3 and VK_KHR_ray_tracing_pipeline was found.\n"
                                 "A ray-tracing capable GPU and current drivers are required.");
    }

    accelerationStructureProperties_ = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    rayTracingProperties_ = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
                             .pNext = &accelerationStructureProperties_};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                           .pNext = &rayTracingProperties_};
    vkGetPhysicalDeviceProperties2(physicalDevice_, &properties);
    rayTracingProperties_.pNext = nullptr;  // the chain was only for the query
    deviceName_ = properties.properties.deviceName;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties_);
    std::println("GPU: {}", deviceName_);
}

void VulkanContext::createDevice() {
    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queueInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                            .queueFamilyIndex = queueFamily_,
                                            .queueCount = 1,
                                            .pQueuePriorities = &priority};

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracing{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR, .rayTracingPipeline = VK_TRUE};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &rayTracing,
        .accelerationStructure = VK_TRUE};
    VkPhysicalDeviceVulkan13Features vulkan13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                              .pNext = &accelerationStructure,
                                              .synchronization2 = VK_TRUE};
    VkPhysicalDeviceVulkan12Features vulkan12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                              .pNext = &vulkan13,
                                              .bufferDeviceAddress = VK_TRUE};

    const VkDeviceCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &vulkan12,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledExtensionCount = static_cast<std::uint32_t>(kDeviceExtensions.size()),
        .ppEnabledExtensionNames = kDeviceExtensions.data(),
    };
    VkDevice device = VK_NULL_HANDLE;
    check(vkCreateDevice(physicalDevice_, &info, nullptr, &device));
    device_.reset(device);
    vkGetDeviceQueue(device, queueFamily_, 0, &queue_);

    load(device, vkrt.createAccelerationStructure, "vkCreateAccelerationStructureKHR");
    load(device, vkrt.destroyAccelerationStructure, "vkDestroyAccelerationStructureKHR");
    load(device, vkrt.getAccelerationStructureBuildSizes, "vkGetAccelerationStructureBuildSizesKHR");
    load(device, vkrt.getAccelerationStructureDeviceAddress, "vkGetAccelerationStructureDeviceAddressKHR");
    load(device, vkrt.cmdBuildAccelerationStructures, "vkCmdBuildAccelerationStructuresKHR");
    load(device, vkrt.createRayTracingPipelines, "vkCreateRayTracingPipelinesKHR");
    load(device, vkrt.getRayTracingShaderGroupHandles, "vkGetRayTracingShaderGroupHandlesKHR");
    load(device, vkrt.cmdTraceRays, "vkCmdTraceRaysKHR");
}

UniqueMemory VulkanContext::allocate(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags properties,
                                     bool deviceAddress) const {
    std::optional<std::uint32_t> type;
    for (std::uint32_t i = 0; i < memoryProperties_.memoryTypeCount && !type; ++i) {
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (memoryProperties_.memoryTypes[i].propertyFlags & properties) == properties) {
            type = i;
        }
    }
    if (!type) throw std::runtime_error("No suitable memory type");

    const VkMemoryAllocateFlagsInfo flags{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                          .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    const VkMemoryAllocateInfo info{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                    .pNext = deviceAddress ? &flags : nullptr,
                                    .allocationSize = requirements.size,
                                    .memoryTypeIndex = *type};
    VkDeviceMemory memory = VK_NULL_HANDLE;
    check(vkAllocateMemory(device(), &info, nullptr, &memory));
    return {device(), memory};
}

Buffer VulkanContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                   VkMemoryPropertyFlags properties) const {
    Buffer result;
    result.size = size;

    const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .size = size,
                                  .usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buffer = VK_NULL_HANDLE;
    check(vkCreateBuffer(device(), &info, nullptr, &buffer));
    result.buffer = {device(), buffer};

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device(), buffer, &requirements);
    result.memory = allocate(requirements, properties, true);
    check(vkBindBufferMemory(device(), buffer, result.memory, 0));

    const VkBufferDeviceAddressInfo addressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                .buffer = buffer};
    result.address = vkGetBufferDeviceAddress(device(), &addressInfo);
    if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        check(vkMapMemory(device(), result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped));
    }
    return result;
}

Image VulkanContext::createImage(VkExtent2D extent, VkFormat format, VkImageUsageFlags usage) const {
    Image result;
    const VkImageCreateInfo info{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                 .imageType = VK_IMAGE_TYPE_2D,
                                 .format = format,
                                 .extent = {extent.width, extent.height, 1},
                                 .mipLevels = 1,
                                 .arrayLayers = 1,
                                 .samples = VK_SAMPLE_COUNT_1_BIT,
                                 .tiling = VK_IMAGE_TILING_OPTIMAL,
                                 .usage = usage,
                                 .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                 .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage image = VK_NULL_HANDLE;
    check(vkCreateImage(device(), &info, nullptr, &image));
    result.image = {device(), image};

    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(device(), image, &requirements);
    result.memory = allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
    check(vkBindImageMemory(device(), image, result.memory, 0));

    const VkImageViewCreateInfo viewInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                         .image = image,
                                         .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                         .format = format,
                                         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView view = VK_NULL_HANDLE;
    check(vkCreateImageView(device(), &viewInfo, nullptr, &view));
    result.view = {device(), view};
    return result;
}

void VulkanContext::submitAndWait(const std::function<void(VkCommandBuffer)>& record) const {
    check(vkResetCommandBuffer(commandBuffer_, 0));
    const VkCommandBufferBeginInfo beginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                             .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    check(vkBeginCommandBuffer(commandBuffer_, &beginInfo));
    record(commandBuffer_);
    check(vkEndCommandBuffer(commandBuffer_));

    const VkCommandBufferSubmitInfo commandInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                                .commandBuffer = commandBuffer_};
    const VkSubmitInfo2 submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                               .commandBufferInfoCount = 1,
                               .pCommandBufferInfos = &commandInfo};
    check(vkQueueSubmit2(queue_, 1, &submit, VK_NULL_HANDLE));
    check(vkQueueWaitIdle(queue_));
}

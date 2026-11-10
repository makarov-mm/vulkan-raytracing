#include "Renderer.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "Scene.h"
#include "VulkanContext.h"

namespace {

static_assert(sizeof(gpu::FrameConstants) == 96, "push constants must match shaders/shared.h");
static_assert(sizeof(gpu::InstanceData) == 32, "instance data must match the std430 layout");

constexpr VkFormat kOutputFormat = VK_FORMAT_R8G8B8A8_UNORM;            // rgba8 in raygen.rgen
constexpr VkFormat kAccumulationFormat = VK_FORMAT_R32G32B32A32_SFLOAT;  // rgba32f in raygen.rgen

// Shader groups, in pipeline order: ray generation, two miss shaders, one hit group.
enum ShaderGroup : std::uint32_t { kRaygen, kMiss, kShadowMiss, kHitGroup, kGroupCount };

gpu::vec4 toVec4(Vec3 v, float w = 0.0f) { return {v.x, v.y, v.z, w}; }

template <typename T>
std::span<const std::byte> bytesOf(const std::vector<T>& values) {
    return std::as_bytes(std::span{values});
}

// UTF-8 for messages; path::string() would use the ANSI code page on Windows.
std::string displayPath(const std::filesystem::path& path) {
    const std::u8string utf8 = path.u8string();
    return {utf8.begin(), utf8.end()};
}

std::vector<std::uint32_t> readSpirv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error(std::format("Cannot open shader {}", displayPath(path)));
    const auto size = static_cast<std::size_t>(file.tellg());
    if (size == 0 || size % sizeof(std::uint32_t) != 0) {
        throw std::runtime_error(std::format("{} is not SPIR-V", displayPath(path)));
    }
    std::vector<std::uint32_t> code(size / sizeof(std::uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));
    return code;
}

UniqueShaderModule loadShader(VkDevice device, const std::filesystem::path& path) {
    const std::vector<std::uint32_t> code = readSpirv(path);
    const VkShaderModuleCreateInfo info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                        .codeSize = code.size() * sizeof(std::uint32_t),
                                        .pCode = code.data()};
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device, &info, nullptr, &module));
    return {device, module};
}

VkImageMemoryBarrier2 imageBarrier(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                                   VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                                   VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    return {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = srcStage,
            .srcAccessMask = srcAccess,
            .dstStageMask = dstStage,
            .dstAccessMask = dstAccess,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
}

void pipelineBarrier(VkCommandBuffer commandBuffer, std::span<const VkImageMemoryBarrier2> barriers) {
    const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                      .imageMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size()),
                                      .pImageMemoryBarriers = barriers.data()};
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void memoryBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                   VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    const VkMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                   .srcStageMask = srcStage,
                                   .srcAccessMask = srcAccess,
                                   .dstStageMask = dstStage,
                                   .dstAccessMask = dstAccess};
    const VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

// Gathers the contents of device-local buffers and copies them through one staging buffer.
class UploadBatch {
public:
    explicit UploadBatch(const VulkanContext& context) : context_{context} {}

    // Creates a device-local buffer that will hold `data`, starting at the first
    // device address aligned to `alignment` (see dataAddress()).
    Buffer add(std::span<const std::byte> data, VkBufferUsageFlags usage, VkDeviceSize alignment = 1) {
        Buffer buffer = context_.createBuffer(data.size() + alignment - 1, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        const VkDeviceSize staged = alignUp<VkDeviceSize>(bytes_.size(), 16);
        bytes_.resize(staged + data.size());
        std::ranges::copy(data, bytes_.begin() + static_cast<std::ptrdiff_t>(staged));
        const VkDeviceSize offset = dataAddress(buffer, alignment) - buffer.address;
        copies_.push_back({buffer.buffer, {.srcOffset = staged, .dstOffset = offset, .size = data.size()}});
        return buffer;
    }

    static VkDeviceAddress dataAddress(const Buffer& buffer, VkDeviceSize alignment) {
        return alignUp<VkDeviceAddress>(buffer.address, alignment);
    }

    // Records the copies; the batch must stay alive until they have executed.
    void record(VkCommandBuffer commandBuffer) {
        staging_ = context_.createBuffer(bytes_.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        std::memcpy(staging_.mapped, bytes_.data(), bytes_.size());
        for (const auto& [target, region] : copies_) vkCmdCopyBuffer(commandBuffer, staging_.buffer, target, 1, &region);
    }

private:
    const VulkanContext& context_;
    std::vector<std::byte> bytes_;
    std::vector<std::pair<VkBuffer, VkBufferCopy>> copies_;
    Buffer staging_;
};

} // namespace

Renderer::Renderer(const VulkanContext& context, const Scene& scene, const RenderSettings& settings,
                   const std::filesystem::path& shaderDirectory)
    : context_{context},
      settings_{settings},
      light_{toVec4(scene.light.center, scene.light.radius)},
      swapchain_{context} {
    createPipeline(shaderDirectory);
    createSceneResources(scene);
    createDescriptorSet();
    createFrames();
}

Renderer::~Renderer() { vkDeviceWaitIdle(context_.device()); }

// ---- Setup -------------------------------------------------------------------

void Renderer::createPipeline(const std::filesystem::path& shaderDirectory) {
    const VkDevice device = context_.device();

    using Binding = VkDescriptorSetLayoutBinding;
    const std::array bindings{
        Binding{gpu::kBindingScene, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        Binding{gpu::kBindingOutput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        Binding{gpu::kBindingAccumulation, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        Binding{gpu::kBindingVertices, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
        Binding{gpu::kBindingIndices, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
        Binding{gpu::kBindingInstances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
    };
    const VkDescriptorSetLayoutCreateInfo setLayoutInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
                                                        .pBindings = bindings.data()};
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    check(vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &setLayout));
    descriptorSetLayout_ = {device, setLayout};

    const VkPushConstantRange pushConstants{VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(gpu::FrameConstants)};
    const VkPipelineLayoutCreateInfo layoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                .setLayoutCount = 1,
                                                .pSetLayouts = &setLayout,
                                                .pushConstantRangeCount = 1,
                                                .pPushConstantRanges = &pushConstants};
    VkPipelineLayout layout = VK_NULL_HANDLE;
    check(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout));
    pipelineLayout_ = {device, layout};

    const std::array modules{
        loadShader(device, shaderDirectory / "raygen.rgen.spv"),
        loadShader(device, shaderDirectory / "miss.rmiss.spv"),
        loadShader(device, shaderDirectory / "shadow.rmiss.spv"),
        loadShader(device, shaderDirectory / "closesthit.rchit.spv"),
    };
    const auto stage = [&](VkShaderStageFlagBits kind, std::size_t module) {
        return VkPipelineShaderStageCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                               .stage = kind,
                                               .module = modules[module],
                                               .pName = "main"};
    };
    const std::array stages{
        stage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0),
        stage(VK_SHADER_STAGE_MISS_BIT_KHR, 1),
        stage(VK_SHADER_STAGE_MISS_BIT_KHR, 2),
        stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 3),
    };

    const auto group = [](VkRayTracingShaderGroupTypeKHR type, std::uint32_t general, std::uint32_t closestHit) {
        return VkRayTracingShaderGroupCreateInfoKHR{.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                                    .type = type,
                                                    .generalShader = general,
                                                    .closestHitShader = closestHit,
                                                    .anyHitShader = VK_SHADER_UNUSED_KHR,
                                                    .intersectionShader = VK_SHADER_UNUSED_KHR};
    };
    const std::array groups{
        group(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0, VK_SHADER_UNUSED_KHR),  // kRaygen
        group(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1, VK_SHADER_UNUSED_KHR),  // kMiss
        group(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 2, VK_SHADER_UNUSED_KHR),  // kShadowMiss
        group(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 3),  // kHitGroup
    };
    static_assert(groups.size() == kGroupCount);

    const VkRayTracingPipelineCreateInfoKHR pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .stageCount = static_cast<std::uint32_t>(stages.size()),
        .pStages = stages.data(),
        .groupCount = static_cast<std::uint32_t>(groups.size()),
        .pGroups = groups.data(),
        .maxPipelineRayRecursionDepth = 1,  // every traceRayEXT call is made from ray generation
        .layout = layout,
    };
    VkPipeline pipeline = VK_NULL_HANDLE;
    check(vkrt.createRayTracingPipelines(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
    pipeline_ = {device, pipeline};
}

// Uploads the geometry, the instance data and the shader binding table, and
// builds the acceleration structures, all in one submission.
void Renderer::createSceneResources(const Scene& scene) {
    const VkDevice device = context_.device();
    UploadBatch upload{context_};

    // ---- Geometry: all meshes in one vertex and one index buffer -------------
    struct MeshRange {
        std::uint32_t firstVertex, vertexCount, firstIndex, indexCount;
    };
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<MeshRange> meshes;
    for (const Mesh& mesh : scene.meshes) {
        meshes.push_back({static_cast<std::uint32_t>(vertices.size()), static_cast<std::uint32_t>(mesh.vertices.size()),
                          static_cast<std::uint32_t>(indices.size()), static_cast<std::uint32_t>(mesh.indices.size())});
        vertices.insert(vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
        indices.insert(indices.end(), mesh.indices.begin(), mesh.indices.end());
    }

    std::vector<gpu::InstanceData> instanceData;
    for (const Instance& instance : scene.instances) {
        const MeshRange& mesh = meshes.at(instance.mesh);
        instanceData.push_back({.albedo = toVec4(instance.albedo, instance.reflectivity),
                                .geometry = {instance.material, mesh.firstIndex, mesh.firstVertex, 0}});
    }

    constexpr VkBufferUsageFlags kGeometryUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    vertices_ = upload.add(bytesOf(vertices), kGeometryUsage);
    indices_ = upload.add(bytesOf(indices), kGeometryUsage);
    instances_ = upload.add(bytesOf(instanceData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // ---- Acceleration structures -------------------------------------------
    const auto createStructure = [&](VkAccelerationStructureTypeKHR type, VkDeviceSize size, Buffer& storage) {
        storage = context_.createBuffer(size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        const VkAccelerationStructureCreateInfoKHR info{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                                                        .buffer = storage.buffer,
                                                        .size = size,
                                                        .type = type};
        VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
        check(vkrt.createAccelerationStructure(device, &info, nullptr, &handle));
        return UniqueAccelerationStructure{device, handle};
    };
    const auto addressOf = [&](VkAccelerationStructureKHR structure) {
        const VkAccelerationStructureDeviceAddressInfoKHR info{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR, .accelerationStructure = structure};
        return vkrt.getAccelerationStructureDeviceAddress(device, &info);
    };

    // Each build gets its own region of one scratch buffer.
    const VkDeviceSize scratchAlignment =
        context_.accelerationStructureProperties().minAccelerationStructureScratchOffsetAlignment;
    std::vector<VkDeviceSize> scratchOffsets;
    VkDeviceSize scratchSize = 0;
    const auto reserveScratch = [&](VkDeviceSize size) {
        scratchOffsets.push_back(scratchSize);
        scratchSize = alignUp(scratchSize + size, scratchAlignment);
    };

    // One bottom-level structure per mesh.
    std::vector<VkAccelerationStructureGeometryKHR> blasGeometries(meshes.size());
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> blasBuilds(meshes.size());
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> blasRanges(meshes.size());
    std::vector<VkDeviceAddress> blasAddresses(meshes.size());
    blasStorage_.resize(meshes.size());
    for (std::size_t i = 0; i < meshes.size(); ++i) {
        const MeshRange& mesh = meshes[i];
        blasGeometries[i] = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            .geometry = {.triangles = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                                       .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                                       .vertexData = {.deviceAddress = vertices_.address + mesh.firstVertex * sizeof(Vertex)},
                                       .vertexStride = sizeof(Vertex),
                                       .maxVertex = mesh.vertexCount - 1,
                                       .indexType = VK_INDEX_TYPE_UINT32,
                                       .indexData = {.deviceAddress = indices_.address +
                                                                      mesh.firstIndex * sizeof(std::uint32_t)}}},
            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
        };
        blasBuilds[i] = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
                         .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                         .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                         .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
                         .geometryCount = 1,
                         .pGeometries = &blasGeometries[i]};
        blasRanges[i] = {.primitiveCount = mesh.indexCount / 3};

        VkAccelerationStructureBuildSizesInfoKHR sizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkrt.getAccelerationStructureBuildSizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blasBuilds[i],
                                                &blasRanges[i].primitiveCount, &sizes);
        blas_.push_back(createStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                        sizes.accelerationStructureSize, blasStorage_[i]));
        blasBuilds[i].dstAccelerationStructure = blas_.back();
        blasAddresses[i] = addressOf(blas_.back());
        reserveScratch(sizes.buildScratchSize);
    }

    // The top-level structure places every instance (uniform scale + translation).
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    for (std::uint32_t i = 0; const Instance& instance : scene.instances) {
        const float s = instance.scale;
        const Vec3 p = instance.position;
        VkAccelerationStructureInstanceKHR& entry = tlasInstances.emplace_back();
        entry.transform = {{{s, 0.0f, 0.0f, p.x}, {0.0f, s, 0.0f, p.y}, {0.0f, 0.0f, s, p.z}}};
        entry.instanceCustomIndex = i++;  // indexes the instance data in the closest-hit shader
        entry.mask = 0xFF;
        entry.instanceShaderBindingTableRecordOffset = 0;
        entry.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        entry.accelerationStructureReference = blasAddresses.at(instance.mesh);
    }
    const Buffer tlasInput = upload.add(bytesOf(tlasInstances),
                                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

    const VkAccelerationStructureGeometryKHR tlasGeometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .geometry = {.instances = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                                   .data = {.deviceAddress = tlasInput.address}}},
    };
    VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = 1,
        .pGeometries = &tlasGeometry};
    const VkAccelerationStructureBuildRangeInfoKHR tlasRange{
        .primitiveCount = static_cast<std::uint32_t>(tlasInstances.size())};
    VkAccelerationStructureBuildSizesInfoKHR tlasSizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkrt.getAccelerationStructureBuildSizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuild,
                                            &tlasRange.primitiveCount, &tlasSizes);
    tlas_ = createStructure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, tlasSizes.accelerationStructureSize,
                            tlasStorage_);
    tlasBuild.dstAccelerationStructure = tlas_;
    reserveScratch(tlasSizes.buildScratchSize);

    const Buffer scratch = context_.createBuffer(scratchSize + scratchAlignment, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const VkDeviceAddress scratchBase = alignUp(scratch.address, scratchAlignment);
    for (std::size_t i = 0; i < blasBuilds.size(); ++i) {
        blasBuilds[i].scratchData.deviceAddress = scratchBase + scratchOffsets[i];
    }
    tlasBuild.scratchData.deviceAddress = scratchBase + scratchOffsets.back();

    // ---- Shader binding table: raygen | miss, shadow miss | hit group --------
    const auto& properties = context_.rayTracingProperties();
    const std::uint32_t handleSize = properties.shaderGroupHandleSize;
    const std::uint32_t handleStride = alignUp(handleSize, properties.shaderGroupHandleAlignment);
    const std::uint32_t baseAlignment = properties.shaderGroupBaseAlignment;
    std::vector<std::byte> handles(std::size_t{kGroupCount} * handleSize);
    check(vkrt.getRayTracingShaderGroupHandles(device, pipeline_, 0, kGroupCount, handles.size(), handles.data()));

    // Every region starts on the base alignment; ray generation's size must equal its stride.
    raygenRegion_ = {.stride = alignUp(handleStride, baseAlignment), .size = alignUp(handleStride, baseAlignment)};
    missRegion_ = {.stride = handleStride, .size = alignUp(2 * handleStride, baseAlignment)};
    hitRegion_ = {.stride = handleStride, .size = alignUp(handleStride, baseAlignment)};
    std::vector<std::byte> table(raygenRegion_.size + missRegion_.size + hitRegion_.size);
    const auto place = [&](ShaderGroup group, VkDeviceSize offset) {
        std::ranges::copy_n(handles.begin() + std::ptrdiff_t{group} * handleSize, handleSize,
                            table.begin() + static_cast<std::ptrdiff_t>(offset));
    };
    place(kRaygen, 0);
    place(kMiss, raygenRegion_.size);
    place(kShadowMiss, raygenRegion_.size + handleStride);
    place(kHitGroup, raygenRegion_.size + missRegion_.size);

    shaderBindingTable_ = upload.add(table, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR, baseAlignment);
    const VkDeviceAddress tableAddress = UploadBatch::dataAddress(shaderBindingTable_, baseAlignment);
    raygenRegion_.deviceAddress = tableAddress;
    missRegion_.deviceAddress = tableAddress + raygenRegion_.size;
    hitRegion_.deviceAddress = tableAddress + raygenRegion_.size + missRegion_.size;

    context_.submitAndWait([&](VkCommandBuffer commandBuffer) {
        upload.record(commandBuffer);
        // Acceleration-structure builds read their inputs as shader reads.
        memoryBarrier(commandBuffer, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_SHADER_READ_BIT);

        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePointers;
        for (const auto& range : blasRanges) rangePointers.push_back(&range);
        vkrt.cmdBuildAccelerationStructures(commandBuffer, static_cast<std::uint32_t>(blasBuilds.size()),
                                            blasBuilds.data(), rangePointers.data());

        memoryBarrier(commandBuffer, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                      VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                      VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                      VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        const VkAccelerationStructureBuildRangeInfoKHR* tlasRangePointer = &tlasRange;
        vkrt.cmdBuildAccelerationStructures(commandBuffer, 1, &tlasBuild, &tlasRangePointer);

        // Everything above is read by the ray-tracing stages from now on.
        memoryBarrier(commandBuffer, VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                      VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_2_MEMORY_READ_BIT);
    });

    std::println("Scene: {} meshes, {} instances, {} triangles", meshes.size(), scene.instances.size(),
                 indices.size() / 3);
}

void Renderer::createDescriptorSet() {
    const VkDevice device = context_.device();
    const std::array sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
    };
    const VkDescriptorPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                              .maxSets = 1,
                                              .poolSizeCount = static_cast<std::uint32_t>(sizes.size()),
                                              .pPoolSizes = sizes.data()};
    VkDescriptorPool pool = VK_NULL_HANDLE;
    check(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool));
    descriptorPool_ = {device, pool};

    const VkDescriptorSetAllocateInfo allocateInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                   .descriptorPool = pool,
                                                   .descriptorSetCount = 1,
                                                   .pSetLayouts = descriptorSetLayout_.address()};
    check(vkAllocateDescriptorSets(device, &allocateInfo, &descriptorSet_));

    // The images are bound in createImages(), once the window size is known.
    const VkWriteDescriptorSetAccelerationStructureKHR scene{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = tlas_.address()};
    const VkDescriptorBufferInfo vertices{vertices_.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo indices{indices_.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo instances{instances_.buffer, 0, VK_WHOLE_SIZE};
    const auto write = [&](std::uint32_t binding, VkDescriptorType type, const VkDescriptorBufferInfo* buffer) {
        return VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                    .dstSet = descriptorSet_,
                                    .dstBinding = binding,
                                    .descriptorCount = 1,
                                    .descriptorType = type,
                                    .pBufferInfo = buffer};
    };
    std::array writes{
        write(gpu::kBindingScene, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr),
        write(gpu::kBindingVertices, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &vertices),
        write(gpu::kBindingIndices, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &indices),
        write(gpu::kBindingInstances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &instances),
    };
    writes[0].pNext = &scene;
    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void Renderer::createFrames() {
    const VkDevice device = context_.device();
    const VkCommandPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                           .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                           .queueFamilyIndex = context_.queueFamily()};
    VkCommandPool pool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(device, &poolInfo, nullptr, &pool));
    commandPool_ = {device, pool};

    std::array<VkCommandBuffer, kFramesInFlight> commandBuffers{};
    const VkCommandBufferAllocateInfo allocateInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                   .commandPool = pool,
                                                   .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                   .commandBufferCount = kFramesInFlight};
    check(vkAllocateCommandBuffers(device, &allocateInfo, commandBuffers.data()));

    const VkSemaphoreCreateInfo semaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    const VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                      .flags = VK_FENCE_CREATE_SIGNALED_BIT};
    for (std::uint32_t i = 0; i < kFramesInFlight; ++i) {
        Frame& frame = frames_[i];
        frame.commandBuffer = commandBuffers[i];
        VkSemaphore semaphore = VK_NULL_HANDLE;
        check(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore));
        frame.imageAcquired = {device, semaphore};
        VkFence fence = VK_NULL_HANDLE;
        check(vkCreateFence(device, &fenceInfo, nullptr, &fence));
        frame.done = {device, fence};
    }
}

// ---- Window size -------------------------------------------------------------

bool Renderer::rebuildSwapchain(VkExtent2D windowSize) {
    check(vkDeviceWaitIdle(context_.device()));
    if (!swapchain_.recreate(windowSize)) return false;
    swapchainValid_ = true;
    windowSize_ = windowSize;
    if (swapchain_.extent() != imageExtent_) {
        createImages(swapchain_.extent());
        resetAccumulation();
    }
    return true;
}

void Renderer::createImages(VkExtent2D extent) {
    output_ = context_.createImage(extent, kOutputFormat,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    accumulation_ = context_.createImage(extent, kAccumulationFormat, VK_IMAGE_USAGE_STORAGE_BIT);
    imageExtent_ = extent;

    // The accumulation image stays in GENERAL; the output image is transitioned every frame.
    context_.submitAndWait([&](VkCommandBuffer commandBuffer) {
        const auto barrier = imageBarrier(accumulation_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                          VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                                          VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        pipelineBarrier(commandBuffer, {&barrier, 1});
    });

    const VkDescriptorImageInfo output{.imageView = output_.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo accumulation{.imageView = accumulation_.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    const auto write = [&](std::uint32_t binding, const VkDescriptorImageInfo* image) {
        return VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                    .dstSet = descriptorSet_,
                                    .dstBinding = binding,
                                    .descriptorCount = 1,
                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                    .pImageInfo = image};
    };
    const std::array writes{write(gpu::kBindingOutput, &output), write(gpu::kBindingAccumulation, &accumulation)};
    vkUpdateDescriptorSets(context_.device(), static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

// ---- Frame -------------------------------------------------------------------

bool Renderer::render(const CameraView& view, VkExtent2D windowSize) {
    if (windowSize.width == 0 || windowSize.height == 0) return false;  // minimised
    if (!swapchainValid_ || windowSize != windowSize_) {
        if (!rebuildSwapchain(windowSize)) return false;
    }

    const VkDevice device = context_.device();
    const Frame& frame = frames_[frameIndex_ % kFramesInFlight];
    check(vkWaitForFences(device, 1, frame.done.address(), VK_TRUE, UINT64_MAX));

    std::uint32_t imageIndex = 0;
    const VkResult acquired = vkAcquireNextImageKHR(device, swapchain_.handle(), UINT64_MAX, frame.imageAcquired,
                                                    VK_NULL_HANDLE, &imageIndex);
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchainValid_ = false;
        return false;
    }
    if (acquired != VK_SUBOPTIMAL_KHR) check(acquired);

    // Reset the fence only now that a submission is certain to signal it again.
    check(vkResetFences(device, 1, frame.done.address()));
    check(vkResetCommandBuffer(frame.commandBuffer, 0));
    record(frame.commandBuffer, imageIndex, view);

    // Ray tracing starts right away; only the blit waits for the swapchain image.
    const VkSemaphoreSubmitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                     .semaphore = frame.imageAcquired,
                                     .stageMask = VK_PIPELINE_STAGE_2_BLIT_BIT};
    const VkSemaphore renderFinished = swapchain_.renderFinished(imageIndex);
    const VkSemaphoreSubmitInfo signal{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                       .semaphore = renderFinished,
                                       .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    const VkCommandBufferSubmitInfo commands{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                             .commandBuffer = frame.commandBuffer};
    const VkSubmitInfo2 submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                               .waitSemaphoreInfoCount = 1,
                               .pWaitSemaphoreInfos = &wait,
                               .commandBufferInfoCount = 1,
                               .pCommandBufferInfos = &commands,
                               .signalSemaphoreInfoCount = 1,
                               .pSignalSemaphoreInfos = &signal};
    check(vkQueueSubmit2(context_.queue(), 1, &submit, frame.done));

    const VkSwapchainKHR swapchain = swapchain_.handle();
    const VkPresentInfoKHR present{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                   .waitSemaphoreCount = 1,
                                   .pWaitSemaphores = &renderFinished,
                                   .swapchainCount = 1,
                                   .pSwapchains = &swapchain,
                                   .pImageIndices = &imageIndex};
    const VkResult presented = vkQueuePresentKHR(context_.queue(), &present);
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR || acquired == VK_SUBOPTIMAL_KHR) {
        swapchainValid_ = false;  // rebuilt at the start of the next frame
    } else {
        check(presented);
    }

    ++frameIndex_;
    ++accumulatedFrames_;
    return true;
}

void Renderer::record(VkCommandBuffer commandBuffer, std::uint32_t imageIndex, const CameraView& view) const {
    const VkCommandBufferBeginInfo beginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                             .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    check(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    // The previous frame's blit must finish reading the output image before it
    // is overwritten, and its average must be written before it is read again.
    const std::array beforeTrace{
        imageBarrier(output_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                     VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
        imageBarrier(accumulation_.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
    };
    pipelineBarrier(commandBuffer, beforeTrace);

    const gpu::FrameConstants constants{
        .eye = toVec4(view.eye),
        .forward = toVec4(view.forward),
        .right = toVec4(view.right),
        .up = toVec4(view.up),
        .light = light_,
        .counters = {accumulatedFrames_, static_cast<std::uint32_t>(frameIndex_), settings_.samplesPerFrame,
                     settings_.maxBounces},
    };
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout_, 0, 1,
                            &descriptorSet_, 0, nullptr);
    vkCmdPushConstants(commandBuffer, pipelineLayout_, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(constants),
                       &constants);
    vkrt.cmdTraceRays(commandBuffer, &raygenRegion_, &missRegion_, &hitRegion_, &callableRegion_,
                      imageExtent_.width, imageExtent_.height, 1);

    // The swapchain image barrier uses the blit stage that the submission's
    // acquire wait is tied to, so its layout change happens after the wait.
    const VkImage target = swapchain_.image(imageIndex);
    const std::array beforeBlit{
        imageBarrier(output_.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT),
        imageBarrier(target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_BLIT_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT),
    };
    pipelineBarrier(commandBuffer, beforeBlit);

    // A blit (not a copy) converts between RGBA and the swapchain's BGRA if needed.
    const VkOffset3D corner{static_cast<std::int32_t>(imageExtent_.width),
                            static_cast<std::int32_t>(imageExtent_.height), 1};
    const VkImageBlit2 region{.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
                              .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .srcOffsets = {{0, 0, 0}, corner},
                              .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .dstOffsets = {{0, 0, 0}, corner}};
    const VkBlitImageInfo2 blit{.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
                                .srcImage = output_.image,
                                .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                .dstImage = target,
                                .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                .regionCount = 1,
                                .pRegions = &region,
                                .filter = VK_FILTER_NEAREST};
    vkCmdBlitImage2(commandBuffer, &blit);

    // The second scope must meet the semaphore signal (ALL_COMMANDS) so that the
    // layout change is ordered before presentation.
    const auto toPresent = imageBarrier(target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                        VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE);
    pipelineBarrier(commandBuffer, {&toPresent, 1});
    check(vkEndCommandBuffer(commandBuffer));
}

#include "Scene.h"

#include <array>
#include <cmath>
#include <numbers>

namespace {

constexpr float kPi = std::numbers::pi_v<float>;

// Unit sphere centred at the origin; the normal equals the position.
Mesh makeSphere(std::uint32_t stacks, std::uint32_t slices) {
    Mesh mesh;
    mesh.vertices.reserve((stacks + 1) * (slices + 1));
    for (std::uint32_t i = 0; i <= stacks; ++i) {
        const float phi = kPi * static_cast<float>(i) / static_cast<float>(stacks);  // 0..pi from the north pole
        for (std::uint32_t j = 0; j <= slices; ++j) {
            const float theta = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(slices);
            const Vec3 n{std::sin(phi) * std::cos(theta), std::cos(phi), std::sin(phi) * std::sin(theta)};
            mesh.vertices.push_back({n, n});
        }
    }

    const std::uint32_t ring = slices + 1;
    for (std::uint32_t i = 0; i < stacks; ++i) {
        for (std::uint32_t j = 0; j < slices; ++j) {
            const std::uint32_t a = i * ring + j;
            const std::uint32_t b = a + ring;
            // The first and last stacks meet at a pole: skip their zero-area triangles.
            if (i != 0) mesh.indices.insert(mesh.indices.end(), {a, b, a + 1});
            if (i != stacks - 1) mesh.indices.insert(mesh.indices.end(), {a + 1, b, b + 1});
        }
    }
    return mesh;
}

// Square in the y = 0 plane.
Mesh makeFloor(float halfSize) {
    const Vec3 up{0.0f, 1.0f, 0.0f};
    return {
        .vertices = {{{-halfSize, 0.0f, -halfSize}, up},
                     {{halfSize, 0.0f, -halfSize}, up},
                     {{halfSize, 0.0f, halfSize}, up},
                     {{-halfSize, 0.0f, halfSize}, up}},
        .indices = {0, 1, 2, 0, 2, 3},
    };
}

} // namespace

Scene createDemoScene() {
    enum : std::uint32_t { kFloor, kSphere };

    Scene scene;
    scene.meshes = {makeFloor(20.0f), makeSphere(64, 128)};
    scene.light = {.center = {4.0f, 7.0f, -2.0f}, .radius = 1.2f};

    scene.instances.push_back({.mesh = kFloor,
                               .albedo = {0.8f, 0.8f, 0.8f},
                               .reflectivity = 0.35f,
                               .material = gpu::kMaterialChecker});
    scene.instances.push_back({.mesh = kSphere,
                               .position = {0.0f, 1.4f, 0.0f},
                               .scale = 1.4f,
                               .albedo = {1.0f, 1.0f, 1.0f},
                               .material = gpu::kMaterialGlass});

    constexpr std::array<Vec3, 6> palette{{
        {0.90f, 0.20f, 0.20f},
        {0.95f, 0.65f, 0.15f},
        {0.90f, 0.90f, 0.20f},
        {0.20f, 0.80f, 0.30f},
        {0.20f, 0.45f, 0.90f},
        {0.65f, 0.25f, 0.85f},
    }};
    for (std::size_t i = 0; i < palette.size(); ++i) {
        const float angle = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(palette.size());
        scene.instances.push_back({.mesh = kSphere,
                                   .position = {4.0f * std::cos(angle), 0.8f, 4.0f * std::sin(angle)},
                                   .scale = 0.8f,
                                   .albedo = palette[i],
                                   .reflectivity = i % 2 == 0 ? 0.1f : 0.45f});  // alternate matte and glossy
    }

    scene.instances.push_back({.mesh = kSphere,
                               .position = scene.light.center,
                               .scale = scene.light.radius,
                               .albedo = {1.0f, 0.95f, 0.85f},
                               .material = gpu::kMaterialEmissive});
    return scene;
}

#pragma once
// The demo scene: two meshes (floor, unit sphere) placed by instances.

#include <cstdint>
#include <vector>

#include "../shaders/shared.h"
#include "Math.h"

struct Vertex {
    Vec3 position;
    Vec3 normal;
};
static_assert(sizeof(Vertex) == gpu::kVertexStride * sizeof(float));

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

// A mesh placed in the world with a uniform scale and a material.
struct Instance {
    std::uint32_t mesh = 0;
    Vec3 position;
    float scale = 1.0f;
    Vec3 albedo;
    float reflectivity = 0.0f;
    std::uint32_t material = gpu::kMaterialSolid;
};

// The spherical area light; it is also an emissive instance in the scene.
struct SphereLight {
    Vec3 center;
    float radius = 1.0f;
};

struct Scene {
    std::vector<Mesh> meshes;
    std::vector<Instance> instances;
    SphereLight light;
};

// Reflective checkerboard floor, a dispersive glass sphere in the middle, a ring
// of six coloured spheres and a visible spherical area light.
[[nodiscard]] Scene createDemoScene();

// Definitions shared by the C++ host and the GLSL shaders.
//
// Only 4-component vectors and 32-bit scalars are used, so the C++ structs and
// the std430 GLSL layouts match by construction (the host static_asserts it).

#ifndef SHARED_H
#define SHARED_H

#ifdef __cplusplus
#include <cstdint>
namespace gpu {
using uint = std::uint32_t;
struct vec4 { float x, y, z, w; };
struct uvec4 { uint x, y, z, w; };
#define SHARED_CONST inline constexpr
#else
#define SHARED_CONST const
#endif

// Descriptor set 0.
SHARED_CONST uint kBindingScene = 0u;          // top-level acceleration structure
SHARED_CONST uint kBindingOutput = 1u;         // rgba8 image shown on screen
SHARED_CONST uint kBindingAccumulation = 2u;   // rgba32f running average
SHARED_CONST uint kBindingVertices = 3u;
SHARED_CONST uint kBindingIndices = 4u;
SHARED_CONST uint kBindingInstances = 5u;

// Material types.
SHARED_CONST uint kMaterialChecker = 0u;       // procedural checkerboard (the floor)
SHARED_CONST uint kMaterialSolid = 1u;         // diffuse with optional mirror reflection
SHARED_CONST uint kMaterialGlass = 2u;         // dispersive dielectric
SHARED_CONST uint kMaterialEmissive = 3u;      // the visible area light

// A vertex is 6 floats: position xyz, normal xyz (object space).
SHARED_CONST uint kVertexStride = 6u;

// Push constants of the ray-generation shader (96 bytes, within the
// guaranteed 128).
struct FrameConstants {
    vec4 eye;       // xyz camera position
    vec4 forward;   // xyz unit view direction
    vec4 right;     // xyz unit right * tan(fovY / 2) * aspect
    vec4 up;        // xyz unit up * tan(fovY / 2)
    vec4 light;     // xyz centre, w radius of the spherical area light
    uvec4 counters; // x accumulated frames (0 restarts), y frame index,
                    // z samples per frame, w maximum bounces
};

// Per-instance data, indexed by gl_InstanceCustomIndexEXT.
struct InstanceData {
    vec4 albedo;    // rgb albedo, a reflectivity
    uvec4 geometry; // x material, y first index, z vertex offset, w unused
};

#ifdef __cplusplus
} // namespace gpu
#endif

#undef SHARED_CONST

#endif // SHARED_H

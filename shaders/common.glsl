// Declarations shared by all ray-tracing stages.

#ifndef COMMON_GLSL
#define COMMON_GLSL

#include "shared.h"

// Payload of the primary/bounce rays (location 0).
struct HitPayload {
    vec3 normal;        // shading normal, facing the incoming ray
    float t;            // hit distance; negative on a miss
    vec3 albedo;
    float reflectivity;
    uint material;      // kMaterial*
    uint frontFace;     // 1 if the ray hit the outer surface, 0 from inside
};

#endif // COMMON_GLSL

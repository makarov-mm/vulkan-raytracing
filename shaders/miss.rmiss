#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload hit;

void main() {
    // The ray escaped; the ray-generation shader adds the sky colour.
    hit.t = -1.0;
}

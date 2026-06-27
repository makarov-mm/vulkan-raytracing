#version 460
#extension GL_EXT_ray_tracing : require

struct HitPayload {
    vec3  hitPos;
    vec3  normal;
    vec3  albedo;
    float reflectivity;
    float matId;
    float t;
    float frontFace;
};

layout(location = 0) rayPayloadInEXT HitPayload payload;

void main() {
    // Negative t signals "no hit"; the ray-gen shader adds the sky colour.
    payload.t = -1.0;
}

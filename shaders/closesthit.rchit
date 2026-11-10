#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload hit;
hitAttributeEXT vec2 barycentrics;

layout(set = 0, binding = kBindingVertices, std430) readonly buffer Vertices { float vertexData[]; };
layout(set = 0, binding = kBindingIndices, std430) readonly buffer Indices { uint indexData[]; };
layout(set = 0, binding = kBindingInstances, std430) readonly buffer Instances { InstanceData instances[]; };

// Three consecutive floats of a vertex: offset 0 = position, 3 = normal.
vec3 vertexVector(uint vertex, uint offset) {
    uint i = vertex * kVertexStride + offset;
    return vec3(vertexData[i], vertexData[i + 1u], vertexData[i + 2u]);
}

void main() {
    InstanceData instance = instances[gl_InstanceCustomIndexEXT];
    uint first = instance.geometry.y + 3u * uint(gl_PrimitiveID);
    uvec3 triangle = uvec3(indexData[first], indexData[first + 1u], indexData[first + 2u]) + instance.geometry.z;
    vec3 w = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics);

    vec3 p0 = vertexVector(triangle.x, 0u);
    vec3 p1 = vertexVector(triangle.y, 0u);
    vec3 p2 = vertexVector(triangle.z, 0u);
    vec3 shading = vertexVector(triangle.x, 3u) * w.x + vertexVector(triangle.y, 3u) * w.y + vertexVector(triangle.z, 3u) * w.z;

    // Instances use uniform scale, so normals transform with the linear part.
    mat3 toWorld = mat3(gl_ObjectToWorldEXT);
    shading = normalize(toWorld * shading);
    vec3 geometric = normalize(toWorld * cross(p1 - p0, p2 - p0));
    if (dot(geometric, shading) < 0.0) geometric = -geometric;  // outward, like the shading normal

    // Inside/outside comes from the flat triangle: near silhouettes the
    // interpolated normal can point the wrong way.
    vec3 dir = gl_WorldRayDirectionEXT;
    bool front = dot(geometric, dir) < 0.0;
    vec3 n = front ? shading : -shading;
    if (dot(n, dir) > 0.0) n = front ? geometric : -geometric;

    hit.normal = n;
    hit.t = gl_HitTEXT;
    hit.albedo = instance.albedo.rgb;
    hit.reflectivity = instance.albedo.a;
    hit.material = instance.geometry.x;
    hit.frontFace = front ? 1u : 0u;
}

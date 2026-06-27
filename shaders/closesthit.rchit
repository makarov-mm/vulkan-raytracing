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
hitAttributeEXT vec2 attribs; // barycentric coordinates of the hit

// Vertex layout matches the C++ side (std430, vec4-padded for safety):
//   posRefl : xyz = position,  w = reflectivity
//   normal  : xyz = normal,    w = matId (0 = floor, 1 = object)
//   color   : xyz = albedo,    w = unused
struct Vertex {
    vec4 posRefl;
    vec4 normal;
    vec4 color;
};

layout(std430, set = 0, binding = 3) readonly buffer Vertices { Vertex v[]; } vertices;
layout(std430, set = 0, binding = 4) readonly buffer Indices  { uint   i[]; } indices;

void main() {
    uint i0 = indices.i[3 * gl_PrimitiveID + 0];
    uint i1 = indices.i[3 * gl_PrimitiveID + 1];
    uint i2 = indices.i[3 * gl_PrimitiveID + 2];

    Vertex a = vertices.v[i0];
    Vertex b = vertices.v[i1];
    Vertex c = vertices.v[i2];

    const vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    // Geometry is baked in world space with an identity instance transform,
    // so interpolated normals are already in world space.
    vec3 N = normalize(a.normal.xyz * bary.x + b.normal.xyz * bary.y + c.normal.xyz * bary.z);
    vec3 col = a.color.xyz * bary.x + b.color.xyz * bary.y + c.color.xyz * bary.z;

    // Flip the normal so it always faces the incoming ray, and record whether
    // we hit the outer (front) surface — needed for glass refraction.
    bool front = dot(N, gl_WorldRayDirectionEXT) < 0.0;
    if (!front) N = -N;

    vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    payload.hitPos       = hitPos;
    payload.normal       = N;
    payload.albedo       = col;
    payload.reflectivity = a.posRefl.w;
    payload.matId        = a.normal.w;
    payload.t            = gl_HitTEXT;
    payload.frontFace    = front ? 1.0 : 0.0;
}

#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT float shadowed;

void main() {
    // If the shadow ray reaches here, nothing blocked the light.
    shadowed = 0.0;
}

#pragma once
// Orbit camera around the middle of the scene.

#include <algorithm>
#include <cmath>
#include <numbers>

#include "Math.h"

// What the ray-generation shader needs: a pixel at NDC (x, y), y pointing
// down, looks along forward + x * right - y * up.
struct CameraView {
    Vec3 eye;
    Vec3 forward;
    Vec3 right;  // scaled by tan(fovY / 2) * aspect
    Vec3 up;     // scaled by tan(fovY / 2)
};

class OrbitCamera {
public:
    // Orbits by a mouse drag of (dx, dy) pixels.
    void rotate(float dx, float dy) {
        yaw_ = std::remainder(yaw_ + dx * kRadiansPerPixel, 2.0f * std::numbers::pi_v<float>);
        pitch_ = std::clamp(pitch_ + dy * kRadiansPerPixel, -kMaxPitch, kMaxPitch);
    }

    // Zooms in for positive wheel notches, out for negative ones.
    void zoom(float notches) {
        distance_ = std::clamp(distance_ * std::pow(kZoomPerNotch, notches), kMinDistance, kMaxDistance);
    }

    [[nodiscard]] CameraView view(float aspect) const {
        const float cosPitch = std::cos(pitch_);
        const Vec3 offset{cosPitch * std::cos(yaw_), std::sin(pitch_), cosPitch * std::sin(yaw_)};
        const Vec3 eye = kTarget + offset * distance_;
        const Vec3 forward = -offset;
        const Vec3 right = normalize(cross(forward, Vec3{0.0f, 1.0f, 0.0f}));
        const Vec3 up = cross(right, forward);
        const float tanHalfFov = std::tan(kFovY * 0.5f);
        return {eye, forward, right * (tanHalfFov * aspect), up * tanHalfFov};
    }

private:
    static constexpr Vec3 kTarget{0.0f, 1.0f, 0.0f};
    static constexpr float kFovY = std::numbers::pi_v<float> / 3.0f;  // 60 degrees
    static constexpr float kRadiansPerPixel = 0.005f;
    static constexpr float kMaxPitch = 1.5f;                            // ~86 degrees
    static constexpr float kZoomPerNotch = 0.9f;
    static constexpr float kMinDistance = 3.0f;
    static constexpr float kMaxDistance = 30.0f;

    float yaw_ = 2.2f;
    float pitch_ = 0.42f;
    float distance_ = 9.0f;
};

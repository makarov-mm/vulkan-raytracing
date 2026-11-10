// Hardware-accelerated Vulkan ray tracing (VK_KHR_ray_tracing_pipeline) in a
// plain Win32 window: no GLFW, no GLM, nothing beyond the Vulkan SDK.

#include <chrono>
#include <cstdio>
#include <exception>
#include <format>
#include <print>
#include <string>
#include <utility>

#include "Camera.h"
#include "Renderer.h"
#include "Scene.h"
#include "VulkanContext.h"
#include "Window.h"

namespace {

constexpr wchar_t kTitle[] = L"Vulkan Hardware Ray Tracing";
constexpr int kWindowWidth = 800;  // client size at 100% display scale
constexpr int kWindowHeight = 800;
constexpr RenderSettings kSettings{.samplesPerFrame = 6, .maxBounces = 8, .targetSamples = 4096};
constexpr std::chrono::milliseconds kTitleInterval{500};

class App {
public:
    App()
        : window_{kTitle, kWindowWidth, kWindowHeight},
          context_{window_.instance(), window_.handle()},
          renderer_{context_, createDemoScene(), kSettings, executableDirectory() / "shaders"},
          gpuName_{widen(context_.deviceName())} {}

    void run() {
        using Clock = std::chrono::steady_clock;
        auto statsStart = Clock::now();
        int statsFrames = 0;
        VkExtent2D lastSize{};

        while (window_.processMessages()) {
            const Input input = window_.takeInput();
            if (!input.empty()) {
                camera_.rotate(input.dragX, input.dragY);
                camera_.zoom(input.wheel);
                renderer_.resetAccumulation();
            }

            const VkExtent2D size{window_.width(), window_.height()};
            const bool resized = size != lastSize;
            lastSize = size;

            // Minimised, or converged with nothing new to show: sleep until a message arrives.
            const bool minimised = size.width == 0 || size.height == 0;
            if (minimised || (renderer_.converged() && !resized)) {
                updateTitle(0.0);
                window_.waitForMessages();
                statsStart = Clock::now();
                statsFrames = 0;
                continue;
            }

            const float aspect = static_cast<float>(size.width) / static_cast<float>(size.height);
            if (renderer_.render(camera_.view(aspect), size)) ++statsFrames;

            const auto now = Clock::now();
            if (now - statsStart >= kTitleInterval && statsFrames > 0) {
                const std::chrono::duration<double, std::milli> elapsed = now - statsStart;
                updateTitle(elapsed.count() / statsFrames);
                statsStart = now;
                statsFrames = 0;
            }
        }
    }

private:
    void updateTitle(double millisecondsPerFrame) {
        const std::uint32_t samples = renderer_.samplesPerPixel();
        const std::wstring status = renderer_.converged() || millisecondsPerFrame <= 0.0
                                        ? std::format(L"{} spp", samples)
                                        : std::format(L"{} spp \u00B7 {:.1f} ms/frame", samples, millisecondsPerFrame);
        std::wstring title = std::format(L"{} \u2014 {} \u00B7 {}", kTitle, gpuName_, status);
        // Setting the same text again would still repaint the caption and wake the idle loop.
        if (title != title_) {
            title_ = std::move(title);
            window_.setTitle(title_);
        }
    }

    // Declaration order is construction order; destruction runs in reverse.
    Window window_;
    VulkanContext context_;
    Renderer renderer_;
    OrbitCamera camera_;
    std::wstring gpuName_;
    std::wstring title_;
};

// Debug builds always get a console (for validation output); release builds
// print only when started from one. Without a console the standard streams
// are sent to NUL: std::print throws when a stream leads nowhere.
void setUpConsole() {
#ifdef NDEBUG
    const bool console = AttachConsole(ATTACH_PARENT_PROCESS) != FALSE;
#else
    const bool console = AttachConsole(ATTACH_PARENT_PROCESS) != FALSE || AllocConsole() != FALSE;
#endif
    const char* target = console ? "CONOUT$" : "NUL";
    FILE* stream = nullptr;
    freopen_s(&stream, target, "w", stdout);
    freopen_s(&stream, target, "w", stderr);
}

} // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);  // crisp on high-DPI displays
    setUpConsole();
    try {
        App app;
        app.run();
    } catch (const std::exception& error) {
        std::println(stderr, "Fatal error: {}", error.what());
        MessageBoxW(nullptr, widen(error.what()).c_str(), L"Vulkan Ray Tracing \u2014 error", MB_OK | MB_ICONERROR);
        return 1;
    }
    return 0;
}

#pragma once
// Win32 window, input and a few platform helpers.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <filesystem>
#include <string>
#include <string_view>

// Mouse input gathered since the last call to Window::takeInput().
struct Input {
    float dragX = 0.0f;  // pixels dragged with the left button held
    float dragY = 0.0f;
    float wheel = 0.0f;  // wheel notches, positive away from the user

    [[nodiscard]] bool empty() const { return dragX == 0.0f && dragY == 0.0f && wheel == 0.0f; }
};

class Window {
public:
    // `width` x `height` is the client size at 100% scaling; it grows with the monitor DPI.
    Window(std::wstring_view title, int width, int height);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Dispatches pending messages; returns false once the window should close.
    bool processMessages();
    // Blocks until a message arrives (used while idle).
    void waitForMessages() const;
    [[nodiscard]] Input takeInput();

    [[nodiscard]] HWND handle() const { return hwnd_; }
    [[nodiscard]] HINSTANCE instance() const { return instance_; }
    // Client area in pixels; zero while minimised.
    [[nodiscard]] UINT width() const;
    [[nodiscard]] UINT height() const;
    void setTitle(const std::wstring& title) const;

private:
    static LRESULT CALLBACK windowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
    LRESULT handleMessage(UINT message, WPARAM wParam, LPARAM lParam);

    HINSTANCE instance_ = nullptr;
    HWND hwnd_ = nullptr;
    bool open_ = true;
    bool dragging_ = false;
    POINT lastCursor_{};
    Input input_;
};

// UTF-8 <-> UTF-16 for the Win32 API.
[[nodiscard]] std::wstring widen(std::string_view utf8);
[[nodiscard]] std::string narrow(std::wstring_view utf16);

// Folder of the running executable (shaders are looked up next to it).
[[nodiscard]] std::filesystem::path executableDirectory();

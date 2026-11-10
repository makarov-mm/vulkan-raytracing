#include "Window.h"

#include <windowsx.h>

#include <stdexcept>
#include <utility>

namespace {

constexpr wchar_t kClassName[] = L"VulkanRayTracingWindow";
constexpr DWORD kStyle = WS_OVERLAPPEDWINDOW;
constexpr LONG kMinTrackSize = 160;  // keeps the swapchain from degenerating

} // namespace

Window::Window(std::wstring_view title, int width, int height) : instance_{GetModuleHandleW(nullptr)} {
    const WNDCLASSEXW windowClass{
        .cbSize = sizeof(WNDCLASSEXW),
        .lpfnWndProc = windowProc,
        .hInstance = instance_,
        .hCursor = LoadCursorW(nullptr, IDC_ARROW),
        .lpszClassName = kClassName,
    };
    if (!RegisterClassExW(&windowClass)) throw std::runtime_error("RegisterClassExW failed");

    // The requested client size is at 100% scaling; grow it with the display scale.
    const UINT dpi = GetDpiForSystem();
    RECT rect{0, 0, MulDiv(width, static_cast<int>(dpi), USER_DEFAULT_SCREEN_DPI),
              MulDiv(height, static_cast<int>(dpi), USER_DEFAULT_SCREEN_DPI)};
    AdjustWindowRectExForDpi(&rect, kStyle, FALSE, 0, dpi);

    const std::wstring caption{title};
    CreateWindowExW(0, kClassName, caption.c_str(), kStyle, CW_USEDEFAULT, CW_USEDEFAULT,
                    rect.right - rect.left, rect.bottom - rect.top, nullptr, nullptr, instance_, this);
    if (!hwnd_) {
        UnregisterClassW(kClassName, instance_);
        throw std::runtime_error("CreateWindowExW failed");
    }
    ShowWindow(hwnd_, SW_SHOWDEFAULT);
}

Window::~Window() {
    DestroyWindow(hwnd_);
    UnregisterClassW(kClassName, instance_);
}

bool Window::processMessages() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        if (message.message == WM_QUIT) open_ = false;
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    return open_;
}

void Window::waitForMessages() const { WaitMessage(); }

Input Window::takeInput() { return std::exchange(input_, {}); }

UINT Window::width() const {
    RECT rect;
    GetClientRect(hwnd_, &rect);
    return static_cast<UINT>(rect.right - rect.left);
}

UINT Window::height() const {
    RECT rect;
    GetClientRect(hwnd_, &rect);
    return static_cast<UINT>(rect.bottom - rect.top);
}

void Window::setTitle(const std::wstring& title) const { SetWindowTextW(hwnd_, title.c_str()); }

LRESULT CALLBACK Window::windowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    if (message == WM_NCCREATE) {
        auto* self = static_cast<Window*>(reinterpret_cast<const CREATESTRUCTW*>(lParam)->lpCreateParams);
        self->hwnd_ = hwnd;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
    }
    if (auto* self = reinterpret_cast<Window*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA))) {
        return self->handleMessage(message, wParam, lParam);
    }
    return DefWindowProcW(hwnd, message, wParam, lParam);
}

LRESULT Window::handleMessage(UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CLOSE:
        open_ = false;  // the owner tears everything down, then destroys the window
        return 0;
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) open_ = false;
        return 0;
    case WM_LBUTTONDOWN:
        dragging_ = true;
        lastCursor_ = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
        SetCapture(hwnd_);
        return 0;
    case WM_LBUTTONUP:
        ReleaseCapture();  // sends WM_CAPTURECHANGED
        return 0;
    case WM_CAPTURECHANGED:  // also when the capture is lost, e.g. on Alt+Tab
        dragging_ = false;
        return 0;
    case WM_MOUSEMOVE:
        if (dragging_) {
            const POINT cursor{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
            input_.dragX += static_cast<float>(cursor.x - lastCursor_.x);
            input_.dragY += static_cast<float>(cursor.y - lastCursor_.y);
            lastCursor_ = cursor;
        }
        return 0;
    case WM_MOUSEWHEEL:
        input_.wheel += static_cast<float>(GET_WHEEL_DELTA_WPARAM(wParam)) / WHEEL_DELTA;
        return 0;
    case WM_DPICHANGED: {
        const auto* suggested = reinterpret_cast<const RECT*>(lParam);
        SetWindowPos(hwnd_, nullptr, suggested->left, suggested->top, suggested->right - suggested->left,
                     suggested->bottom - suggested->top, SWP_NOZORDER | SWP_NOACTIVATE);
        return 0;
    }
    case WM_GETMINMAXINFO:
        reinterpret_cast<MINMAXINFO*>(lParam)->ptMinTrackSize = {kMinTrackSize, kMinTrackSize};
        return 0;
    default:
        return DefWindowProcW(hwnd_, message, wParam, lParam);
    }
}

std::wstring widen(std::string_view utf8) {
    if (utf8.empty()) return {};
    const int size = static_cast<int>(utf8.size());
    std::wstring result(static_cast<std::size_t>(MultiByteToWideChar(CP_UTF8, 0, utf8.data(), size, nullptr, 0)), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), size, result.data(), static_cast<int>(result.size()));
    return result;
}

std::string narrow(std::wstring_view utf16) {
    if (utf16.empty()) return {};
    const int size = static_cast<int>(utf16.size());
    std::string result(
        static_cast<std::size_t>(WideCharToMultiByte(CP_UTF8, 0, utf16.data(), size, nullptr, 0, nullptr, nullptr)),
        '\0');
    WideCharToMultiByte(CP_UTF8, 0, utf16.data(), size, result.data(), static_cast<int>(result.size()), nullptr,
                        nullptr);
    return result;
}

std::filesystem::path executableDirectory() {
    std::wstring path(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (length < path.size()) {
            path.resize(length);
            return std::filesystem::path{path}.parent_path();
        }
        path.resize(path.size() * 2);  // truncated: retry with a larger buffer
    }
}

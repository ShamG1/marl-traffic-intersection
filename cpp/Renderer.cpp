#include "Renderer.h"
#include "IntersectionEnv.h"
#include "RenderColors.h"

#include <GLFW/glfw3.h>

#define NOMINMAX
#define GLFW_EXPOSE_NATIVE_WIN32
#include <Windows.h>
#include <GLFW/glfw3native.h>

#include <cmath>
#include <iostream>
#include <array>
#include <cstdio>
#include <algorithm>
#include <vector>

struct Renderer::Impl {
    GLFWwindow* window{nullptr};
    HWND hwnd{nullptr};
    HDC hdc{nullptr};
    HFONT font{nullptr};
    HFONT font_lane_ids{nullptr};
    HICON hicon_small{nullptr};
    HICON hicon_big{nullptr};
    int fb_w{0};
    int fb_h{0};
};

// NDC mapping is based on fixed logical coordinates (WIDTH x HEIGHT).
// Window resizing is handled via glViewport letterbox/pillarbox.
static inline float ndc_x(float px){return px/(WIDTH*0.5f)-1.0f;}
static inline float ndc_y(float py){return 1.0f - py/(HEIGHT*0.5f);} // invert y

static void draw_rect_ndc(float x_px,float y_px,float w_px,float h_px, float r,float g,float b,float a=1.0f){
    float x0 = ndc_x(x_px);
    float y0 = ndc_y(y_px);
    float x1 = ndc_x(x_px + w_px);
    float y1 = ndc_y(y_px + h_px);
    glColor4f(r,g,b,a);
    glBegin(GL_QUADS);
    glVertex2f(x0,y0);
    glVertex2f(x1,y0);
    glVertex2f(x1,y1);
    glVertex2f(x0,y1);
    glEnd();
}

static void draw_line_px(float x0,float y0,float x1,float y1,float width, float r,float g,float b,float a=1.0f){
    glLineWidth(width);
    glColor4f(r,g,b,a);
    glBegin(GL_LINES);
    glVertex2f(ndc_x(x0), ndc_y(y0));
    glVertex2f(ndc_x(x1), ndc_y(y1));
    glEnd();
}

static void draw_circle_px(float cx,float cy,float radius,int segments,float r,float g,float b){
    glColor3f(r,g,b);
    glBegin(GL_TRIANGLE_FAN);
    for(int i=0;i<=segments;i++){
        constexpr float PI_F = 3.14159265358979323846f;
        float a = 2.0f * PI_F * float(i) / float(segments);
        float x = cx + std::cos(a)*radius;
        float y = cy + std::sin(a)*radius;
        glVertex2f(ndc_x(x), ndc_y(y));
    }
    glEnd();
}

Renderer::Renderer() {
    if(!init_glfw()) return;
    impl = std::make_unique<Impl>();
    // We use immediate-mode OpenGL (glBegin/glEnd). Core profile removes these APIs,
    // so we must request a COMPAT profile.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    impl->window = glfwCreateWindow(WIDTH*1.5, HEIGHT*1.5, "Intersection", nullptr, nullptr);
    if(!impl->window){
        glfwTerminate();
        return;
    }

    // Set window icon (ICO only; PNG via GDI+ was unstable)
    impl->hwnd = glfwGetWin32Window(impl->window);
    if(impl->hwnd){
        // Running from repo root (E:\IMPORTANT_\SCI\train), assets are under cpp\assets
        HICON icon_big = static_cast<HICON>(
            LoadImageW(nullptr, L"cpp\\assets\\icon.ico", IMAGE_ICON, 0, 0,
                       LR_LOADFROMFILE | LR_DEFAULTSIZE)
        );
        HICON icon_small = static_cast<HICON>(
            LoadImageW(nullptr, L"cpp\\assets\\icon.ico", IMAGE_ICON, 16, 16,
                       LR_LOADFROMFILE)
        );
        if(icon_big){
            SendMessageW(impl->hwnd, WM_SETICON, ICON_BIG, (LPARAM)icon_big);
            impl->hicon_big = icon_big;
        }
        if(icon_small){
            SendMessageW(impl->hwnd, WM_SETICON, ICON_SMALL, (LPARAM)icon_small);
            impl->hicon_small = icon_small;
        }
    }

    glfwMakeContextCurrent(impl->window);
    glfwSwapInterval(1);
    // No GLAD in this build: using immediate-mode OpenGL functions provided via system GL headers.
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

#ifdef _WIN32
    impl->hwnd = glfwGetWin32Window(impl->window);
    impl->hdc = nullptr;
    impl->font = CreateFontW(
        -18, 0, 0, 0,
        FW_NORMAL,
        FALSE, FALSE, FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE,
        L"Segoe UI");

    // Slightly smaller font for lane IDs (independent from HUD)
    impl->font_lane_ids = CreateFontW(
        -12, 0, 0, 0,
        FW_NORMAL,
        FALSE, FALSE, FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE,
        L"Segoe UI");
#endif

    initialized = true;
}

Renderer::~Renderer(){
#ifdef _WIN32
    if(impl){
        if(impl->hdc){
            ReleaseDC(impl->hwnd, impl->hdc);
            impl->hdc = nullptr;
        }
        if(impl->font){
            DeleteObject(impl->font);
            impl->font = nullptr;
        }
        if(impl->font_lane_ids){
            DeleteObject(impl->font_lane_ids);
            impl->font_lane_ids = nullptr;
        }
        if(impl->hicon_big){
            DestroyIcon(impl->hicon_big);
            impl->hicon_big = nullptr;
        }
        if(impl->hicon_small){
            DestroyIcon(impl->hicon_small);
            impl->hicon_small = nullptr;
        }
    }
#endif
    if(impl && impl->window){
        glfwDestroyWindow(impl->window);
        glfwTerminate();
    }
}

bool Renderer::init_glfw(){
    if(!glfwInit()){
        std::cerr << "Failed to init GLFW" << std::endl;
        return false;
    }
    return true;
}

bool Renderer::window_should_close() const {
    if(!impl || !impl->window) return true;
    return glfwWindowShouldClose(impl->window) != 0;
}

void Renderer::poll_events() const {
    glfwPollEvents();
}

bool Renderer::key_pressed(int glfw_key) const {
    if(!impl || !impl->window) return false;
    return glfwGetKey(impl->window, glfw_key) == GLFW_PRESS;
}

void Renderer::render(const IntersectionEnv& env, bool show_lane_ids, bool show_lidar){
    if(!initialized) return;
    glfwMakeContextCurrent(impl->window);

    // Dynamic viewport (support window resizing) + keep aspect ratio (letterbox/pillarbox)
    int full_w=WIDTH, full_h=HEIGHT;
    glfwGetFramebufferSize(impl->window, &full_w, &full_h);
    const int view = (full_w < full_h) ? full_w : full_h;
    const int vp_x = (full_w - view) / 2;
    const int vp_y = (full_h - view) / 2;
    glViewport(vp_x, vp_y, view, view);

    glClearColor(RenderColors::Background.r, RenderColors::Background.g, RenderColors::Background.b, RenderColors::Background.a);
    glClear(GL_COLOR_BUFFER_BIT);

    draw_road(env.num_lanes);
    draw_route(env);
    draw_cars(env);
    if(show_lidar) draw_lidar(env);

    // Lane IDs + HUD are drawn via Win32 GDI overlay

    glfwSwapBuffers(impl->window);

    if(show_lane_ids){
        gdi_begin_frame(full_w, full_h);
        draw_lane_ids(env);
        gdi_end_frame();
    }

    // Keep GLFW input responsive even if Python doesn't call poll_events()
    glfwPollEvents();
}

static std::wstring to_wide(const std::string& s){
    if(s.empty()) return std::wstring();
    int needed = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if(needed <= 0){
        std::wstring out;
        out.reserve(s.size());
        for(unsigned char ch : s) out.push_back((wchar_t)ch);
        return out;
    }
    std::wstring w;
    w.resize((size_t)needed - 1);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), needed);
    return w;
}

void Renderer::gdi_begin_frame(int fb_w, int fb_h) const{
    if(!impl || !impl->hwnd) return;
    if(!impl->hdc) impl->hdc = GetDC(impl->hwnd);
    impl->fb_w = fb_w;
    impl->fb_h = fb_h;

    if(!impl->hdc) return;
    SetBkMode(impl->hdc, TRANSPARENT);
    if(impl->font) SelectObject(impl->hdc, impl->font);
}

void Renderer::gdi_draw_text_px(int x, int y, const std::string& text, unsigned int rgb) const{
    if(!impl || !impl->hdc) return;

    const COLORREF color = RGB((rgb >> 16) & 0xFFu, (rgb >> 8) & 0xFFu, rgb & 0xFFu);
    SetTextColor(impl->hdc, color);

    auto w = to_wide(text);
    TextOutW(impl->hdc, x, y, w.c_str(), (int)w.size());
}

void Renderer::gdi_end_frame() const{
    if(!impl || !impl->hwnd || !impl->hdc) return;
    ReleaseDC(impl->hwnd, impl->hdc);
    impl->hdc = nullptr;
}

void Renderer::draw_lane_ids(const IntersectionEnv& env) const{
    if(!impl || !impl->hdc) return;

    HGDIOBJ old_font = nullptr;
    if(impl->font_lane_ids){
        old_font = SelectObject(impl->hdc, impl->font_lane_ids);
    }
    const unsigned int in_color = RenderColors::LaneIdInRGB;
    const unsigned int out_color = RenderColors::LaneIdOutRGB;

    // To match the old layout, we need to know the viewport transform
    const int view = (impl->fb_w < impl->fb_h) ? impl->fb_w : impl->fb_h;
    const int vp_x = (impl->fb_w - view) / 2;
    const int vp_y = (impl->fb_h - view) / 2;

    // Stable draw order to avoid label flicker when texts overlap
    std::vector<std::pair<std::string, std::pair<float,float>>> items;
    items.reserve(env.lane_layout.points.size());
    for(const auto& kv : env.lane_layout.points){
        items.push_back(kv);
    }

    auto key = [](const std::string& id){
        int group = 2; // others
        if(id.rfind("IN_", 0) == 0) group = 0;
        else if(id.rfind("OUT_", 0) == 0) group = 1;

        int num = -1;
        size_t us = id.find('_');
        if(us != std::string::npos){
            try { num = std::stoi(id.substr(us + 1)); } catch(...) { num = -1; }
        }
        return std::pair<int,int>(group, num);
    };

    std::sort(items.begin(), items.end(), [&](const auto& a, const auto& b){
        auto ka = key(a.first);
        auto kb = key(b.first);
        if(ka != kb) return ka < kb;
        return a.first < b.first;
    });

    for(const auto& kv : items){
        const std::string& id = kv.first;
        const auto& p = kv.second;
        const bool is_in = id.rfind("IN_", 0) == 0;

        // Convert logical px to framebuffer px
        int fb_px = vp_x + int(p.first * view / WIDTH);
        int fb_py = vp_y + int(p.second * view / HEIGHT);

        // Center text
        SIZE text_size;
        auto wide_id = to_wide(id);
        GetTextExtentPoint32W(impl->hdc, wide_id.c_str(), (int)wide_id.size(), &text_size);
        fb_px -= text_size.cx / 2;
        fb_py -= text_size.cy / 2;

        gdi_draw_text_px(fb_px, fb_py, id, is_in ? in_color : out_color);
    }

    if(old_font){
        SelectObject(impl->hdc, old_font);
    }
}

void Renderer::draw_hud(const IntersectionEnv& env) const{
    int agents_alive = 0;
    for(const auto& c: env.cars) if(c.alive) agents_alive++;

    std::string line = "STEP: " + std::to_string(env.step_count) + " | AGENTS: " + std::to_string(agents_alive);
    if(env.traffic_flow){
        line += " | TRAFFIC: " + std::to_string((int)env.traffic_cars.size());
    }

    std::string lidar_line = "LIDAR: " + std::to_string((int)env.lidars.size());

    if(!env.lidars.empty()){
        const auto& lid = env.lidars[0];
        char buf2[128];
        std::snprintf(buf2, sizeof(buf2), " | RAYS: %d", (int)lid.distances.size());
        lidar_line += buf2;
    }

    if(!env.cars.empty() && env.cars[0].alive){
        float speed_ms = (env.cars[0].state.v * FPS) / SCALE;
        char buf[64];
        std::snprintf(buf, sizeof(buf), " | SPEED: %.1f M/S", speed_ms);
        lidar_line += buf;
    }

    if(impl && impl->hdc){
        gdi_draw_text_px(10, 10, line, RenderColors::HudTextRGB);
        gdi_draw_text_px(10, 34, lidar_line, RenderColors::HudTextRGB);
    }
}


// ROUTE ---------------------------------------------------
void Renderer::draw_route(const IntersectionEnv& env) const{
    if(env.cars.empty()) return;
    const auto& car = env.cars[0];
    if(car.path.empty()) return;

    // Match screenshot style: cyan route
    glLineWidth(2.0f);
    glColor4f(RenderColors::RouteCyan.r, RenderColors::RouteCyan.g, RenderColors::RouteCyan.b, RenderColors::RouteCyan.a);
    glBegin(GL_LINE_STRIP);
    for(const auto& p : car.path){
        glVertex2f(ndc_x(p.first), ndc_y(p.second));
    }
    glEnd();

    // Lookahead target point (match Intersection agent observation)
    // lookahead = 10, target_idx = min(path_index + lookahead, len(path)-1)
    const int lookahead = 10;
    int target_idx = car.path_index + lookahead;
    if(target_idx < 0) target_idx = 0;
    if(target_idx >= (int)car.path.size()) target_idx = (int)car.path.size() - 1;

    const float tx = car.path[target_idx].first;
    const float ty = car.path[target_idx].second;

    // draw a small red dot
    draw_circle_px(tx, ty, 4.0f, 10, RenderColors::TargetRed.r, RenderColors::TargetRed.g, RenderColors::TargetRed.b);
}
// ROAD GEOMETRY -----------------------------------------
static void draw_center_lines(int num_lanes, float rw) {
    const float center_gap = 2.0f;
    const float cx = WIDTH * 0.5f;
    const float cy = HEIGHT * 0.5f;
    const float stop_off = rw + CORNER_RADIUS;

    // vertical yellow lines
    draw_line_px(cx - center_gap, 0, cx - center_gap, cy - stop_off, 2, RenderColors::CenterLineYellow.r, RenderColors::CenterLineYellow.g, RenderColors::CenterLineYellow.b, RenderColors::CenterLineYellow.a);
    draw_line_px(cx + center_gap, 0, cx + center_gap, cy - stop_off, 2, RenderColors::CenterLineYellow.r, RenderColors::CenterLineYellow.g, RenderColors::CenterLineYellow.b, RenderColors::CenterLineYellow.a);
    draw_line_px(cx - center_gap, HEIGHT, cx - center_gap, cy + stop_off, 2, RenderColors::CenterLineYellow.r, RenderColors::CenterLineYellow.g, RenderColors::CenterLineYellow.b, RenderColors::CenterLineYellow.a);
    draw_line_px(cx + center_gap, HEIGHT, cx + center_gap, cy + stop_off, 2, RenderColors::CenterLineYellow.r, RenderColors::CenterLineYellow.g, RenderColors::CenterLineYellow.b, RenderColors::CenterLineYellow.a);

    // horizontal yellow lines
    draw_line_px(0, cy - center_gap, cx - stop_off, cy - center_gap, 2, RenderColors::CenterLineYellow.r, RenderColors::CenterLineYellow.g, RenderColors::CenterLineYellow.b, RenderColors::CenterLineYellow.a);
    draw_line_px(0, cy + center_gap, cx - stop_off, cy + center_gap, 2, RenderColors::CenterLineYellow.r, RenderColors::CenterLineYellow.g, RenderColors::CenterLineYellow.b, RenderColors::CenterLineYellow.a);
    draw_line_px(WIDTH, cy - center_gap, cx + stop_off, cy - center_gap, 2, RenderColors::CenterLineYellow.r, RenderColors::CenterLineYellow.g, RenderColors::CenterLineYellow.b, RenderColors::CenterLineYellow.a);
    draw_line_px(WIDTH, cy + center_gap, cx + stop_off, cy + center_gap, 2, RenderColors::CenterLineYellow.r, RenderColors::CenterLineYellow.g, RenderColors::CenterLineYellow.b, RenderColors::CenterLineYellow.a);
}

static void draw_stop_lines(float rw) {
    const float cx = WIDTH * 0.5f;
    const float cy = HEIGHT * 0.5f;
    const float stop_off = rw + CORNER_RADIUS;
    const float w = 4.0f;

    draw_line_px(cx - rw, cy - stop_off, cx, cy - stop_off, w, RenderColors::MarkingWhite.r, RenderColors::MarkingWhite.g, RenderColors::MarkingWhite.b, RenderColors::MarkingWhite.a);
    draw_line_px(cx, cy + stop_off, cx + rw, cy + stop_off, w, RenderColors::MarkingWhite.r, RenderColors::MarkingWhite.g, RenderColors::MarkingWhite.b, RenderColors::MarkingWhite.a);
    draw_line_px(cx - stop_off, cy, cx - stop_off, cy + rw, w, RenderColors::MarkingWhite.r, RenderColors::MarkingWhite.g, RenderColors::MarkingWhite.b, RenderColors::MarkingWhite.a);
    draw_line_px(cx + stop_off, cy, cx + stop_off, cy - rw, w, RenderColors::MarkingWhite.r, RenderColors::MarkingWhite.g, RenderColors::MarkingWhite.b, RenderColors::MarkingWhite.a);
}

static void draw_road_boundaries(float rw) {
    const float cx = WIDTH * 0.5f;
    const float cy = HEIGHT * 0.5f;
    const float cr = CORNER_RADIUS;
    const float w = 3.0f;

    // Outer edges of the two road strips
    draw_line_px(cx - rw, 0, cx - rw, cy - rw - cr, w, RenderColors::RoadBoundary.r, RenderColors::RoadBoundary.g, RenderColors::RoadBoundary.b, RenderColors::RoadBoundary.a);
    draw_line_px(cx + rw, 0, cx + rw, cy - rw - cr, w, RenderColors::RoadBoundary.r, RenderColors::RoadBoundary.g, RenderColors::RoadBoundary.b, RenderColors::RoadBoundary.a);
    draw_line_px(cx - rw, HEIGHT, cx - rw, cy + rw + cr, w, RenderColors::RoadBoundary.r, RenderColors::RoadBoundary.g, RenderColors::RoadBoundary.b, RenderColors::RoadBoundary.a);
    draw_line_px(cx + rw, HEIGHT, cx + rw, cy + rw + cr, w, RenderColors::RoadBoundary.r, RenderColors::RoadBoundary.g, RenderColors::RoadBoundary.b, RenderColors::RoadBoundary.a);

    draw_line_px(0, cy - rw, cx - rw - cr, cy - rw, w, RenderColors::RoadBoundary.r, RenderColors::RoadBoundary.g, RenderColors::RoadBoundary.b, RenderColors::RoadBoundary.a);
    draw_line_px(0, cy + rw, cx - rw - cr, cy + rw, w, RenderColors::RoadBoundary.r, RenderColors::RoadBoundary.g, RenderColors::RoadBoundary.b, RenderColors::RoadBoundary.a);
    draw_line_px(WIDTH, cy - rw, cx + rw + cr, cy - rw, w, RenderColors::RoadBoundary.r, RenderColors::RoadBoundary.g, RenderColors::RoadBoundary.b, RenderColors::RoadBoundary.a);
    draw_line_px(WIDTH, cy + rw, cx + rw + cr, cy + rw, w, RenderColors::RoadBoundary.r, RenderColors::RoadBoundary.g, RenderColors::RoadBoundary.b, RenderColors::RoadBoundary.a);

    // Rounded corners (quarter arcs)
    auto draw_arc = [&](float ox, float oy, float a0, float a1) {
        const int segments = 48;
        float prev_x = ox + cr * std::cos(a0);
        float prev_y = oy + cr * std::sin(a0);
        for(int i=1;i<=segments;i++){
            float t = float(i)/float(segments);
            float a = a0 + (a1 - a0) * t;
            float x = ox + cr * std::cos(a);
            float y = oy + cr * std::sin(a);
            draw_line_px(prev_x, prev_y, x, y, w, RenderColors::RoadBoundary.r, RenderColors::RoadBoundary.g, RenderColors::RoadBoundary.b, RenderColors::RoadBoundary.a);
            prev_x = x;
            prev_y = y;
        }
    };

    // The 4 corner arc centers are same as grass centers
    draw_arc(cx - rw - cr, cy - rw - cr, 0.0f, 1.57079632679f);
    draw_arc(cx + rw + cr, cy - rw - cr, 1.57079632679f, 3.14159265359f);
    draw_arc(cx - rw - cr, cy + rw + cr, -1.57079632679f, 0.0f);
    draw_arc(cx + rw + cr, cy + rw + cr, 3.14159265359f, 4.71238898038f);
}

static void draw_lane_dashes(int num_lanes, float rw) {
    const float cx = WIDTH * 0.5f;
    const float cy = HEIGHT * 0.5f;
    const float stop_off = rw + CORNER_RADIUS;

    auto dash = [&](float x0, float y0, float x1, float y1) {
        float dist = std::hypot(x1 - x0, y1 - y0);
        const float dash_len = 20.0f;
        int steps = int(dist / (dash_len * 2));
        float dx = (x1 - x0) / dist;
        float dy = (y1 - y0) / dist;

        for (int i = 0; i <= steps; i++) {
            float sx = x0 + dx * i * dash_len * 2;
            float sy = y0 + dy * i * dash_len * 2;
            float ex = sx + dx * dash_len;
            float ey = sy + dy * dash_len;

            // Clamp last segment so lane dashes never cross the stop line end point
            const float t_end = (i == steps) ? 1.0f : float(i * dash_len * 2 + dash_len) / dist;
            if(t_end >= 1.0f){
                ex = x1;
                ey = y1;
            }

            draw_line_px(sx, sy, ex, ey, 2, RenderColors::MarkingWhite.r, RenderColors::MarkingWhite.g, RenderColors::MarkingWhite.b, RenderColors::MarkingWhite.a);
        }
    };

    for (int i = 1; i < num_lanes; i++) {
        float off = i * LANE_WIDTH_PX;
        // vertical dashes
        dash(cx - off, 0, cx - off, cy - stop_off);
        dash(cx + off, 0, cx + off, cy - stop_off);
        dash(cx - off, HEIGHT, cx - off, cy + stop_off);
        dash(cx + off, HEIGHT, cx + off, cy + stop_off);
        // horizontal dashes
        dash(0, cy - off, cx - stop_off, cy - off);
        dash(0, cy + off, cx - stop_off, cy + off);
        dash(WIDTH, cy - off, cx + stop_off, cy - off);
        dash(WIDTH, cy + off, cx + stop_off, cy + off);
    }
}

void Renderer::draw_road(int num_lanes) const{
    float rw = num_lanes * LANE_WIDTH_PX;

    // Base road surface
    draw_rect_ndc(WIDTH * 0.5f - rw, 0, 2 * rw, HEIGHT, RenderColors::RoadSurface.r, RenderColors::RoadSurface.g, RenderColors::RoadSurface.b, RenderColors::RoadSurface.a);
    draw_rect_ndc(0, HEIGHT * 0.5f - rw, WIDTH, 2 * rw, RenderColors::RoadSurface.r, RenderColors::RoadSurface.g, RenderColors::RoadSurface.b, RenderColors::RoadSurface.a);

    // Rounded corner handling (match Intersection/env.py)
    const float cr = CORNER_RADIUS;
    const float cx = WIDTH * 0.5f;
    const float cy = HEIGHT * 0.5f;

    std::array<std::pair<float,float>,4> corner_squares={
        std::make_pair(cx - rw - cr, cy - rw - cr),
        std::make_pair(cx + rw,      cy - rw - cr),
        std::make_pair(cx - rw - cr, cy + rw),
        std::make_pair(cx + rw,      cy + rw)};
    for(auto p: corner_squares) draw_rect_ndc(p.first, p.second, cr, cr, RenderColors::RoadSurface.r, RenderColors::RoadSurface.g, RenderColors::RoadSurface.b, RenderColors::RoadSurface.a);

    std::array<std::pair<float,float>,4> grass_centers={
        std::make_pair(cx - rw - cr, cy - rw - cr),
        std::make_pair(cx + rw + cr, cy - rw - cr),
        std::make_pair(cx - rw - cr, cy + rw + cr),
        std::make_pair(cx + rw + cr, cy + rw + cr)};
    for(auto c: grass_centers) draw_circle_px(c.first, c.second, cr, 32, RenderColors::Grass.r, RenderColors::Grass.g, RenderColors::Grass.b);

    draw_center_lines(num_lanes, rw);
    draw_stop_lines(rw);
    draw_lane_dashes(num_lanes, rw);
    draw_road_boundaries(rw);
}

// CAR DRAW -----------------------------------------------
void Renderer::draw_cars(const IntersectionEnv& env) const{
    auto draw_one=[&](const Car& car, float r,float g,float b, bool npc){
        if(!car.alive) return;
        float x=car.state.x; float y=car.state.y; float heading=car.state.heading;
        float len=CAR_LENGTH; float wid=CAR_WIDTH;

        float hl=len*0.5f; float hw=wid*0.5f;

        auto rot=[&](float lx,float ly){
            float vx = lx * std::cos(-heading) - ly * std::sin(-heading);
            float vy = lx * std::sin(-heading) + ly * std::cos(-heading);
            return std::pair<float,float>(x+vx, y+vy);
        };

        // Body
        glColor3f(r,g,b);
        std::array<std::pair<float,float>,4> body={{
            rot(+hl,+hw), rot(+hl,-hw), rot(-hl,-hw), rot(-hl,+hw)
        }};
        glBegin(GL_QUADS);
        for(const auto& p: body){ glVertex2f(ndc_x(p.first), ndc_y(p.second)); }
        glEnd();

        // Head marker rectangle (matches Intersection/env.py)
        float mr = npc ? RenderColors::TrafficHeadBlack.r : RenderColors::AgentHeadMarker.r;
        float mg = npc ? RenderColors::TrafficHeadBlack.g : RenderColors::AgentHeadMarker.g;
        float mb = npc ? RenderColors::TrafficHeadBlack.b : RenderColors::AgentHeadMarker.b;
        glColor3f(mr,mg,mb);

        float x0 = -hl + 0.70f*len;
        float x1 = -hl + 0.95f*len;
        float y0 = -hw + 2.0f;
        float y1 = +hw - 2.0f;

        std::array<std::pair<float,float>,4> head={{
            rot(x0,y0), rot(x1,y0), rot(x1,y1), rot(x0,y1)
        }};
        glBegin(GL_QUADS);
        for(const auto& p: head){ glVertex2f(ndc_x(p.first), ndc_y(p.second)); }
        glEnd();
    };

    static const std::array<std::array<float,3>,6> colors={{
        {231/255.f,76/255.f,60/255.f},{52/255.f,152/255.f,219/255.f},{46/255.f,204/255.f,113/255.f},
        {155/255.f,89/255.f,182/255.f},{241/255.f,196/255.f,15/255.f},{230/255.f,126/255.f,34/255.f}}};

    // Ego/agents
    for(size_t idx=0; idx<env.cars.size(); ++idx){
        auto col = colors[idx%colors.size()];
        draw_one(env.cars[idx], col[0], col[1], col[2], false);
    }

    // Traffic NPCs: gray body + black head marker
    for(const auto& npc : env.traffic_cars){
        draw_one(npc, RenderColors::TrafficBodyGray.r, RenderColors::TrafficBodyGray.g, RenderColors::TrafficBodyGray.b, true);
    }
}

// LIDAR ---------------------------------------------------
void Renderer::draw_lidar(const IntersectionEnv& env) const{
    const float line_r = RenderColors::LidarRayGreen.r;
    const float line_g = RenderColors::LidarRayGreen.g;
    const float line_b = RenderColors::LidarRayGreen.b;
    const float line_a = RenderColors::LidarRayGreen.a;

    const float hit_r = RenderColors::LidarHitRed.r;
    const float hit_g = RenderColors::LidarHitRed.g;
    const float hit_b = RenderColors::LidarHitRed.b;

    // Match Intersection/sensor.py: draw only hit rays
    const bool draw_all = false;

    for(size_t i=0;i<env.cars.size() && i<env.lidars.size();++i){
        if(!env.cars[i].alive) continue;
        const auto &lid=env.lidars[i];
        const auto &car=env.cars[i];
        float cx=car.state.x; float cy=car.state.y; float heading=car.state.heading;
        for(size_t k=0;k<lid.distances.size();++k){
            float dist=lid.distances[k];
            const bool hit = dist < lid.max_dist - 0.1f;
            if(!draw_all && !hit) continue;

            float ang=heading + lid.rel_angles[k];
            float ex=cx + dist*std::cos(ang);
            float ey=cy - dist*std::sin(ang);

            draw_line_px(cx,cy,ex,ey,2.0f,line_r,line_g,line_b,line_a);
            if(hit){
                draw_circle_px(ex,ey,2.0f,6,hit_r,hit_g,hit_b);
            }
        }
    }
}


#pragma once

#include <vector>
#include <memory>
#include <string>
#include "constants.h"

#ifndef _WIN32
#error "Renderer is Windows-only. Build on Windows (_WIN32) only."
#endif

// Forward declaration
class IntersectionEnv;

class Renderer {
public:
    Renderer();
    ~Renderer();

    // Disable copy
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Render entire scene based on env state
    void render(const IntersectionEnv& env,
                bool show_lane_ids = false,
                bool show_lidar = false);

    // Input / window state (GLFW)
    bool window_should_close() const;
    void poll_events() const;
    bool key_pressed(int glfw_key) const;

    // Whether initialization succeeded
    bool ok() const { return initialized; }
private:
    bool init_glfw();
    void draw_road(int num_lanes) const;
    void draw_cars(const IntersectionEnv& env) const;
    void draw_lidar(const IntersectionEnv& env) const;
    void draw_lane_ids(const IntersectionEnv& env) const;
    void draw_hud(const IntersectionEnv& env) const;
    void draw_route(const IntersectionEnv& env) const;

#ifdef _WIN32
    // Win32 GDI overlay text (better HUD font)
    void gdi_begin_frame(int fb_w, int fb_h) const;
    void gdi_draw_text_px(int x, int y, const std::string& text, unsigned int rgb) const;
    void gdi_end_frame() const;
#endif

    bool initialized{false};
    struct Impl;
    std::unique_ptr<Impl> impl;
};

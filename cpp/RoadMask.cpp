#include "RoadMask.h"
#include <algorithm>
#include <cmath>

RoadMask::RoadMask(int num_lanes) {
    grid.assign(size_t(width) * size_t(height), uint8_t(1));
    generate(num_lanes);
}

void RoadMask::fill_rect(std::vector<uint8_t>& g, int w, int h, int x, int y, int rw, int rh, uint8_t v) {
    int x0 = std::max(0, x);
    int y0 = std::max(0, y);
    int x1 = std::min(w, x + rw);
    int y1 = std::min(h, y + rh);
    for (int yy = y0; yy < y1; ++yy) {
        size_t base = size_t(yy) * size_t(w);
        for (int xx = x0; xx < x1; ++xx) {
            g[base + size_t(xx)] = v;
        }
    }
}

void RoadMask::fill_circle(std::vector<uint8_t>& g, int w, int h, int cx, int cy, int r, uint8_t v) {
    // 离散像素圆：与 pygame.draw.circle 的“填充圆”一致的常见实现（用 <= r^2）
    const int r2 = r * r;
    int y0 = std::max(0, cy - r);
    int y1 = std::min(h - 1, cy + r);
    int x0 = std::max(0, cx - r);
    int x1 = std::min(w - 1, cx + r);
    for (int y = y0; y <= y1; ++y) {
        int dy = y - cy;
        int dy2 = dy * dy;
        size_t base = size_t(y) * size_t(w);
        for (int x = x0; x <= x1; ++x) {
            int dx = x - cx;
            if (dx * dx + dy2 <= r2) {
                g[base + size_t(x)] = v;
            }
        }
    }
}

void RoadMask::generate(int num_lanes) {
    // 完整复刻 Intersection/env.py::Road._generate_collision_mask
    // White=obstacle(1), Black=road(0)
    const int cx = WIDTH / 2;
    const int cy = HEIGHT / 2;
    const int rw = int(std::lround(float(num_lanes) * LANE_WIDTH_PX));
    const int cr = int(std::lround(CORNER_RADIUS));

    // 初始全白(障碍)
    std::fill(grid.begin(), grid.end(), uint8_t(1));

    // Cut out intersection (road)
    fill_rect(grid, width, height, cx - rw, 0, rw * 2, height, 0);
    fill_rect(grid, width, height, 0, cy - rw, width, rw * 2, 0);

    // Fill dead corners (set to road)
    fill_rect(grid, width, height, cx - rw - cr, cy - rw - cr, cr, cr, 0);
    fill_rect(grid, width, height, cx + rw,      cy - rw - cr, cr, cr, 0);
    fill_rect(grid, width, height, cx - rw - cr, cy + rw,      cr, cr, 0);
    fill_rect(grid, width, height, cx + rw,      cy + rw,      cr, cr, 0);

    // NOTE:
    // The collision mask is used by LiDAR; it must closely match the rendered road.
    // The previous implementation re-applied obstacle circles at the outer corners,
    // which can erroneously mark road pixels near the intersection as obstacles and
    // cause LiDAR rays to immediately hit at step_size.
    // Keep the rounded-corner visuals in the renderer, but keep the collision mask
    // conservative (treat the road cutouts as driveable).
}


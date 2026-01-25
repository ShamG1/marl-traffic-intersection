#include "RouteGen.h"
#include "constants.h"
#include <cmath>

static constexpr float PI_F = 3.14159265358979323846f;

LaneLayout build_lane_layout_cpp(int num_lanes) {
    LaneLayout layout;
    const float CX = WIDTH * 0.5f;
    const float CY = HEIGHT * 0.5f;
    const float MARGIN = 30.0f;

    const char dir_order_arr[4] = {'N','E','S','W'};
    layout.dir_order = {"N", "E", "S", "W"};
    layout.in_by_dir = {{"N", {}}, {"E", {}}, {"S", {}}, {"W", {}}};
    layout.out_by_dir = {{"N", {}}, {"E", {}}, {"S", {}}, {"W", {}}};

    for (int d_idx = 0; d_idx < 4; ++d_idx) {
        char d = dir_order_arr[d_idx];
        for (int j = 0; j < num_lanes; ++j) {
            float offset = LANE_WIDTH_PX * (0.5f + float(j));
            std::string in_name = "IN_" + std::to_string(d_idx * num_lanes + j + 1);
            std::string out_name = "OUT_" + std::to_string(d_idx * num_lanes + j + 1);

            float in_x=0, in_y=0, out_x=0, out_y=0;
            if (d=='N') {
                in_x = CX - offset; in_y = MARGIN;
                out_x = CX + offset; out_y = MARGIN;
            } else if (d=='S') {
                in_x = CX + offset; in_y = HEIGHT - MARGIN;
                out_x = CX - offset; out_y = HEIGHT - MARGIN;
            } else if (d=='E') {
                in_x = WIDTH - MARGIN; in_y = CY - offset;
                out_x = WIDTH - MARGIN; out_y = CY + offset;
            } else { // W
                in_x = MARGIN; in_y = CY + offset;
                out_x = MARGIN; out_y = CY - offset;
            }

            layout.points[in_name] = {in_x, in_y};
            layout.points[out_name] = {out_x, out_y};
            layout.dir_of[in_name] = std::string(1, d);
            layout.dir_of[out_name] = std::string(1, d);
            layout.idx_of[in_name] = j;
            layout.idx_of[out_name] = j;

            layout.in_by_dir[std::string(1, d)].push_back(in_name);
            layout.out_by_dir[std::string(1, d)].push_back(out_name);
        }
    }

    return layout;
}

int determine_intent(const LaneLayout& layout, const std::string& start_id, const std::string& end_id) {
    auto it_s = layout.dir_of.find(start_id);
    auto it_e = layout.dir_of.find(end_id);
    if (it_s == layout.dir_of.end() || it_e == layout.dir_of.end()) return INTENT_LEFT;
    const std::string& s_str = it_s->second;
    const std::string& e_str = it_e->second;
    const char s = s_str.empty() ? 'N' : s_str[0];
    const char e = e_str.empty() ? 'S' : e_str[0];

    auto opposite = [&](char d){
        if (d=='N') return 'S';
        if (d=='S') return 'N';
        if (d=='E') return 'W';
        return 'E';
    };
    auto left_turn = [&](char d){
        if (d=='N') return 'E';
        if (d=='E') return 'S';
        if (d=='S') return 'W';
        return 'N';
    };
    auto right_turn = [&](char d){
        if (d=='N') return 'W';
        if (d=='W') return 'S';
        if (d=='S') return 'E';
        return 'N';
    };

    if (e == opposite(s)) return INTENT_STRAIGHT;
    if (e == left_turn(s)) return INTENT_LEFT;
    if (e == right_turn(s)) return INTENT_RIGHT;
    return INTENT_LEFT;
}

static inline std::pair<float,float> project_to_box(const std::pair<float,float>& pt, int num_lanes) {
    const float CX = WIDTH * 0.5f;
    const float CY = HEIGHT * 0.5f;
    float turn_bound = num_lanes * LANE_WIDTH_PX;
    float bx_l = CX - turn_bound, bx_r = CX + turn_bound;
    float by_t = CY - turn_bound, by_b = CY + turn_bound;

    float x = pt.first, y = pt.second;
    if (y < by_t) return {x, by_t};
    if (y > by_b) return {x, by_b};
    if (x < bx_l) return {bx_l, y};
    return {bx_r, y};
}

static inline std::pair<float,float> bezier_point(float t, const std::pair<float,float>& p0,
                                                  const std::pair<float,float>& p1,
                                                  const std::pair<float,float>& p2) {
    float x = (1-t)*(1-t)*p0.first + 2*(1-t)*t*p1.first + t*t*p2.first;
    float y = (1-t)*(1-t)*p0.second + 2*(1-t)*t*p1.second + t*t*p2.second;
    return {x,y};
}

std::vector<std::pair<float,float>> generate_path_cpp(const LaneLayout& layout,
                                                      int num_lanes,
                                                      int intent,
                                                      const std::string& start_id,
                                                      const std::string& end_id) {
    const float CX = WIDTH * 0.5f;
    const float CY = HEIGHT * 0.5f;

    auto p_start = layout.points.at(start_id);
    auto p_end = layout.points.at(end_id);
    auto entry_p = project_to_box(p_start, num_lanes);
    auto exit_p = project_to_box(p_end, num_lanes);

    std::vector<std::pair<float,float>> path;
    path.reserve(200);

    if (intent == INTENT_STRAIGHT || intent == INTENT_LEFT) {
        for (int i=0;i<50;++i){
            float t=float(i)/50.0f;
            path.push_back({p_start.first + (entry_p.first-p_start.first)*t,
                            p_start.second + (entry_p.second-p_start.second)*t});
        }

        if (intent == INTENT_STRAIGHT) {
            for (int i=0;i<60;++i){
                float t=float(i)/60.0f;
                path.push_back({entry_p.first + (exit_p.first-entry_p.first)*t,
                                entry_p.second + (exit_p.second-entry_p.second)*t});
            }
        } else {
            std::pair<float,float> ctrl{CX, CY};
            for (int i=0;i<60;++i){
                float t=float(i)/60.0f;
                path.push_back(bezier_point(t, entry_p, ctrl, exit_p));
            }
        }

        for (int i=0;i<50;++i){
            float t=float(i)/50.0f;
            path.push_back({exit_p.first + (p_end.first-exit_p.first)*t,
                            exit_p.second + (p_end.second-exit_p.second)*t});
        }
        return path;
    }

    // Right turn arc
    char start_dir = 'N';
    auto it = layout.dir_of.find(start_id);
    if (it != layout.dir_of.end() && !it->second.empty()) start_dir = it->second[0];

    float road_half_width = num_lanes * LANE_WIDTH_PX;

    float cx_c=0, cy_c=0, theta_start=0, theta_end=0;
    if (start_dir == 'N') {
        cx_c = CX - road_half_width - CORNER_RADIUS;
        cy_c = CY - road_half_width - CORNER_RADIUS;
        theta_start = 0.0f; theta_end = PI_F/2.0f;
    } else if (start_dir == 'E') {
        cx_c = CX + road_half_width + CORNER_RADIUS;
        cy_c = CY - road_half_width - CORNER_RADIUS;
        theta_start = PI_F/2.0f; theta_end = PI_F;
    } else if (start_dir == 'S') {
        cx_c = CX + road_half_width + CORNER_RADIUS;
        cy_c = CY + road_half_width + CORNER_RADIUS;
        theta_start = PI_F; theta_end = 3.0f*PI_F/2.0f;
    } else {
        cx_c = CX - road_half_width - CORNER_RADIUS;
        cy_c = CY + road_half_width + CORNER_RADIUS;
        theta_start = -PI_F/2.0f; theta_end = 0.0f;
    }

    float r = CORNER_RADIUS + 0.5f * LANE_WIDTH_PX;
    std::pair<float,float> arc_start{cx_c + r*std::cos(theta_start), cy_c + r*std::sin(theta_start)};
    std::pair<float,float> arc_end{cx_c + r*std::cos(theta_end), cy_c + r*std::sin(theta_end)};

    for (int i=0;i<50;++i){
        float t=float(i)/50.0f;
        path.push_back({p_start.first + (arc_start.first-p_start.first)*t,
                        p_start.second + (arc_start.second-p_start.second)*t});
    }

    for (int i=0;i<60;++i){
        float t=float(i)/60.0f;
        float theta = theta_start + (theta_end-theta_start)*t;
        path.push_back({cx_c + r*std::cos(theta), cy_c + r*std::sin(theta)});
    }

    for (int i=0;i<50;++i){
        float t=float(i)/50.0f;
        path.push_back({arc_end.first + (p_end.first-arc_end.first)*t,
                        arc_end.second + (p_end.second-arc_end.second)*t});
    }

    return path;
}

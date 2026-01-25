#include "IntersectionEnv.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

static constexpr float PI_F_TF = 3.14159265358979323846f;
static inline float wrap_angle_rad_tf(float a) {
    a = std::fmod(a + PI_F_TF, 2.0f * PI_F_TF);
    if (a < 0) a += 2.0f * PI_F_TF;
    return a - PI_F_TF;
}

// Helper: clamp index
static inline size_t clamp_idx(size_t i, size_t n) { return (n == 0) ? 0 : (i < n ? i : (n - 1)); }

// --- C++ port (incremental) of Intersection/agent.py::Car.plan_autonomous_action ---
// 当前版本实现：
// - 横向：路径 lookahead 的 heading_error * 3.0
// - 纵向：巡航到 target_speed=PHYSICS_MAX_SPEED*0.6，并做前车跟车制动
// - 鬼影路径扫描：检测路口冲突并分级制动（简化版）
static inline float get_front_car_dist_tf(const Car& self, const std::vector<const Car*>& all_vehicles) {
    float min_d = 1e9f;

    const float vx = std::cos(self.state.heading);
    const float vy = -std::sin(self.state.heading);

    for (const auto* other : all_vehicles) {
        if (other == &self) continue;
        if (!other->alive) continue;

        const float dx = other->state.x - self.state.x;
        const float dy = other->state.y - self.state.y;
        const float dist = std::hypot(dx, dy);
        if (dist > 80.0f) continue;

        const float dot = (dx * vx + dy * vy) / (dist + 1e-5f);
        if (dot > 0.8f) {
            const float angle_diff = std::fabs(wrap_angle_rad_tf(self.state.heading - other->state.heading));
            if (angle_diff < (45.0f * PI_F_TF / 180.0f)) {
                if (dist < min_d) min_d = dist;
            }
        }
    }

    return min_d;
}

static inline std::pair<float, float> plan_npc_action_tf(const Car& npc, const std::vector<const Car*>& all_vehicles) {
    // --- 1) 横向控制（路径追踪） ---
    float steer_cmd = 0.0f;
    if (!npc.path.empty()) {
        const int lookahead = 12;
        const int idx = npc.path_index;
        const int target_idx = std::min(idx + lookahead, int(npc.path.size()) - 1);
        const float tx = npc.path[target_idx].first;
        const float ty = npc.path[target_idx].second;

        const float dx = tx - npc.state.x;
        const float dy = ty - npc.state.y;
        const float angle_to_target = std::atan2(-dy, dx);
        const float heading_err = wrap_angle_rad_tf(angle_to_target - npc.state.heading);
        steer_cmd = std::max(-1.0f, std::min(1.0f, heading_err * 3.0f));
    }

    // --- 2) 纵向控制（巡航 + 跟车） ---
    const float target_speed = PHYSICS_MAX_SPEED * 0.4f;

    float acc_throttle = 0.0f;
    if (npc.state.v < target_speed) acc_throttle = 0.5f;
    else if (npc.state.v > target_speed + 1.0f) acc_throttle = -0.1f;

    const float front_dist = get_front_car_dist_tf(npc, all_vehicles);
    if (front_dist < 30.0f) acc_throttle = -1.0f;
    else if (front_dist < 50.0f) acc_throttle = std::min(acc_throttle, -0.2f);

    // --- 3) 鬼影路径扫描 (Ghost Path Scanning) ---
    bool conflict_detected = false;
    float min_conflict_dist = 1e9f;

    const int SCAN_STEPS = 120;
    const int SCAN_STEP_SIZE = 1;
    const float SAFE_RADIUS = CAR_WIDTH * 2.0f;
    const float SAFE_RADIUS_SQ = SAFE_RADIUS * SAFE_RADIUS;

    const float my_dist_to_center = std::hypot(npc.state.x - WIDTH * 0.5f, npc.state.y - HEIGHT * 0.5f);

    const int start_idx = npc.path_index;
    const int end_idx = std::min(start_idx + SCAN_STEPS, (int)npc.path.size());

    for (int i = start_idx; i < end_idx; i += SCAN_STEP_SIZE) {
        const float ghost_x = npc.path[i].first;
        const float ghost_y = npc.path[i].second;

        for (const auto* other : all_vehicles) {
            if (other == &npc) continue;
            if (!other->alive) continue;

            const float dx_og = other->state.x - ghost_x;
            const float dy_og = other->state.y - ghost_y;
            if (dx_og * dx_og + dy_og * dy_og < SAFE_RADIUS_SQ) {
                // 排除同向行驶的前车 (交给 ACC 处理)
                const float angle_diff = std::fabs(wrap_angle_rad_tf(npc.state.heading - other->state.heading));
                if (angle_diff < (60.0f * PI_F_TF / 180.0f)) continue;

                // === 排除并排行驶的车辆 (对齐 Intersection/agent.py 的方法4) ===
                {
                    const float dx_to_other = other->state.x - npc.state.x;
                    const float dy_to_other = other->state.y - npc.state.y;
                    const float dist_to_other = std::hypot(dx_to_other, dy_to_other);

                    if (dist_to_other > 1e-5f) {
                        const float my_dir_x = std::cos(npc.state.heading);
                        const float my_dir_y = -std::sin(npc.state.heading);

                        const float angle_diff_norm = std::min(angle_diff, 2.0f * PI_F_TF - angle_diff);
                        const bool is_parallel_angle = (angle_diff_norm < (30.0f * PI_F_TF / 180.0f)) ||
                                                       (angle_diff_norm > (150.0f * PI_F_TF / 180.0f));

                        if (is_parallel_angle) {
                            const float longitudinal_dist = dx_to_other * my_dir_x + dy_to_other * my_dir_y;
                            const float lateral_sq = std::max(0.0f, dist_to_other * dist_to_other - longitudinal_dist * longitudinal_dist);
                            const float lateral_dist = std::sqrt(lateral_sq);

                            const bool is_sideways_lateral = std::fabs(lateral_dist) < (LANE_WIDTH_PX * 1.5f);
                            const bool is_not_far_ahead_behind = std::fabs(longitudinal_dist) < (CAR_LENGTH * 2.0f);

                            if (is_sideways_lateral && is_not_far_ahead_behind) {
                                // 路径稳定性检查
                                const float future_dist = 20.0f;
                                const float my_future_x = npc.state.x + my_dir_x * future_dist;
                                const float my_future_y = npc.state.y + my_dir_y * future_dist;

                                const float other_dir_x = std::cos(other->state.heading);
                                const float other_dir_y = -std::sin(other->state.heading);
                                const float other_future_x = other->state.x + other_dir_x * future_dist;
                                const float other_future_y = other->state.y + other_dir_y * future_dist;

                                const float future_dx = other_future_x - my_future_x;
                                const float future_dy = other_future_y - my_future_y;
                                const float future_dist_mag = std::hypot(future_dx, future_dy);

                                if (future_dist_mag > 1e-5f) {
                                    const float future_longitudinal = future_dx * my_dir_x + future_dy * my_dir_y;
                                    const float future_lateral_sq = std::max(0.0f, future_dist_mag * future_dist_mag - future_longitudinal * future_longitudinal);
                                    const float future_lateral = std::sqrt(future_lateral_sq);

                                    const float lateral_change = std::fabs(future_lateral - lateral_dist);
                                    const bool is_stable_lateral = lateral_change < (LANE_WIDTH_PX * 0.5f);

                                    if (is_stable_lateral) {
                                        // 认为是并排行驶，跳过
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }

                bool should_yield = false;
                const float other_dist_to_center = std::hypot(other->state.x - WIDTH * 0.5f, other->state.y - HEIGHT * 0.5f);
                const float dist_to_crash = std::hypot(ghost_x - npc.state.x, ghost_y - npc.state.y);

                if (dist_to_crash < 15.0f) {
                    should_yield = true;
                } else if (npc.state.v < 1.0f && other->state.v > 3.0f && other_dist_to_center < my_dist_to_center + 25.0f) {
                    should_yield = true;
                } else if (other_dist_to_center < my_dist_to_center - 5.0f) {
                    should_yield = true;
                } else if (std::fabs(other_dist_to_center - my_dist_to_center) <= 5.0f) {
                    // 规则4：ID 仲裁（用地址近似）
                    if ((uintptr_t)&npc < (uintptr_t)other) {
                        should_yield = true;
                    }
                }

                if (should_yield) {
                    conflict_detected = true;
                    if (dist_to_crash < min_conflict_dist) min_conflict_dist = dist_to_crash;
                }
            }
        }
        if (conflict_detected) break;
    }

    // --- 4) 合成动作 ---
    float final_throttle = acc_throttle;
    if (conflict_detected) {
        if (min_conflict_dist < 35.0f) final_throttle = -1.0f;
        else if (min_conflict_dist < 60.0f) final_throttle = -0.8f;
        else final_throttle = std::min(final_throttle, 0.0f);
    }

    return {final_throttle, steer_cmd};
}

void IntersectionEnv::init_traffic_routes() {
    traffic_routes.clear();

    // Mirror Intersection/env.py::_init_traffic_routes fallback (straight + left using lane indices)
    // dir_order in python: ['N','E','S','W']
    const auto &dir_order = lane_layout.dir_order;

    const std::unordered_map<std::string, std::string> opposite = {
        {"N", "S"}, {"S", "N"}, {"E", "W"}, {"W", "E"}
    };
    const std::unordered_map<std::string, std::string> left_turn = {
        {"N", "E"}, {"E", "S"}, {"S", "W"}, {"W", "N"}
    };

    for (const auto &direction : dir_order) {
        auto it_in = lane_layout.in_by_dir.find(direction);
        if (it_in == lane_layout.in_by_dir.end()) continue;

        const auto &in_lanes = it_in->second;

        auto it_straight_out = lane_layout.out_by_dir.find(opposite.at(direction));
        auto it_left_out = lane_layout.out_by_dir.find(left_turn.at(direction));
        const auto &straight_out_lanes = (it_straight_out != lane_layout.out_by_dir.end()) ? it_straight_out->second : std::vector<std::string>{};
        const auto &left_out_lanes = (it_left_out != lane_layout.out_by_dir.end()) ? it_left_out->second : std::vector<std::string>{};

        for (const auto &start_id : in_lanes) {
            size_t idx = 0;
            auto it_idx = lane_layout.idx_of.find(start_id);
            if (it_idx != lane_layout.idx_of.end()) idx = size_t(std::max(0, it_idx->second));

            if (!straight_out_lanes.empty()) {
                const auto &out_id = straight_out_lanes[clamp_idx(idx, straight_out_lanes.size())];
                traffic_routes.emplace_back(start_id, out_id);
            }
            if (!left_out_lanes.empty()) {
                const auto &out_id = left_out_lanes[clamp_idx(idx, left_out_lanes.size())];
                traffic_routes.emplace_back(start_id, out_id);
            }
        }
    }
}

bool IntersectionEnv::is_spawn_blocked(float sx, float sy) const {
    const float min_dist = CAR_LENGTH * 2.5f;
    const float min_d2 = min_dist * min_dist;

    // Check ego cars
    for (const auto &c : cars) {
        float dx = c.state.x - sx;
        float dy = c.state.y - sy;
        if (dx * dx + dy * dy < min_d2) return true;
    }

    // Check NPC cars
    for (const auto &c : traffic_cars) {
        float dx = c.state.x - sx;
        float dy = c.state.y - sy;
        if (dx * dx + dy * dy < min_d2) return true;
    }

    return false;
}

bool IntersectionEnv::is_arrived(const Car &car, float tol) const {
    if (car.path.empty()) return false;
    const auto goal = car.path.back();
    float d = std::hypot(car.state.x - goal.first, car.state.y - goal.second);
    return d < tol;
}

bool IntersectionEnv::is_out_of_screen(const Car &car, float margin) const {
    const float x = car.state.x;
    const float y = car.state.y;
    if (x < -margin || x > float(WIDTH) + margin || y < -margin || y > float(HEIGHT) + margin) return true;
    return false;
}

void IntersectionEnv::try_spawn_traffic_car() {
    if (traffic_routes.empty()) return;

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist(0, traffic_routes.size() - 1);

    const auto &route = traffic_routes[dist(rng)];
    const auto it = lane_layout.points.find(route.first);
    if (it == lane_layout.points.end()) return;

    const float sx = it->second.first;
    const float sy = it->second.second;
    if (is_spawn_blocked(sx, sy)) return;

    const int intent = determine_intent(lane_layout, route.first, route.second);
    auto path = generate_path_cpp(lane_layout, num_lanes, intent, route.first, route.second);
    if (path.size() < 2) return;

    float heading = 0.0f;
    {
        float dx = path[1].first - path[0].first;
        float dy = path[1].second - path[0].second;
        heading = std::atan2(-dy, dx);
    }

    Car npc;
    npc.state.x = sx;
    npc.state.y = sy;
    npc.state.v = 0.0f;
    npc.state.heading = heading;
    npc.spawn_state = npc.state;
    npc.alive = true;
    npc.intention = intent;
    npc.path = std::move(path);
    npc.path_index = 0;
    npc.prev_dist_to_goal = 0.0f;
    npc.prev_action = {0.0f, 0.0f};

    traffic_cars.push_back(std::move(npc));
    traffic_lidars.emplace_back();
}

void IntersectionEnv::update_traffic_flow(float dt) {
    if (!traffic_flow) return;

    // Spawn probability: 1 - exp(-arrival_rate * dt)
    const float arrival_rate = traffic_density;
    const float spawn_prob = 1.0f - std::exp(-arrival_rate * dt);

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    if (uni(rng) < spawn_prob) {
        try_spawn_traffic_car();
    }

    // --- NPC Controller Update ---
    // NPC planning ignores ego vehicles: only consider NPC traffic cars
    std::vector<const Car*> all_vehicles;
    all_vehicles.reserve(traffic_cars.size());
    for (const auto& c : traffic_cars) all_vehicles.push_back(&c);

    for (auto& npc : traffic_cars) {
        if (!npc.alive) continue;

        npc.update_path_index();
        const auto action = plan_npc_action_tf(npc, all_vehicles);
        npc.update(action.first, action.second, dt);
        npc.update_path_index();
    }

    // --- NPC-NPC collision: remove both (match Intersection/env.py behavior) ---
    for (size_t i = 0; i < traffic_cars.size(); ++i) {
        if (!traffic_cars[i].alive) continue;
        for (size_t j = i + 1; j < traffic_cars.size(); ++j) {
            if (!traffic_cars[j].alive) continue;
            if (traffic_cars[i].check_collision(traffic_cars[j])) {
                traffic_cars[i].alive = false;
                traffic_cars[j].alive = false;
            }
        }
    }

    // Remove arrived / out-of-screen / collided
    for (size_t i = 0; i < traffic_cars.size();) {
        if (!traffic_cars[i].alive || is_arrived(traffic_cars[i], 20.0f) || is_out_of_screen(traffic_cars[i], 100.0f)) {
            traffic_cars.erase(traffic_cars.begin() + long(i));
            traffic_lidars.erase(traffic_lidars.begin() + long(i));
            continue;
        }
        ++i;
    }
}

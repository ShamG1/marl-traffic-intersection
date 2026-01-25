#pragma once
#include <string>
#include <memory>
#include <utility>
#include <vector>

#include "Car.h"
#include "Lidar.h"
#include "RoadGeometry.h"
#include "RoadMask.h"
#include "LineMask.h"
#include "RouteGen.h"
#include "constants.h"
#include "Reward.h"
#include "EnvState.h"
#include "Renderer.h"

// Match Intersection/config.py constants
constexpr int NEIGHBOR_COUNT = 5;

class Renderer;

class IntersectionEnv {
public:
    ~IntersectionEnv();
    // Config
    int num_lanes;
    bool use_team_reward{false};
    bool respawn_enabled{true};
    int max_steps{2000};
    RewardConfig reward_config;

    // Traffic flow (NPC) - mode1: single ego + NPCs
    bool traffic_flow{false};
    float traffic_density{0.5f};

    // State
    LaneLayout lane_layout;
    RoadGeometry geom;
    RoadMask road_mask;
    LineMask line_mask;

    // Ego agents
    std::vector<Car> cars;
    std::vector<Lidar> lidars;
    std::vector<long long> agent_ids;

    // NPC traffic
    std::vector<Car> traffic_cars;
    std::vector<Lidar> traffic_lidars;
    std::vector<std::pair<std::string, std::string>> traffic_routes;

    long long next_agent_id{1};
    int step_count{0};

    explicit IntersectionEnv(int num_lanes_ = 3)
        : num_lanes(num_lanes_),
          lane_layout(build_lane_layout_cpp(num_lanes_)),
          geom(num_lanes_),
          road_mask(num_lanes_),
          line_mask(num_lanes_) {
        init_traffic_routes();
    }

    void configure(bool use_team, bool respawn, int max_s);

    // Enable/disable traffic flow and set density (arrival rate)
    void configure_traffic(bool enabled, float density);

    // Configure routes for NPCs from Python
    void configure_routes(const std::vector<std::pair<std::string, std::string>>& routes);

    void reset();

    void add_car_with_route(const std::string& start_id, const std::string& end_id);

    StepResult step(const std::vector<float>& throttles,
                    const std::vector<float>& steerings,
                    float dt = DT_DEFAULT);

    std::vector<std::vector<float>> get_observations() const;

    // Snapshot API for fast MCTS rollbacks
    EnvState get_state() const;
    void set_state(const EnvState& s);

    void render(bool show_lane_ids = false, bool show_lidar = false);

    // GLFW input/window helpers (available only after first render() creates a window)
    bool window_should_close() const;
    void poll_events() const;
    bool key_pressed(int glfw_key) const;

private:
    // Rendering
    bool render_enabled{false};
    std::unique_ptr<Renderer> renderer; // allocated only when render_enabled

    void init_traffic_routes();
    void update_traffic_flow(float dt);
    void try_spawn_traffic_car();
    bool is_spawn_blocked(float sx, float sy) const;
    bool is_arrived(const Car& car, float tol = 20.0f) const;
    bool is_out_of_screen(const Car& car, float margin = 100.0f) const;
};

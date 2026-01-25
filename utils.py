# This file is a local copy of necessary components from the 'Intersection' package
# to make C_MCTS a self-contained module.
# === From Intersection/config.py ===
WIDTH, HEIGHT = 900, 900
SCALE = 12
LANE_WIDTH_M = 3.5
LANE_WIDTH_PX = int(LANE_WIDTH_M * SCALE)



OBS_DIM = 127

# Default reward config (can be overridden via config dict)
DEFAULT_REWARD_CONFIG = {
    'use_team_reward': False,  # Use team reward mixing (for multi-agent)
    'traffic_flow': False,      # If True, forces individual reward (single-agent with traffic)
    'reward_config': {
        'progress_scale': 10.0, 
        'stuck_speed_threshold': 1.0,  # m/s
        'stuck_penalty': -0.01,
        'crash_vehicle_penalty': -10.0,
        'crash_object_penalty': -5.0,  # Includes CRASH_WALL (off-road) and CRASH_LINE
        'success_reward': 10.0,
        'action_smoothness_scale': -0.02,
        'team_alpha': 0.2,
    }
}
# === From Intersection/env.py ===
DEFAULT_ROUTE_MAPPING_2LANES = {
    "IN_1": ["OUT_3"],
    "IN_2": ["OUT_6"],
    "IN_3": ["OUT_5"],
    "IN_4": ["OUT_8"],
    "IN_6": ["OUT_2"],
    "IN_7": ["OUT_1"],
    "IN_8": ["OUT_4"],
}

DEFAULT_ROUTE_MAPPING_3LANES = {
    "IN_1": ["OUT_4"],
    "IN_2": ["OUT_8"],
    "IN_3": ["OUT_12"],
    "IN_4": ["OUT_7"],
    "IN_5": ["OUT_11"],
    "IN_6": ["OUT_3"],
    "IN_7": ["OUT_10"],
    "IN_8": ["OUT_2"],
    "IN_9": ["OUT_6"],
    "IN_10": ["OUT_1"],
    "IN_11": ["OUT_5"],
    "IN_12": ["OUT_9"],
}

# === From Intersection/agent.py ===
def build_lane_layout(num_lanes: int):
    dir_order = ['N', 'E', 'S', 'W']
    points = {}
    in_by_dir = {d: [] for d in dir_order}
    out_by_dir = {d: [] for d in dir_order}
    dir_of = {}
    idx_of = {}
    MARGIN = 30
    CX, CY = WIDTH // 2, HEIGHT // 2

    for d_idx, d in enumerate(dir_order):
        for j in range(num_lanes):
            offset = LANE_WIDTH_PX * (0.5 + j)
            in_name = f"IN_{d_idx * num_lanes + j + 1}"
            out_name = f"OUT_{d_idx * num_lanes + j + 1}"

            if d == 'N':
                points[in_name] = (CX - offset, MARGIN)
                points[out_name] = (CX + offset, MARGIN)
            elif d == 'S':
                points[in_name] = (CX + offset, HEIGHT - MARGIN)
                points[out_name] = (CX - offset, HEIGHT - MARGIN)
            elif d == 'E':
                points[in_name] = (WIDTH - MARGIN, CY - offset)
                points[out_name] = (WIDTH - MARGIN, CY + offset)
            else:  # 'W'
                points[in_name] = (MARGIN, CY + offset)
                points[out_name] = (MARGIN, CY - offset)

            in_by_dir[d].append(in_name)
            out_by_dir[d].append(out_name)
            dir_of[in_name] = d
            dir_of[out_name] = d
            idx_of[in_name] = j
            idx_of[out_name] = j

    return {
        'points': points,
        'in_by_dir': in_by_dir,
        'out_by_dir': out_by_dir,
        'dir_of': dir_of,
        'idx_of': idx_of,
        'dir_order': dir_order,
    }

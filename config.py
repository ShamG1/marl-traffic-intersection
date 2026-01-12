# --- config.py ---

import math

# === Screen and display settings ===
WIDTH, HEIGHT = 900, 900
FPS = 60
TITLE = "MARL Intersection"

# === Physical dimensions and scale ===
SCALE = 12                  # 1 meter = 12 pixels
LANE_WIDTH_M = 3.5          # Lane width 3.5 meters
LANE_WIDTH_PX = int(LANE_WIDTH_M * SCALE)  # ~42px
NUM_LANES = 2               # Number of lanes per direction
ROAD_HALF_WIDTH = NUM_LANES * LANE_WIDTH_PX
CORNER_RADIUS = int(7 * SCALE) # Corner radius
TURN_BOUND = ROAD_HALF_WIDTH    # Turn trigger boundary

# === Vehicle appearance ===
CAR_LENGTH = int(4.5 * SCALE)
CAR_WIDTH = int(2.0 * SCALE)

# === Physics control parameters (Kinematic Bicycle Model) ===
DT = 1.0 / 60.0             # Simulation timestep
MAX_ACC = 15.0              # Maximum acceleration (px/s^2)
MAX_STEERING_ANGLE = math.radians(35) # Maximum steering angle (35 degrees)
FRICTION = 2.0              # Simulated friction
WHEELBASE = CAR_LENGTH      # Wheelbase

# Physical max speed (pixels/frame) -> for normalization
# 8 px/frame * 60 fps / 12 scale = 40 m/s = 144 km/h
PHYSICS_MAX_SPEED = 8.0     

# === Intentions and actions ===
ACTION_STRAIGHT = 0
ACTION_LEFT = 1 

# === RL observation space ===
NEIGHBOR_COUNT = 8          # K
LIDAR_RAYS = 72             # L
LIDAR_RANGE = 250           # Lidar range
LIDAR_FOV = 360             # Lidar field of view
OBS_DIM = 6 + (5 * NEIGHBOR_COUNT) + LIDAR_RAYS # 118 dimensions

# === Colors ===
COLOR_GRASS = (34, 139, 34)
COLOR_ROAD = (60, 60, 60)
COLOR_YELLOW = (255, 204, 0)
COLOR_WHITE = (240, 240, 240)
COLOR_LIDAR_LINE = (0, 255, 0, 40)
COLOR_LIDAR_HIT = (255, 0, 0)
COLOR_CAR_LIST = [
    (231, 76, 60), (52, 152, 219), (46, 204, 113), 
    (155, 89, 182), (241, 196, 15), (230, 126, 34)
]

# === Reward configuration ===
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
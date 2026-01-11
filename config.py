# --- config.py ---

import math

# === 屏幕与显示设置 ===
WIDTH, HEIGHT = 900, 900
FPS = 60
TITLE = "MARL Intersection"

# === 物理尺寸与比例尺 ===
SCALE = 12                  # 1米 = 12像素
LANE_WIDTH_M = 3.5          # 车道宽 3.5米
LANE_WIDTH_PX = int(LANE_WIDTH_M * SCALE)  # ~42px
NUM_LANES = 2               # 单向车道数
ROAD_HALF_WIDTH = NUM_LANES * LANE_WIDTH_PX
CORNER_RADIUS = int(7 * SCALE) # 拐角半径
TURN_BOUND = ROAD_HALF_WIDTH    # 转弯触发边界

# === 车辆外观 ===
CAR_LENGTH = int(4.5 * SCALE)
CAR_WIDTH = int(2.0 * SCALE)

# === 物理控制参数 (Kinematic Bicycle Model) ===
DT = 1.0 / 60.0             # 仿真步长
MAX_ACC = 15.0              # 最大加速度 (px/s^2)
MAX_STEERING_ANGLE = math.radians(35) # 最大转角 (35度)
FRICTION = 2.0              # 模拟阻力
WHEELBASE = CAR_LENGTH      # 轴距

# 物理极限速度 (像素/帧) -> 用于归一化
# 8 px/frame * 60 fps / 12 scale = 40 m/s = 144 km/h
PHYSICS_MAX_SPEED = 8.0     

# === 意图与动作 ===
ACTION_STRAIGHT = 0
ACTION_LEFT = 1 

# === RL 观测空间 ===
NEIGHBOR_COUNT = 8          # K
LIDAR_RAYS = 72             # L
LIDAR_RANGE = 250           # 雷达距离
LIDAR_FOV = 360             # 雷达视野
OBS_DIM = 6 + (5 * NEIGHBOR_COUNT) + LIDAR_RAYS # 118维

# === 颜色 ===
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
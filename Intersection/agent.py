# --- agent.py ---

import pygame
import math
import numpy as np
import random
# Support both relative and absolute imports
try:
    from .config import *
    from .utils import get_bezier_point, calculate_angle, wrap_angle_rad, calculate_heading_rad
    from .sensor import Lidar
except ImportError:
    from config import *
    from utils import get_bezier_point, calculate_angle, wrap_angle_rad, calculate_heading_rad
    from sensor import Lidar

# Screen center and lane layout helpers
CX, CY = WIDTH // 2, HEIGHT // 2
MARGIN = 30  # Keep start/end points inside the screen


def build_lane_layout(num_lanes: int):
    """Build lane points and metadata for a given lane count."""
    dir_order = ['N', 'E', 'S', 'W']
    points = {}
    in_by_dir = {d: [] for d in dir_order}
    out_by_dir = {d: [] for d in dir_order}
    dir_of = {}
    idx_of = {}

    for d_idx, d in enumerate(dir_order):
        for j in range(num_lanes):
            offset = LANE_WIDTH_PX * (0.5 + j)
            in_name = f"IN_{d_idx * num_lanes + j + 1}"
            out_name = f"OUT_{d_idx * num_lanes + j + 1}"

            if d == 'N':  # Top, driving south
                points[in_name] = (CX - offset, MARGIN)
                points[out_name] = (CX + offset, MARGIN)
            elif d == 'S':  # Bottom, driving north
                points[in_name] = (CX + offset, HEIGHT - MARGIN)
                points[out_name] = (CX - offset, HEIGHT - MARGIN)
            elif d == 'E':  # Right, driving west
                points[in_name] = (WIDTH - MARGIN, CY - offset)
                points[out_name] = (WIDTH - MARGIN, CY + offset)
            else:  # 'W' Left, driving east
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


# Default lane layout (3 lanes) - used as fallback when not provided
# This will be overridden by environment's lane_layout when Car is created
_DEFAULT_NUM_LANES = 3
_DEFAULT_LANE_LAYOUT = build_lane_layout(_DEFAULT_NUM_LANES)
_DEFAULT_POINTS = _DEFAULT_LANE_LAYOUT['points']

class Car(pygame.sprite.Sprite):
    def __init__(self, start_id, end_id, speed_factor=1.0, respawn_enabled=False,
                 points=None, lane_layout=None, graphics_enabled: bool = True):
        super().__init__()
        
        # 0. Respawn & performance properties
        self.respawn_enabled = respawn_enabled
        self.respawn_count = 0  # Track number of respawns
        self.original_start_id = start_id  # Save original route for respawn
        self.original_end_id = end_id
        # Whether to update pygame Surfaces / Masks each step.
        # For high-speed, headless rollouts (e.g. MCTS) this can be disabled to
        # avoid expensive image rotations and mask recomputation.
        self.graphics_enabled = graphics_enabled
        
        # Use provided points and lane_layout, or fall back to default (3 lanes)
        self.points = points if points is not None else _DEFAULT_POINTS
        self.lane_layout = lane_layout if lane_layout is not None else _DEFAULT_LANE_LAYOUT
        
        # 1. Visual properties
        self.length = CAR_LENGTH
        self.width = CAR_WIDTH
        self.color = random.choice(COLOR_CAR_LIST)
        
        # Create car surface
        self.image_orig = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        pygame.draw.rect(self.image_orig, self.color, (0,0,self.length, self.width), border_radius=4)
        # Windshield
        pygame.draw.rect(self.image_orig, (200,200,200), (self.length*0.7, 2, self.length*0.25, self.width-4))
        
        # 2. Physics state (kinematics)
        start_pos = self.points[start_id]
        self.pos_x = float(start_pos[0])
        self.pos_y = float(start_pos[1])
        
        self.speed = 0.0      # Scalar speed
        self.acc = 0.0        # Acceleration
        self.steering = 0.0   # Steering angle
        
        # 3. Navigation & path
        self.intention = self._determine_intention(start_id, end_id)
        self.path = self._generate_path(start_id, end_id)
        self.path_index = 0   # Closest index on the path
        
        # Determine actual physical target (last point of path)
        self.end_x, self.end_y = self.path[-1]

        # Initial heading based on path
        if len(self.path) > 1:
            self.heading = calculate_heading_rad(self.path[0], self.path[1])
        else:
            self.heading = 0.0

        # 4. Sprite initialization
        self.image = self.image_orig
        self.rect = self.image.get_rect(center=(int(self.pos_x), int(self.pos_y)))
        self.mask = pygame.mask.from_surface(self.image)
        
        # 5. Sensors
        self.lidar = Lidar(self)

    def update(self, action=None):
        """
        Update physics based on action [throttle, steer].
        Action range: -1.0 to 1.0
        """
        if action is None:
            throttle, steer_input = 0.0, 0.0
        else:
            throttle, steer_input = action

        # --- 1. Map inputs ---
        self.acc = throttle * MAX_ACC
        
        # Steering smoothing
        target_steering = steer_input * MAX_STEERING_ANGLE
        self.steering += (target_steering - self.steering) * 0.2
        
        # Friction/drag
        if throttle == 0: 
            self.speed *= 0.95

        # --- 2. Kinematic updates ---
        # Update speed
        self.speed += self.acc * DT
        self.speed = np.clip(self.speed, -PHYSICS_MAX_SPEED/4, PHYSICS_MAX_SPEED)
        if self.speed < 0:
            self.speed = 0
        # Update heading (bicycle model)
        # angular_velocity = (v / L) * tan(delta)
        if abs(self.speed) > 0.1:
            ang_vel = (self.speed / WHEELBASE) * math.tan(self.steering)
            self.heading += ang_vel 
        
        self.heading = wrap_angle_rad(self.heading)

        # Update position
        # Screen Y is inverted (Down is +), so dy needs subtraction for standard math
        self.pos_x += self.speed * math.cos(self.heading)
        self.pos_y -= self.speed * math.sin(self.heading)

        # --- 3. Visual update ---
        # In headless / fast-physics mode (graphics_disabled), we still maintain a
        # reasonable rect for cheap AABB checks, but skip expensive rotation and
        # mask generation, which are major CPU hotspots during MCTS rollouts.
        self.rect.center = (int(self.pos_x), int(self.pos_y))
        if self.graphics_enabled:
            # Full graphical update for interactive / rendered environments
            deg = math.degrees(self.heading)
            self.image = pygame.transform.rotate(self.image_orig, deg)
            self.rect = self.image.get_rect(center=self.rect.center)
            self.mask = pygame.mask.from_surface(self.image)

        # --- 4. Navigation update ---
        # Find nearest point on path
        self._update_path_index()

    def _update_path_index(self):
        """Find the closest point index on the pre-calculated path."""
        search_range = 50 
        start_i = int(self.path_index)
        end_i = min(start_i + search_range, len(self.path))
        
        min_d = float('inf')
        best_i = start_i
        
        for i in range(start_i, end_i):
            px, py = self.path[i]
            d = (px - self.pos_x)**2 + (py - self.pos_y)**2
            if d < min_d:
                min_d = d
                best_i = i
        self.path_index = best_i

    # === Collision Detection ===
    def check_collision(self, road_mask, line_mask, all_vehicles):
        """
        Check collision with Goal, Wall, Yellow Lines, and Cars.
        line_mask: Mask where yellow lines are 1, others are 0.
        """
        w, h = road_mask.get_size()
        
        # --- 1. Success check ---
        end_pt = self.path[-1]
        prev_pt = self.path[-2]
        dx_road = end_pt[0] - prev_pt[0]
        dy_road = end_pt[1] - prev_pt[1]
        
        LATERAL_TOLERANCE = 15 
        LONGITUDINAL_TOLERANCE = 40
        is_success = False
        
        if abs(dx_road) > abs(dy_road): 
            # Horizontal road
            lat_error = abs(self.pos_y - self.end_y)
            long_error = abs(self.pos_x - self.end_x)
            if lat_error < LATERAL_TOLERANCE and long_error < LONGITUDINAL_TOLERANCE:
                is_success = True
        else:
            # Vertical road
            lat_error = abs(self.pos_x - self.end_x)
            long_error = abs(self.pos_y - self.end_y)
            if lat_error < LATERAL_TOLERANCE and long_error < LONGITUDINAL_TOLERANCE:
                is_success = True

        if is_success:
            return True, "SUCCESS"
        
        # Get car corners for checking
        corners = self.get_corners()
        
        # --- 2. Wall / off-road check ---
        for x, y in corners:
            ix, iy = int(x), int(y)
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                return True, "CRASH_WALL"
            if road_mask.get_at((ix, iy)):
                return True, "CRASH_WALL"
        
        # --- 3. Yellow line check ---
        # Check if any corner touches the yellow line mask
        if line_mask is not None:
            for x, y in corners:
                ix, iy = int(x), int(y)
                # Ensure point is inside screen before checking mask
                if 0 <= ix < w and 0 <= iy < h:
                    if line_mask.get_at((ix, iy)):
                        return True, "CRASH_LINE"

        # --- 4. Vehicle-to-vehicle check ---
        for other in all_vehicles:
            if other is self: continue
            if self.rect.colliderect(other.rect):
                off_x = other.rect.x - self.rect.x
                off_y = other.rect.y - self.rect.y
                if self.mask.overlap(other.mask, (off_x, off_y)):
                    return True, "CRASH_CAR"
        
        return False, "ALIVE"

    def get_corners(self):
        """Calculate 4 corners of the rotated bounding box."""
        cx, cy = self.pos_x, self.pos_y
        l, w = self.length/2, self.width/2
        cos_a = math.cos(self.heading)
        sin_a = math.sin(self.heading)
        
        # Vectors for length (forward) and width (side)
        fx, fy = cos_a * l, -sin_a * l
        sx, sy = sin_a * w, cos_a * w
        
        return [
            (cx + fx + sx, cy + fy + sy), # Front-Right
            (cx + fx - sx, cy + fy - sy), # Front-Left
            (cx - fx - sx, cy - fy - sy), # Rear-Left
            (cx - fx + sx, cy - fy + sy)  # Rear-Right
        ]

    # === RL Observation (118 dims) ===
    def get_observation(self, all_vehicles):
        obs_self = self._get_self_state()
        obs_nei = self._get_neighbor_state(all_vehicles)
        obs_lidar = self.lidar.get_observation_data()
        return np.concatenate([obs_self, obs_nei, obs_lidar])

    def _get_self_state(self):
        # Normalize variables
        norm_x = self.pos_x / WIDTH
        norm_y = self.pos_y / HEIGHT
        norm_v = self.speed / PHYSICS_MAX_SPEED
        norm_theta = self.heading / np.pi 
        
        # Lookahead navigation target
        lookahead = 10
        target_idx = min(int(self.path_index) + lookahead, len(self.path) - 1)
        tx, ty = self.path[target_idx]
        
        # Relative distance to target
        dx_dest = tx - self.pos_x
        dy_dest = ty - self.pos_y
        d_dst = math.hypot(dx_dest, dy_dest) / WIDTH
        
        # Relative angle to target
        angle_to_target = math.atan2(-dy_dest, dx_dest) 
        theta_error = wrap_angle_rad(angle_to_target - self.heading) / np.pi
        
        return np.array([norm_x, norm_y, norm_v, norm_theta, d_dst, theta_error], dtype=np.float32)

    def _get_neighbor_state(self, all_vehicles):
        neighbors = []
        for other in all_vehicles:
            if other is self: continue
            dist = math.hypot(other.pos_x - self.pos_x, other.pos_y - self.pos_y)
            neighbors.append((dist, other))
            
        # Sort by distance
        neighbors.sort(key=lambda x: x[0])
        neighbors = neighbors[:NEIGHBOR_COUNT]
        
        feats = []
        for dist, other in neighbors:
            dx = (other.pos_x - self.pos_x) / WIDTH
            dy = (other.pos_y - self.pos_y) / HEIGHT
            dv = (other.speed - self.speed) / PHYSICS_MAX_SPEED
            dtheta = wrap_angle_rad(other.heading - self.heading) / np.pi
            intent = float(other.intention)
            feats.extend([dx, dy, dv, dtheta, intent])
            
        # Padding
        padding_len = NEIGHBOR_COUNT - len(neighbors)
        if padding_len > 0:
            feats.extend([0.0] * (5 * padding_len))
            
        return np.array(feats, dtype=np.float32)

    @property
    def heading_rad(self): return self.heading

    # === Helpers ===
    def _determine_intention(self, start, end):
        """Classify route as straight / left / right based on lane directions."""
        dir_of = self.lane_layout['dir_of']
        start_dir = dir_of.get(start, None)
        end_dir = dir_of.get(end, None)

        if start_dir is None or end_dir is None:
            return ACTION_LEFT

        opposite = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
        left_turn = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
        right_turn = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}

        if end_dir == opposite[start_dir]:
            return ACTION_STRAIGHT
        if end_dir == left_turn[start_dir]:
            return ACTION_LEFT
        if end_dir == right_turn[start_dir]:
            return ACTION_RIGHT
        return ACTION_LEFT

    def _project_to_box(self, pt):
        """Projects a point onto the intersection box boundary."""
        x, y = pt
        # Calculate turn bound from lane layout (road half width)
        num_lanes = len(self.lane_layout['in_by_dir'].get('N', [])) if self.lane_layout['in_by_dir'].get('N') else 3
        turn_bound = num_lanes * LANE_WIDTH_PX
        bx_l, bx_r = CX - turn_bound, CX + turn_bound
        by_t, by_b = CY - turn_bound, CY + turn_bound
        if y < by_t: return (x, by_t)
        if y > by_b: return (x, by_b)
        if x < bx_l: return (bx_l, y)
        return (bx_r, y)

    def _generate_path(self, start_id, end_id):
        """Generates 3-phase path: Straight -> Curve/Straight -> Straight"""
        p_start = self.points[start_id]
        p_end = self.points[end_id]
        entry_p = self._project_to_box(p_start)
        exit_p = self._project_to_box(p_end)
        
        path = []

        if self.intention in (ACTION_STRAIGHT, ACTION_LEFT):
            # Phase 1: Straight to intersection entry
            for i in range(50):
                t = i/50
                path.append((p_start[0]+(entry_p[0]-p_start[0])*t, p_start[1]+(entry_p[1]-p_start[1])*t))
            
            # Phase 2: Inside intersection
            if self.intention == ACTION_STRAIGHT:
                for i in range(60):
                    t = i/60
                    path.append((entry_p[0]+(exit_p[0]-entry_p[0])*t, entry_p[1]+(exit_p[1]-entry_p[1])*t))
            else:  # ACTION_LEFT
                ctrl = (CX, CY)
                for i in range(60):
                    t = i/60
                    path.append(get_bezier_point(t, entry_p, ctrl, exit_p))
            
            # Phase 3: Straight to end
            for i in range(50):
                t = i/50
                path.append((exit_p[0]+(p_end[0]-exit_p[0])*t, exit_p[1]+(p_end[1]-exit_p[1])*t))
        else:  # ACTION_RIGHT
            # Right turn: use same corner centers as road rounding in env.py,
            # with radius = CORNER_RADIUS + 0.5 * lane width from that corner.
            layout = self.lane_layout
            start_dir = layout['dir_of'].get(start_id, None)
            idx = layout['idx_of'].get(start_id, 0)
            
            # Calculate road half width from lane layout (number of lanes per direction)
            # Get num_lanes from the first direction's lane count
            num_lanes = len(layout['in_by_dir'].get('N', [])) if layout['in_by_dir'].get('N') else 3
            road_half_width = num_lanes * LANE_WIDTH_PX

            # Corner centers used in Road.draw():
            # (cx - rw - cr, cy - rw - cr),
            # (cx + rw + cr, cy - rw - cr),
            # (cx - rw - cr, cy + rw + cr),
            # (cx + rw + cr, cy + rw + cr)
            if start_dir == 'N':      # Turning right from North to West
                cx_c, cy_c = CX - road_half_width - CORNER_RADIUS, CY - road_half_width - CORNER_RADIUS
                theta_start, theta_end = 0.0, math.pi / 2  # Quadrant inside intersection
            elif start_dir == 'E':    # Turning right from East to North
                cx_c, cy_c = CX + road_half_width + CORNER_RADIUS, CY - road_half_width - CORNER_RADIUS
                theta_start, theta_end = math.pi / 2, math.pi
            elif start_dir == 'S':    # Turning right from South to East
                cx_c, cy_c = CX + road_half_width + CORNER_RADIUS, CY + road_half_width + CORNER_RADIUS
                theta_start, theta_end = math.pi, 3 * math.pi / 2
            else:                     # Turning right from West to South
                cx_c, cy_c = CX - road_half_width - CORNER_RADIUS, CY + road_half_width + CORNER_RADIUS
                theta_start, theta_end = -math.pi / 2, 0.0

            # Radius: road corner radius + half lane width (follow user's spec)
            r = CORNER_RADIUS + 0.5 * LANE_WIDTH_PX

            # Compute arc start/end points on this circle
            arc_start = (cx_c + r * math.cos(theta_start), cx_c * 0 + cy_c + r * math.sin(theta_start))
            arc_end = (cx_c + r * math.cos(theta_end), cx_c * 0 + cy_c + r * math.sin(theta_end))

            # Phase 1: Straight from start to arc_start
            for i in range(50):
                t = i / 50
                path.append((p_start[0] + (arc_start[0] - p_start[0]) * t,
                             p_start[1] + (arc_start[1] - p_start[1]) * t))

            # Phase 2: Quarter-circle arc following road corner geometry
            for i in range(60):
                t = i / 60
                theta = theta_start + (theta_end - theta_start) * t
                x = cx_c + r * math.cos(theta)
                y = cy_c + r * math.sin(theta)
                path.append((x, y))

            # Phase 3: Straight from arc_end to final end point
            for i in range(50):
                t = i / 50
                path.append((arc_end[0] + (p_end[0] - arc_end[0]) * t,
                             arc_end[1] + (p_end[1] - arc_end[1]) * t))
            
        return path

    def plan_autonomous_action(self, road_mask, all_vehicles):
        """
        终极方案 (修复版)：基于路径扫描 + 速度动态权重的绝对防御
        解决连续避让失败的问题。
        """
        
        # --- 1. 基础横向控制 (PID) ---
        lookahead = 12 
        target_idx = min(int(self.path_index) + lookahead, len(self.path) - 1)
        tx, ty = self.path[target_idx]
        heading_error = wrap_angle_rad(math.atan2(-(ty - self.pos_y), tx - self.pos_x) - self.heading)
        steer_cmd = np.clip(heading_error * 3.0, -1.0, 1.0)
        
        # --- 2. 基础纵向控制 (ACC预设) ---
        target_speed = PHYSICS_MAX_SPEED * 0.65
        if self.speed < target_speed: acc_throttle = 0.5
        elif self.speed > target_speed + 1.0: acc_throttle = -0.1
        else: acc_throttle = 0.0

        # --- 3. 鬼影路径扫描 (Ghost Path Scanning) ---
        
        # 扫描参数配置
        SCAN_STEPS = 120      # 向前预测距离
        SCAN_STEP_SIZE = 1   # 采样密度 (越小越准，防止漏掉高速车)
        SAFE_RADIUS = CAR_WIDTH * 2 # 碰撞判定半径 (稍微大一点，给自己留余量)
        
        conflict_detected = False
        min_conflict_dist = float('inf') 
        
        my_dist_to_center = math.hypot(self.pos_x - CX, self.pos_y - CY)
        
        start_idx = int(self.path_index)
        end_idx = min(start_idx + SCAN_STEPS, len(self.path))
        
        # 开始扫描未来路径
        for i in range(start_idx, end_idx, SCAN_STEP_SIZE):
            ghost_x, ghost_y = self.path[i]
            
            for other in all_vehicles:
                if other is self: continue
                
                # 计算其他车目前与我的鬼影点(未来位置)的距离
                dist_other_to_ghost = math.hypot(other.pos_x - ghost_x, other.pos_y - ghost_y)
                
                # 判定：空间是否被占用
                if dist_other_to_ghost < SAFE_RADIUS:
                    
                    # === 排除同向行驶的前车 (交给 ACC 处理) ===
                    angle_diff = abs(wrap_angle_rad(self.heading - other.heading))
                    if angle_diff < math.radians(60): 
                        continue
                    
                    # === 排除并排行驶的车辆 (方法4: 角度差 + 横向距离 + 路径稳定性) ===
                    # 计算从我的当前位置到其他车辆的向量
                    dx_to_other = other.pos_x - self.pos_x
                    dy_to_other = other.pos_y - self.pos_y
                    dist_to_other = math.hypot(dx_to_other, dy_to_other)
                    
                    if dist_to_other > 0:
                        # 我的行驶方向向量
                        my_dir_x = math.cos(self.heading)
                        my_dir_y = -math.sin(self.heading)
                        
                        # 检查角度差：同向并排（接近0度）或反向并排（接近180度）
                        angle_diff_normalized = min(angle_diff, 2 * math.pi - angle_diff)
                        is_parallel_angle = (angle_diff_normalized < math.radians(30)) or (angle_diff_normalized > math.radians(150))
                        
                        if is_parallel_angle:
                            # 计算横向距离（垂直于我的行驶方向）
                            # 纵向距离（沿我的行驶方向）
                            longitudinal_dist = dx_to_other * my_dir_x + dy_to_other * my_dir_y
                            # 横向距离（垂直于我的行驶方向）
                            lateral_dist = math.sqrt(max(0, dist_to_other**2 - longitudinal_dist**2))
                            
                            # 判断是否在侧方（横向距离小，纵向距离不大）
                            is_sideways_lateral = abs(lateral_dist) < LANE_WIDTH_PX * 1.5  # 在1.5个车道宽度内
                            is_not_far_ahead_behind = abs(longitudinal_dist) < CAR_LENGTH * 2  # 不在前方/后方太远
                            
                            if is_sideways_lateral and is_not_far_ahead_behind:
                                # 路径稳定性检查：预测未来位置，检查横向距离是否保持稳定
                                future_dist = 20  # 向前预测距离
                                my_future_x = self.pos_x + my_dir_x * future_dist
                                my_future_y = self.pos_y + my_dir_y * future_dist
                                
                                # 其他车辆的行驶方向向量
                                other_dir_x = math.cos(other.heading)
                                other_dir_y = -math.sin(other.heading)
                                other_future_x = other.pos_x + other_dir_x * future_dist
                                other_future_y = other.pos_y + other_dir_y * future_dist
                                
                                # 计算未来位置的相对向量
                                future_dx = other_future_x - my_future_x
                                future_dy = other_future_y - my_future_y
                                future_dist_mag = math.hypot(future_dx, future_dy)
                                
                                if future_dist_mag > 0:
                                    # 未来位置的横向距离
                                    future_longitudinal = future_dx * my_dir_x + future_dy * my_dir_y
                                    future_lateral = math.sqrt(max(0, future_dist_mag**2 - future_longitudinal**2))
                                    
                                    # 横向距离变化量
                                    lateral_change = abs(future_lateral - lateral_dist)
                                    
                                    # 如果横向距离保持稳定（变化小），说明是并排行驶
                                    is_stable_lateral = lateral_change < LANE_WIDTH_PX * 0.5
                                    
                                    if is_stable_lateral:
                                        continue  # 跳过并排行驶的车辆
                    
                    # === 核心仲裁逻辑 ===
                    should_yield = False
                    
                    other_dist_to_center = math.hypot(other.pos_x - CX, other.pos_y - CY)
                    dist_to_crash = math.hypot(ghost_x - self.pos_x, ghost_y - self.pos_y)

                    # 规则 1: 绝对物理阻挡
                    # 如果障碍物离我非常近 (已经贴脸了)，不管路权，必须停
                    if dist_to_crash < 15.0:
                        should_yield = True

                    # 规则 2: 速度/动量博弈 (解决"第二次碰撞"的关键)
                    # 如果我已经停下(无动量)，而对方很快且正要过来
                    # 即使我离路口中心更近，我也不能起步，因为我起步慢，会截断对方
                    elif self.speed < 1.0 and other.speed > 3.0 and other_dist_to_center < my_dist_to_center + 25:
                        should_yield = True
                        
                    # 规则 3: 常规距离博弈
                    # 大家都在动，谁离路口远谁让
                    elif other_dist_to_center < my_dist_to_center - 5: 
                        should_yield = True
                    
                    # 规则 4: 死锁打破 (ID仲裁)
                    # 距离差不多，速度差不多，按ID或坐标强制规定
                    elif abs(other_dist_to_center - my_dist_to_center) <= 5:
                        if id(self) < id(other):
                            should_yield = True
                            
                    if should_yield:
                        conflict_detected = True
                        min_conflict_dist = min(min_conflict_dist, dist_to_crash)
                        # 这里不 break，继续检查是否有更近的威胁
            
            # 如果在当前鬼影点发现冲突，对于"我"来说，更远的路径没必要看了
            if conflict_detected:
                break

        # --- 4. 执行决策 ---
        
        # 步骤 A: 先计算 ACC (跟车)
        front_dist = self._get_front_car_dist(all_vehicles)
        if front_dist < 30: acc_throttle = -1.0
        elif front_dist < 50: acc_throttle = min(acc_throttle, -0.2)
        
        # 步骤 B: 叠加路口避让
        final_throttle = acc_throttle
        
        if conflict_detected:
            # 根据危机距离，施加刹车
            if min_conflict_dist < 35:
                # 极度危险 / 贴脸：抱死刹车
                final_throttle = -1.0 
            elif min_conflict_dist < 60:
                # 中度危险：强力减速，不要试图加速抢道
                final_throttle = -0.8 
            else:
                # 远端潜在冲突：收油滑行，准备停车
                final_throttle = min(final_throttle, 0.0)
                
        return [final_throttle, steer_cmd]

    def _get_front_car_dist(self, all_vehicles):
        """辅助函数：获取正前方同向车的距离"""
        min_d = float('inf')
        vx, vy = math.cos(self.heading), -math.sin(self.heading)
        for other in all_vehicles:
            if other is self: continue
            dx, dy = other.pos_x - self.pos_x, other.pos_y - self.pos_y
            dist = math.hypot(dx, dy)
            if dist > 80: continue
            
            # 点积判断是否在前方
            dot = (dx * vx + dy * vy) / (dist + 1e-5)
            if dot > 0.8: # 在前方约 35度 范围内
                # 额外检查：是否同向 (防止误判对向来车)
                angle_diff = abs(wrap_angle_rad(self.heading - other.heading))
                if angle_diff < math.radians(45):
                    min_d = min(min_d, dist)
        return min_d
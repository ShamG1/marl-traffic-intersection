# --- agent.py ---

import pygame
import math
import numpy as np
import random
from config import *
from utils import get_bezier_point, calculate_angle, wrap_angle_rad, calculate_heading_rad
from sensor import Lidar

# Screen center and Lane offsets
CX, CY = WIDTH // 2, HEIGHT // 2
OFF_0 = LANE_WIDTH_PX * 0.5  # Inner lane offset
OFF_1 = LANE_WIDTH_PX * 1.5  # Outer lane offset

# Margin to keep start/end points inside the screen
MARGIN = 30 

# Define Start (IN) and End (OUT) coordinates
POINTS = {
    # Top (North) - y = MARGIN
    'IN_1': (CX - OFF_0, MARGIN),     'IN_2': (CX - OFF_1, MARGIN),
    'OUT_1':(CX + OFF_0, MARGIN),     'OUT_2':(CX + OFF_1, MARGIN),
    
    # Right (East) - x = WIDTH - MARGIN
    'IN_3': (WIDTH - MARGIN, CY - OFF_0),  'IN_4': (WIDTH - MARGIN, CY - OFF_1),
    'OUT_3':(WIDTH - MARGIN, CY + OFF_0),  'OUT_4':(WIDTH - MARGIN, CY + OFF_1),
    
    # Bottom (South) - y = HEIGHT - MARGIN
    'IN_5': (CX + OFF_0, HEIGHT - MARGIN), 'IN_6': (CX + OFF_1, HEIGHT - MARGIN),
    'OUT_5':(CX - OFF_0, HEIGHT - MARGIN), 'OUT_6':(CX - OFF_1, HEIGHT - MARGIN),
    
    # Left (West) - x = MARGIN
    'IN_7': (MARGIN, CY + OFF_0),     'IN_8': (MARGIN, CY + OFF_1),
    'OUT_7':(MARGIN, CY - OFF_0),     'OUT_8':(MARGIN, CY - OFF_1),
}

class Car(pygame.sprite.Sprite):
    def __init__(self, start_id, end_id, speed_factor=1.0):
        super().__init__()
        
        # 1. Visual Properties
        self.length = CAR_LENGTH
        self.width = CAR_WIDTH
        self.color = random.choice(COLOR_CAR_LIST)
        
        # Create car surface
        self.image_orig = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        pygame.draw.rect(self.image_orig, self.color, (0,0,self.length, self.width), border_radius=4)
        # Windshield
        pygame.draw.rect(self.image_orig, (200,200,200), (self.length*0.7, 2, self.length*0.25, self.width-4))
        
        # 2. Physics State (Kinematics)
        start_pos = POINTS[start_id]
        self.pos_x = float(start_pos[0])
        self.pos_y = float(start_pos[1])
        
        self.speed = 0.0      # Scalar speed
        self.acc = 0.0        # Acceleration
        self.steering = 0.0   # Steering angle
        
        # 3. Navigation & Path
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

        # 4. Sprite Initialization
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

        # --- 1. Map Inputs ---
        self.acc = throttle * MAX_ACC
        
        # Steering smoothing
        target_steering = steer_input * MAX_STEERING_ANGLE
        self.steering += (target_steering - self.steering) * 0.2
        
        # Friction/Drag
        if throttle == 0: 
            self.speed *= 0.95

        # --- 2. Kinematic Updates ---
        # Update Speed
        self.speed += self.acc * DT
        self.speed = np.clip(self.speed, -PHYSICS_MAX_SPEED/2, PHYSICS_MAX_SPEED)

        # Update Heading (Bicycle Model)
        # angular_velocity = (v / L) * tan(delta)
        if abs(self.speed) > 0.1:
            ang_vel = (self.speed / WHEELBASE) * math.tan(self.steering)
            self.heading += ang_vel 
        
        self.heading = wrap_angle_rad(self.heading)

        # Update Position
        # Screen Y is inverted (Down is +), so dy needs subtraction for standard math
        self.pos_x += self.speed * math.cos(self.heading)
        self.pos_y -= self.speed * math.sin(self.heading)

        # --- 3. Visual Update ---
        self.rect.center = (int(self.pos_x), int(self.pos_y))
        
        # Rotate image (Pygame rotates CCW)
        deg = math.degrees(self.heading)
        self.image = pygame.transform.rotate(self.image_orig, deg)
        self.rect = self.image.get_rect(center=self.rect.center)
        self.mask = pygame.mask.from_surface(self.image)

        # --- 4. Navigation Update ---
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
        
        # --- 1. Success Check ---
        end_pt = self.path[-1]
        prev_pt = self.path[-2]
        dx_road = end_pt[0] - prev_pt[0]
        dy_road = end_pt[1] - prev_pt[1]
        
        LATERAL_TOLERANCE = 15 
        LONGITUDINAL_TOLERANCE = 40
        is_success = False
        
        if abs(dx_road) > abs(dy_road): 
            # Horizontal Road
            lat_error = abs(self.pos_y - self.end_y)
            long_error = abs(self.pos_x - self.end_x)
            if lat_error < LATERAL_TOLERANCE and long_error < LONGITUDINAL_TOLERANCE:
                is_success = True
        else:
            # Vertical Road
            lat_error = abs(self.pos_x - self.end_x)
            long_error = abs(self.pos_y - self.end_y)
            if lat_error < LATERAL_TOLERANCE and long_error < LONGITUDINAL_TOLERANCE:
                is_success = True

        if is_success:
            return True, "SUCCESS"
        
        # Get car corners for checking
        corners = self.get_corners()
        
        # --- 2. Wall / Off-road Check ---
        for x, y in corners:
            ix, iy = int(x), int(y)
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                return True, "CRASH_WALL"
            if road_mask.get_at((ix, iy)):
                return True, "CRASH_WALL"
        
        # --- 3. Yellow Line Check (New) ---
        # Check if any corner touches the yellow line mask
        if line_mask is not None:
            for x, y in corners:
                ix, iy = int(x), int(y)
                # Ensure point is inside screen before checking mask
                if 0 <= ix < w and 0 <= iy < h:
                    if line_mask.get_at((ix, iy)):
                        return True, "CRASH_LINE"

        # --- 4. Vehicle-to-Vehicle Check ---
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
        pair = (start, end)
        straight_pairs = [('IN_6', 'OUT_2'), ('IN_4', 'OUT_8'), ('IN_2', 'OUT_6'), ('IN_8', 'OUT_4')]
        return ACTION_STRAIGHT if pair in straight_pairs else ACTION_LEFT

    def _project_to_box(self, pt):
        """Projects a point onto the intersection box boundary."""
        x, y = pt
        bx_l, bx_r = CX - TURN_BOUND, CX + TURN_BOUND
        by_t, by_b = CY - TURN_BOUND, CY + TURN_BOUND
        if y < by_t: return (x, by_t)
        if y > by_b: return (x, by_b)
        if x < bx_l: return (bx_l, y)
        return (bx_r, y)

    def _generate_path(self, start_id, end_id):
        """Generates 3-phase path: Straight -> Curve/Straight -> Straight"""
        p_start = POINTS[start_id]
        p_end = POINTS[end_id]
        entry_p = self._project_to_box(p_start)
        exit_p = self._project_to_box(p_end)
        
        path = []
        # Phase 1: Straight to intersection entry
        for i in range(50):
            t = i/50
            path.append((p_start[0]+(entry_p[0]-p_start[0])*t, p_start[1]+(entry_p[1]-p_start[1])*t))
        
        # Phase 2: Inside intersection
        if self.intention == ACTION_STRAIGHT:
             for i in range(60):
                t = i/60
                path.append((entry_p[0]+(exit_p[0]-entry_p[0])*t, entry_p[1]+(exit_p[1]-entry_p[1])*t))
        else:
             ctrl = (CX, CY)
             for i in range(60):
                t = i/60
                path.append(get_bezier_point(t, entry_p, ctrl, exit_p))
        
        # Phase 3: Straight to end
        for i in range(50):
            t = i/50
            path.append((exit_p[0]+(p_end[0]-exit_p[0])*t, exit_p[1]+(p_end[1]-exit_p[1])*t))
            
        return path
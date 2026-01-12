# --- sensor.py ---

import pygame
import math
import numpy as np
from config import *

class Lidar:
    def __init__(self, owner_car):
        self.owner = owner_car
        self.rays = []
        self.readings = np.ones(LIDAR_RAYS, dtype=np.float32)
        
        start_angle = -LIDAR_FOV / 2
        step = LIDAR_FOV / (LIDAR_RAYS - 1) if LIDAR_RAYS > 1 else 0
        self.rel_angles = [math.radians(start_angle + i * step) for i in range(LIDAR_RAYS)]

    def update(self, road_mask, all_vehicles):
        self.rays.clear()
        
        cx, cy = self.owner.rect.center
        car_heading = self.owner.heading_rad
        
        # Get screen size for boundary checking
        w, h = road_mask.get_size()
        step_size = 4
        
        for i, rel_angle in enumerate(self.rel_angles):
            ray_angle = car_heading + rel_angle
            # Screen coordinate system: Y axis down, so dy = -sin
            dx = math.cos(ray_angle)
            dy = -math.sin(ray_angle)
            
            hit_found = False
            hit_pos = (cx, cy)
            final_dist = LIDAR_RANGE # Default to max range
            
            # Ray marching
            for dist in range(0, LIDAR_RANGE, step_size):
                check_x = int(cx + dx * dist)
                check_y = int(cy + dy * dist)
                
                # --- 1. Screen boundary check ---
                # If point goes outside screen, consider it shot into void (safe area)
                # Stop detecting this ray, keep final_dist as LIDAR_RANGE
                if check_x < 0 or check_x >= w or check_y < 0 or check_y >= h:
                    # Don't mark hit_found, directly end loop
                    # Means this ray "shot through" screen boundary, didn't hit obstacle
                    break 
                
                # --- 2. Static road detection (Mask) ---
                # 1 = obstacle (grass), 0 = road
                if road_mask.get_at((check_x, check_y)):
                    hit_found = True
                    hit_pos = (check_x, check_y)
                    final_dist = dist
                    break
                
                # --- 3. Dynamic vehicle detection ---
                collision = False
                for car in all_vehicles:
                    if car is self.owner: continue
                    if car.rect.collidepoint(check_x, check_y):
                        collision = True
                        break
                
                if collision:
                    hit_found = True
                    hit_pos = (check_x, check_y)
                    final_dist = dist
                    break
            
            # If loop ends (or breaks due to out of bounds) without hitting, set endpoint to max distance
            if not hit_found:
                hit_pos = (int(cx + dx * LIDAR_RANGE), int(cy + dy * LIDAR_RANGE))
                final_dist = LIDAR_RANGE

            self.rays.append(((cx, cy), hit_pos, hit_found))
            self.readings[i] = final_dist

    def get_observation_data(self):
        return self.readings / LIDAR_RANGE

    def draw(self, screen):
        s = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        for start, end, hit in self.rays:
            if hit:
                pygame.draw.line(s, COLOR_LIDAR_LINE, start, end, 1)
                pygame.draw.circle(s, COLOR_LIDAR_HIT, end, 2)
        screen.blit(s, (0,0))
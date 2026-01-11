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
        
        # 获取屏幕尺寸用于边界判断
        w, h = road_mask.get_size()
        step_size = 4
        
        for i, rel_angle in enumerate(self.rel_angles):
            ray_angle = car_heading + rel_angle
            # 屏幕坐标系: Y轴向下, 所以 dy = -sin
            dx = math.cos(ray_angle)
            dy = -math.sin(ray_angle)
            
            hit_found = False
            hit_pos = (cx, cy)
            final_dist = LIDAR_RANGE # 默认为最大射程
            
            # 射线步进 (Ray Marching)
            for dist in range(0, LIDAR_RANGE, step_size):
                check_x = int(cx + dx * dist)
                check_y = int(cy + dy * dist)
                
                # --- 1. 屏幕边界检查 ---
                # 如果点跑出了屏幕外，认为它是射向了虚空（安全区域）
                # 停止检测这条射线，保留 final_dist 为 LIDAR_RANGE
                if check_x < 0 or check_x >= w or check_y < 0 or check_y >= h:
                    # 不标记 hit_found，直接结束循环
                    # 意味着这条射线"射穿"了屏幕边界，没有撞到障碍
                    break 
                
                # --- 2. 静态路面检测 (Mask) ---
                # 1 = 障碍(草地), 0 = 路面
                if road_mask.get_at((check_x, check_y)):
                    hit_found = True
                    hit_pos = (check_x, check_y)
                    final_dist = dist
                    break
                
                # --- 3. 动态车辆检测 ---
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
            
            # 如果循环结束（或因出界break）都没撞到，终点设为最大距离
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
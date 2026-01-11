# --- utils.py ---

import math
import numpy as np

def get_bezier_point(t, p0, p1, p2):
    """二次贝塞尔曲线"""
    x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
    y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
    return (x, y)

def calculate_angle(p1, p2):
    """计算 Pygame 旋转角度 (角度制)"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rads = math.atan2(-dy, dx) 
    rads %= 2 * math.pi
    return math.degrees(rads)

def wrap_angle_rad(angle):
    """限制弧度在 [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_heading_rad(p1, p2):
    """计算数学方向角 (弧度制)"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(-dy, dx)
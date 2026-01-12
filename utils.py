# --- utils.py ---

import math
import numpy as np

def get_bezier_point(t, p0, p1, p2):
    """Quadratic Bezier curve"""
    x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
    y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
    return (x, y)

def calculate_angle(p1, p2):
    """Calculate Pygame rotation angle (in degrees)"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rads = math.atan2(-dy, dx) 
    rads %= 2 * math.pi
    return math.degrees(rads)

def wrap_angle_rad(angle):
    """Wrap angle in radians to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_heading_rad(p1, p2):
    """Calculate mathematical direction angle (in radians)"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(-dy, dx)
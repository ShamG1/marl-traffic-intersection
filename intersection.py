# --- road.py ---

import pygame
from config import *

class Road:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.cx = WIDTH // 2
        self.cy = HEIGHT // 2
        self.rw = ROAD_HALF_WIDTH
        self.cr = CORNER_RADIUS
        
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.collision_mask = self._generate_collision_mask()
        self.line_mask = self._generate_line_mask()

    def _generate_collision_mask(self):
        """白色=障碍, 黑色=路"""
        mask_surf = pygame.Surface((self.width, self.height))
        mask_surf.fill((255, 255, 255))
        
        # 挖出路口
        pygame.draw.rect(mask_surf, 0, (self.cx - self.rw, 0, self.rw * 2, self.height))
        pygame.draw.rect(mask_surf, 0, (0, self.cy - self.rw, self.width, self.rw * 2))
        
        # 填补死角
        pygame.draw.rect(mask_surf, 0, (self.cx - self.rw - self.cr, self.cy - self.rw - self.cr, self.cr, self.cr))
        pygame.draw.rect(mask_surf, 0, (self.cx + self.rw,      self.cy - self.rw - self.cr, self.cr, self.cr))
        pygame.draw.rect(mask_surf, 0, (self.cx - self.rw - self.cr, self.cy + self.rw,      self.cr, self.cr))
        pygame.draw.rect(mask_surf, 0, (self.cx + self.rw,      self.cy + self.rw,      self.cr, self.cr))

        # 画回圆角
        pygame.draw.circle(mask_surf, (255,255,255), (self.cx - self.rw - self.cr, self.cy - self.rw - self.cr), self.cr)
        pygame.draw.circle(mask_surf, (255,255,255), (self.cx + self.rw + self.cr, self.cy - self.rw - self.cr), self.cr)
        pygame.draw.circle(mask_surf, (255,255,255), (self.cx - self.rw - self.cr, self.cy + self.rw + self.cr), self.cr)
        pygame.draw.circle(mask_surf, (255,255,255), (self.cx + self.rw + self.cr, self.cy + self.rw + self.cr), self.cr)
        
        return pygame.mask.from_threshold(mask_surf, (255, 255, 255), (10, 10, 10))

    def _generate_line_mask(self):
        """Generate a mask where yellow lines are white (1) and others are black (0)"""
        mask_surf = pygame.Surface((self.width, self.height))
        mask_surf.fill(0) # Black background
        
        stop_offset = self.rw + self.cr
        # Draw double yellow lines (same logic as _draw_markings but simpler)
        # Vertical Lines
        pygame.draw.line(mask_surf, (255,255,255), (self.cx-2, 0), (self.cx-2, self.cy-stop_offset), 2)
        pygame.draw.line(mask_surf, (255,255,255), (self.cx+2, 0), (self.cx+2, self.cy-stop_offset), 2)
        pygame.draw.line(mask_surf, (255,255,255), (self.cx-2, self.height), (self.cx-2, self.cy+stop_offset), 2)
        pygame.draw.line(mask_surf, (255,255,255), (self.cx+2, self.height), (self.cx+2, self.cy+stop_offset), 2)
        
        # Horizontal Lines
        pygame.draw.line(mask_surf, (255,255,255), (0, self.cy-2), (self.cx-stop_offset, self.cy-2), 2)
        pygame.draw.line(mask_surf, (255,255,255), (0, self.cy+2), (self.cx-stop_offset, self.cy+2), 2)
        pygame.draw.line(mask_surf, (255,255,255), (self.width, self.cy-2), (self.cx+stop_offset, self.cy-2), 2)
        pygame.draw.line(mask_surf, (255,255,255), (self.width, self.cy+2), (self.cx+stop_offset, self.cy+2), 2)
        
        return pygame.mask.from_threshold(mask_surf, (255, 255, 255), (10, 10, 10))
        
    def draw(self, screen, show_lane_ids=False):
        screen.fill(COLOR_GRASS)
        # 基础路面
        pygame.draw.rect(screen, COLOR_ROAD, (self.cx - self.rw, 0, self.rw * 2, self.height))
        pygame.draw.rect(screen, COLOR_ROAD, (0, self.cy - self.rw, self.width, self.rw * 2))
        
        # 圆角处理
        corners = [
            (self.cx - self.rw - self.cr, self.cy - self.rw - self.cr),
            (self.cx + self.rw,      self.cy - self.rw - self.cr),
            (self.cx - self.rw - self.cr, self.cy + self.rw),
            (self.cx + self.rw,      self.cy + self.rw)
        ]
        for x, y in corners:
            pygame.draw.rect(screen, COLOR_ROAD, (x, y, self.cr, self.cr))
            
        centers = [
            (self.cx - self.rw - self.cr, self.cy - self.rw - self.cr),
            (self.cx + self.rw + self.cr, self.cy - self.rw - self.cr),
            (self.cx - self.rw - self.cr, self.cy + self.rw + self.cr),
            (self.cx + self.rw + self.cr, self.cy + self.rw + self.cr)
        ]
        for c in centers:
            pygame.draw.circle(screen, COLOR_GRASS, c, self.cr)

        self._draw_markings(screen)
        if show_lane_ids:
            self._draw_lane_ids(screen)

    def _draw_markings(self, screen):
        stop_offset = self.rw + self.cr
        # 双黄线
        pygame.draw.line(screen, COLOR_YELLOW, (self.cx-2, 0), (self.cx-2, self.cy-stop_offset), 2)
        pygame.draw.line(screen, COLOR_YELLOW, (self.cx+2, 0), (self.cx+2, self.cy-stop_offset), 2)
        pygame.draw.line(screen, COLOR_YELLOW, (self.cx-2, self.height), (self.cx-2, self.cy+stop_offset), 2)
        pygame.draw.line(screen, COLOR_YELLOW, (self.cx+2, self.height), (self.cx+2, self.cy+stop_offset), 2)
        
        pygame.draw.line(screen, COLOR_YELLOW, (0, self.cy-2), (self.cx-stop_offset, self.cy-2), 2)
        pygame.draw.line(screen, COLOR_YELLOW, (0, self.cy+2), (self.cx-stop_offset, self.cy+2), 2)
        pygame.draw.line(screen, COLOR_YELLOW, (self.width, self.cy-2), (self.cx+stop_offset, self.cy-2), 2)
        pygame.draw.line(screen, COLOR_YELLOW, (self.width, self.cy+2), (self.cx+stop_offset, self.cy+2), 2)

        # 停止线
        stop_w = 4
        pygame.draw.line(screen, COLOR_WHITE, (self.cx-self.rw, self.cy-stop_offset), (self.cx, self.cy-stop_offset), stop_w)
        pygame.draw.line(screen, COLOR_WHITE, (self.cx, self.cy+stop_offset), (self.cx+self.rw, self.cy+stop_offset), stop_w)
        pygame.draw.line(screen, COLOR_WHITE, (self.cx-stop_offset, self.cy), (self.cx-stop_offset, self.cy+self.rw), stop_w)
        pygame.draw.line(screen, COLOR_WHITE, (self.cx+stop_offset, self.cy), (self.cx+stop_offset, self.cy-self.rw), stop_w)

        # 虚线
        self._draw_dash(screen, (self.cx - LANE_WIDTH_PX, 0), (self.cx - LANE_WIDTH_PX, self.cy - stop_offset))
        self._draw_dash(screen, (self.cx + LANE_WIDTH_PX, 0), (self.cx + LANE_WIDTH_PX, self.cy - stop_offset))
        self._draw_dash(screen, (self.cx - LANE_WIDTH_PX, self.height), (self.cx - LANE_WIDTH_PX, self.cy + stop_offset))
        self._draw_dash(screen, (self.cx + LANE_WIDTH_PX, self.height), (self.cx + LANE_WIDTH_PX, self.cy + stop_offset))
        self._draw_dash(screen, (0, self.cy - LANE_WIDTH_PX), (self.cx - stop_offset, self.cy - LANE_WIDTH_PX))
        self._draw_dash(screen, (0, self.cy + LANE_WIDTH_PX), (self.cx - stop_offset, self.cy + LANE_WIDTH_PX))
        self._draw_dash(screen, (self.width, self.cy - LANE_WIDTH_PX), (self.cx + stop_offset, self.cy - LANE_WIDTH_PX))
        self._draw_dash(screen, (self.width, self.cy + LANE_WIDTH_PX), (self.cx + stop_offset, self.cy + LANE_WIDTH_PX))

    def _draw_dash(self, screen, start, end):
        x1, y1 = start
        x2, y2 = end
        dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
        if dist == 0: return
        dx, dy = (x2-x1)/dist, (y2-y1)/dist
        dash_len = 20
        steps = int(dist/(dash_len*2))
        for i in range(steps+1):
            sx = x1 + dx*i*dash_len*2
            sy = y1 + dy*i*dash_len*2
            ex = sx + dx*dash_len
            ey = sy + dy*dash_len
            if (dx>0 and ex>x2) or (dx<0 and ex<x2) or (dy>0 and ey>y2) or (dy<0 and ey<y2): continue
            pygame.draw.line(screen, COLOR_WHITE, (sx,sy), (ex,ey), 2)

    def _draw_lane_ids(self, screen):
        COLOR_IN, COLOR_OUT = (0,0,200), (200,0,0)
        m = 35
        def label(t, x, y, c):
            s = self.font.render(t, True, (255,255,255))
            r = s.get_rect(center=(x,y))
            pygame.draw.rect(screen, c, r.inflate(10,6), border_radius=4)
            screen.blit(s, r)
        label("IN_1", self.cx - LANE_WIDTH_PX*0.5, m, COLOR_IN)
        label("IN_2", self.cx - LANE_WIDTH_PX*1.5, m, COLOR_IN)
        label("IN_3", self.width-m, self.cy - LANE_WIDTH_PX*0.5, COLOR_IN)
        label("IN_4", self.width-m, self.cy - LANE_WIDTH_PX*1.5, COLOR_IN)
        label("IN_5", self.cx + LANE_WIDTH_PX*0.5, self.height-m, COLOR_IN)
        label("IN_6", self.cx + LANE_WIDTH_PX*1.5, self.height-m, COLOR_IN)
        label("IN_7", m, self.cy + LANE_WIDTH_PX*0.5, COLOR_IN)
        label("IN_8", m, self.cy + LANE_WIDTH_PX*1.5, COLOR_IN)
        
        label("OUT_1", self.cx + LANE_WIDTH_PX*0.5, m, COLOR_OUT)
        label("OUT_2", self.cx + LANE_WIDTH_PX*1.5, m, COLOR_OUT)
        label("OUT_3", self.width-m, self.cy + LANE_WIDTH_PX*0.5, COLOR_OUT)
        label("OUT_4", self.width-m, self.cy + LANE_WIDTH_PX*1.5, COLOR_OUT)
        label("OUT_5", self.cx - LANE_WIDTH_PX*0.5, self.height-m, COLOR_OUT)
        label("OUT_6", self.cx - LANE_WIDTH_PX*1.5, self.height-m, COLOR_OUT)
        label("OUT_7", m, self.cy - LANE_WIDTH_PX*0.5, COLOR_OUT)
        label("OUT_8", m, self.cy - LANE_WIDTH_PX*1.5, COLOR_OUT)
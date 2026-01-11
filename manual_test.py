# --- main_manual.py ---

import pygame
import numpy as np
import math
from config import *
from intersection import Road
from agent import Car

def reset_game():
    """初始化游戏场景：一辆玩家车，一辆前方静止的障碍车"""
    # 1. 玩家车 (南向北左转)
    p_car = Car('IN_5', 'OUT_7')
    
    # 2. 障碍车 (完全一样的路线，但是位置在前)
    obs_car = Car('IN_5', 'OUT_7')
    
    # [关键] 手动修改障碍车位置
    # IN_5 是从下往上(Y减小)，我们将Y减去200像素，让它停在前方
    obs_car.pos_y -= 200 
    
    # 强制更新一下图形位置和Mask
    obs_car.rect.center = (int(obs_car.pos_x), int(obs_car.pos_y))
    obs_car.mask = pygame.mask.from_surface(obs_car.image)
    
    # 3. 编组
    sprites = pygame.sprite.Group()
    sprites.add(p_car)
    sprites.add(obs_car)
    
    return p_car, obs_car, sprites

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MARL Intersection")
    clock = pygame.time.Clock()

    road = Road()
    
    # 初始化场景
    player_car, obstacle_car, all_sprites = reset_game()
    
    font = pygame.font.SysFont("Arial", 24, bold=True)
    game_state = "ALIVE"

    running = True
    while running:
        # 1. 键盘控制
        keys = pygame.key.get_pressed()
        throttle = 0.0
        steer = 0.0
        
        if keys[pygame.K_UP]: throttle = 0.3      
        if keys[pygame.K_DOWN]: throttle = -0.5   
        if keys[pygame.K_LEFT]: steer = 1.0       
        if keys[pygame.K_RIGHT]: steer = -1.0     
            
        action = [throttle, steer]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                player_car, obstacle_car, all_sprites = reset_game()
                game_state = "ALIVE"
                print("Game Reset!")

        # 2. 物理更新
        if game_state == "ALIVE":
            player_car.update(action)
            # 障碍车不动
            obstacle_car.update([0.0, 0.0])
            
            # 雷达更新
            player_car.lidar.update(road.collision_mask, all_sprites)
            
            # 碰撞检测
            done, info = player_car.check_collision(road.collision_mask, road.line_mask, all_sprites)
            
            if done:
                game_state = info
                print(f"💥 Collision Detected: {info}")

        # 3. 绘图
        road.draw(screen, show_lane_ids=True)
        all_sprites.draw(screen)
        
        # 绘制雷达
        player_car.lidar.draw(screen)
        
        # 标记障碍车
        pygame.draw.circle(screen, (255, 0, 0), obstacle_car.rect.center, 10, 2)

        # === [补回来的部分] 可视化导航路径 ===
        # 1. 画目标预瞄点 (黄点)
        lookahead = 10
        idx = min(player_car.path_index + lookahead, len(player_car.path)-1)
        target_pt = player_car.path[idx]
        pygame.draw.circle(screen, (255, 255, 0), (int(target_pt[0]), int(target_pt[1])), 5)
        
        # 2. 画完整轨迹线 (青色线)
        if len(player_car.path) > 1:
            pygame.draw.lines(screen, (0, 255, 255), False, player_car.path, 1)

        # 状态文字
        color = (0, 255, 0) if game_state == "ALIVE" else (255, 0, 0)
        s_surf = font.render(f"State: {game_state} | Spd: {player_car.speed:.1f}", True, color)
        screen.blit(s_surf, (20, 20))
        
        # hint_surf = font.render("Path is Cyan. Target is Yellow. Red circle is Obstacle.", True, (255, 255, 255))
        # screen.blit(hint_surf, (20, 50))
        
        if game_state != "ALIVE":
            r_surf = font.render("Press R to Reset", True, (255,255,0))
            screen.blit(r_surf, (20, 80))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == '__main__':
    main()
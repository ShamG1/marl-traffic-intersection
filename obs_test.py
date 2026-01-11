import pygame
import random
import numpy as np
from config import *
from intersection import Road
from agent import Car

# --- 辅助函数：格式化打印观测数据 ---
def print_formatted_obs(obs):
    """
    将 118 维向量切分并打印，方便调试查看
    obs 结构: [Self(6), Neighbors(40), Lidar(72)]
    """
    # 1. 切片索引
    idx_self_end = 6
    idx_nei_end = 6 + (5 * NEIGHBOR_COUNT) # 6 + 40 = 46
    
    # 2. 提取数据
    vec_self = obs[:idx_self_end]
    vec_nei = obs[idx_self_end:idx_nei_end]
    vec_lidar = obs[idx_nei_end:]
    
    print("\n" + "="*50)
    print(f"🔍 选中车辆观测数据 (总维度: {len(obs)})")
    
    # --- 自车状态 ---
    print(f"\n🚗 [Self State] (6 dim):")
    headers = ["Norm_X", "Norm_Y", "Norm_V", "Heading", "Dist_Dst", "Theta_Err"]
    print(f"   {vec_self}")
    # 打印对应含义
    info = ", ".join([f"{h}:{v:.2f}" for h, v in zip(headers, vec_self)])
    print(f"   解析: {info}")

    # --- 邻居信息 ---
    print(f"\n🚙 [Neighbors] ({5*NEIGHBOR_COUNT} dim - Top {NEIGHBOR_COUNT} nearest):")
    # 把一维向量 reshape 成 (8, 5) 方便看
    nei_matrix = vec_nei.reshape(NEIGHBOR_COUNT, 5)
    print("   [Rel_X, Rel_Y, Rel_V, Rel_Theta, Intention]")
    for i, row in enumerate(nei_matrix):
        #只打印非全0的邻居（即真实存在的邻居）
        if not np.all(row == 0):
            print(f"   N{i+1}: {row}")
        else:
            print(f"   N{i+1}: [Empty / Padding]")
            break # 后面都是padding，不用打印了

    # --- 雷达信息 ---
    print(f"\n📡 [Lidar] ({len(vec_lidar)} dim):")
    # 打印简报：最小距离、最大距离、平均距离
    min_dist = np.min(vec_lidar)
    avg_dist = np.mean(vec_lidar)
    print(f"   Min Dist: {min_dist:.4f} (0=Crash, 1=Clear)")
    print(f"   Avg Dist: {avg_dist:.4f}")
    # 打印前10个和后10个数据作为示例
    print(f"   Raw (First 10): {vec_lidar[:10]}")
    print(f"   Raw (Last 10):  {vec_lidar[-10:]}")
    print("="*50 + "\n")

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()

    road = Road()
    all_sprites = pygame.sprite.Group()
    
    # 路线定义
    ROUTES = [
        ('IN_6', 'OUT_2'), ('IN_5', 'OUT_7'),
        ('IN_4', 'OUT_8'), ('IN_3', 'OUT_5'),
        ('IN_2', 'OUT_6'), ('IN_1', 'OUT_3')
    ]
    
    ADD_CAR_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(ADD_CAR_EVENT, 1500) 

    show_debug = True
    
    # [NEW] 当前选中的车辆
    selected_car = None
    
    # [NEW] 打印计时器（避免每帧都打印，刷屏太快）
    print_timer = 0
    PRINT_INTERVAL = 30 # 每30帧(约0.5秒)打印一次

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    show_debug = not show_debug
            
            elif event.type == ADD_CAR_EVENT:
                route = random.choice(ROUTES)
                spd = random.uniform(0.8, 1.2)
                car = Car(route[0], route[1], speed_factor=spd)
                all_sprites.add(car)
            
            # [NEW] 鼠标点击选择车辆
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # 左键点击
                    pos = pygame.mouse.get_pos()
                    # 找到鼠标点击位置下的所有车辆
                    clicked = [s for s in all_sprites if s.rect.collidepoint(pos)]
                    if clicked:
                        selected_car = clicked[0] # 选中最上面的一辆
                        print(f"✅ 选中车辆 ID: {id(selected_car)}")
                    else:
                        selected_car = None # 点击空地取消选择
                        print("❌ 取消选择")

        # 逻辑更新
        all_sprites.update()
        
        # 雷达与观测更新
        for car in all_sprites:
            car.lidar.update(road.collision_mask, all_sprites)

        # 绘图
        road.draw(screen, show_lane_ids=show_debug)
        all_sprites.draw(screen)
        
        if show_debug:
            for car in all_sprites:
                car.lidar.draw(screen)

        # [NEW] 处理选中车辆的高亮和打印
        if selected_car is not None:
            # 1. 检查车辆是否还活着（可能跑出屏幕被销毁了）
            if selected_car.alive():
                # 2. 画一个黄色的框
                pygame.draw.rect(screen, (255, 255, 0), selected_car.rect, 3)
                
                # 3. 定时打印观测数据
                print_timer += 1
                if print_timer >= PRINT_INTERVAL:
                    # 获取该车的 118维 观测向量
                    obs = selected_car.get_observation(all_sprites)
                    print_formatted_obs(obs)
                    print_timer = 0
            else:
                print("⚠️ 选中车辆已销毁")
                selected_car = None

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == '__main__':
    main()
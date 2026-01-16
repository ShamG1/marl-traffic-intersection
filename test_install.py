# 从任何地方导入
from Intersection import IntersectionEnv, DEFAULT_REWARD_CONFIG
import pygame
import numpy as np

# 创建环境
config = {
    'traffic_flow': False,
    'num_agents': 1,
    'num_lanes': 2,
    'render_mode': 'human',
    'max_steps': 2000,
    'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
}

env = IntersectionEnv(config)

# 重置环境（必须！否则没有内容显示）
obs, info = env.reset()

# 渲染循环 - 保持窗口打开
running = True
while running:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                running = False
    
    # 渲染环境
    env.render(show_lane_ids=True, show_lidar=True)
    
    # 可选：执行一步（让车辆移动）
    # action = np.array([0.3, 0.0])  # [throttle, steer]
    # obs, reward, terminated, truncated, info = env.step(action)
    # if terminated or truncated:
    #     obs, info = env.reset()

# 关闭环境
env.close()
print("窗口已关闭")
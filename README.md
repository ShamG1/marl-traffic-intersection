# MARL 无信号路口环境 🚦

这是一个基于 **Pygame** 开发的轻量级多智能体强化学习 (MARL) 无信号路口仿真环境。

项目实现了基于 **运动学自行车模型** 的车辆控制、**贝塞尔曲线** 导航、**线束激光雷达** 感知以及符合学术标准的 **RL 观测空间**。

![Screenshot](/assets/screenshot.png)

---

## 📂 文件结构

### 核心文件
- `env.py`：RL 环境接口，集成了奖励计算和交通流生成
- `agent.py`：智能体类，包含物理更新、碰撞检测、RL 状态获取
- `sensor.py`：激光雷达逻辑
- `config.py`：全局参数配置（物理、RL 维度、默认奖励配置）
- `utils.py`：数学工具函数

### 测试文件
- `manual_test.py`：手动控制测试（无交通流，显示雷达和路径）
- `traffic_flow_test.py`：交通流测试（单智能体+NPC，显示车道ID和主车雷达）

---

## 🚀 快速开始

### 基本使用

```python
from env import IntersectionEnv
from config import DEFAULT_REWARD_CONFIG
import numpy as np

# 创建环境
config = {
    'traffic_flow': True,  # True=单智能体+交通流, False=多智能体
    'traffic_density': 0.5,  # 交通密度
    'render_mode': 'human',  # 'human' 或 None
    'max_steps': 2000,
    'reward_config': DEFAULT_REWARD_CONFIG['reward_config']
}

env = IntersectionEnv(config)

# 重置环境
obs, info = env.reset()

# 运行一个回合
for step in range(100):
    # 随机动作 [throttle, steer]，范围 [-1, 1]
    action = np.array([0.5, 0.0])
    
    # 执行一步
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # 渲染（如果 render_mode='human'）
    env.render()
    
    if done:
        obs, info = env.reset()

env.close()
```

---

## ⚙️ 环境配置

### 单智能体模式（带交通流）

```python
config = {
    'traffic_flow': True,  # 启用交通流
    'traffic_density': 0.5,  # 交通密度
    'render_mode': 'human',
    'max_steps': 2000,
    'ego_routes': [('IN_6', 'OUT_2')],  # 可选：指定 ego 车辆路线
    'reward_config': {...}
}
```

### 多智能体模式（无交通流）

```python
config = {
    'traffic_flow': False,  # 禁用交通流
    'num_agents': 1,  # 智能体数量
    'use_team_reward': True,  # 是否使用团队奖励
    'render_mode': 'human',
    'max_steps': 2000,
    'ego_routes': [  # 可选：指定每个智能体的路线
        ('IN_6', 'OUT_2'),
        ('IN_4', 'OUT_8'),
        ('IN_5', 'OUT_7')
    ],
    'reward_config': {...}
}
```

---

## 🎯 奖励函数配置

奖励函数已集成在 `env.py` 中，可以通过 `reward_config` 参数自定义：

```python
from config import DEFAULT_REWARD_CONFIG

# 使用默认配置
config = {
    'reward_config': DEFAULT_REWARD_CONFIG['reward_config']
}

# 自定义奖励配置
custom_reward_config = {
    'progress_scale': 10.0,             # 进度奖励系数
    'stuck_speed_threshold': 1.0,        # 卡住速度阈值 (m/s)
    'stuck_penalty': -0.01,              # 卡住惩罚
    'crash_vehicle_penalty': -10.0,      # 车辆碰撞惩罚
    'crash_object_penalty': -5.0,       # 物体/墙壁碰撞惩罚（包括离开道路）
    'success_reward': 10.0,             # 成功到达奖励
    'action_smoothness_scale': -0.02,    # 动作平滑度系数（负值鼓励平滑）
    'team_alpha': 0.2,                  # 团队奖励混合系数（仅多智能体）
}

config = {
    'reward_config': custom_reward_config
}
```

### 奖励组成

**个体奖励** (Individual Reward):
```
r_i^ind(t) = r_prog(t) + r_stuck(t) + r_crashV(t) + 
             r_crashO(t) + r_succ(t) + r_smooth(t)
```

**团队奖励** (Team Reward, 仅多智能体模式):
```
r_i^mix(t) = (1 - α) * r_i^ind(t) + α * r̄^ind(t)
```

其中：
- `r_prog`: 基于距离减少的进度奖励（归一化后乘以 `progress_scale`）
- `r_stuck`: 速度过低时的卡住惩罚（每步）
- `r_crashV`: 车辆碰撞惩罚（一次性）
- `r_crashO`: 物体/墙壁碰撞惩罚（一次性，包括离开道路）
- `r_succ`: 成功到达目标点的奖励（一次性）
- `r_smooth`: 动作平滑度奖励（鼓励平滑驾驶，每步）

**注意**：`out_of_road_penalty` 已移除，因为离开道路的情况已由 `crash_object_penalty` 通过 `CRASH_WALL` 状态覆盖。

---

## 🚗 交通流设置

### 交通密度

`traffic_density` 参数控制 NPC 车辆的生成频率：

```python
config = {
    'traffic_flow': True,
    'traffic_density': 0.5,  # 交通密度 
}
```

### NPC 车辆行为

NPC 车辆使用自主驾驶算法：
- **横向控制**：PID 控制器跟踪路径
- **纵向控制**：ACC（自适应巡航控制）避免碰撞
- **碰撞处理**：NPC 之间碰撞时，两车都会被移除

---

## 🎮 运行测试

### 手动控制测试（无交通流）

```bash
python manual_test.py
```

- 显示雷达和导航路径
- 无 NPC 车辆
- 适合测试基本控制

### 交通流测试

```bash
python traffic_test.py
```

- 显示车道ID、主车雷达和导航路径
- 包含 NPC 交通流
- 适合测试避障和导航

---

## 📝 TODO

- [x] 集成奖励计算到环境
- [x] 集成交通流生成到环境
- [x] 支持单智能体和多智能体模式
- [x] 使用 MAPPO 算法训练
- [ ] 使用 MCTS 算法训练

---

## 📄 许可证

本项目遵循 MIT License。

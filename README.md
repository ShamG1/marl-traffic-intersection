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

### 安装

#### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/ShamG1/marl-traffic-intersection.git
cd marl-traffic-intersection

# 安装包（开发模式，推荐）
pip install -e .

# 或普通安装
pip install .
```

安装后，可以从任何地方导入：

```python
from Intersection import IntersectionEnv, DEFAULT_REWARD_CONFIG
```

### 基本使用

```python
from Intersection import IntersectionEnv, DEFAULT_REWARD_CONFIG
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
from Intersection import DEFAULT_REWARD_CONFIG

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

安装包后，有多种方式运行测试脚本：

### 方法 1：作为 Python 模块运行（推荐）

```bash
# 手动控制测试（无交通流，显示雷达和路径）
python -m Intersection.manual_test

# 交通流测试（有 NPC 车辆，显示车道ID和主车雷达）
python -m Intersection.traffic_test
```

### 方法 2：直接运行脚本文件

```bash
# 从项目目录运行
python Intersection/manual_test.py
python Intersection/traffic_test.py

# 或使用完整路径（从任何地方）
python "e:\Intersection\manual_test.py"
```

### 方法 3：在代码中导入使用

```python
from Intersection.manual_test import main as manual_test_main
from Intersection.traffic_test import main as traffic_test_main

# 运行测试
manual_test_main()  # 手动控制测试
traffic_test_main()  # 交通流测试
```

### 测试脚本说明

#### manual_test.py - 手动控制测试

**功能**：
- 手动控制单个智能体
- 无交通流（只有你的车辆）
- 显示雷达和导航路径

**控制方式**：
- `UP/DOWN` 箭头：油门控制
- `LEFT/RIGHT` 箭头：转向控制
- `R` 键：重置环境
- `ESC/Q` 键：退出程序

**适用场景**：
- 测试基本车辆控制
- 验证环境基本功能
- 调试导航路径

#### traffic_test.py - 交通流测试

**功能**：
- 手动控制单个智能体
- 包含 NPC 交通流
- 显示车道ID、主车雷达和导航路径

**控制方式**：
- 与 `manual_test.py` 相同

**适用场景**：
- 测试避障能力
- 验证交通流交互
- 测试多车辆场景下的导航

### 注意事项

- 确保已安装包：`pip install -e .`
- 确保在正确的 conda 环境中运行
- 窗口会保持打开，按 `ESC` 或 `Q` 退出

---

## 📝 TODO

- [x] 集成奖励计算到环境
- [x] 集成交通流生成到环境
- [x] 支持单智能体和多智能体模式
- [x] 使用 MAPPO 算法训练
- [ ] 使用 MCTS 算法训练
- [ ] 支持多地图测试

---

## 📄 许可证

本项目遵循 MIT License。

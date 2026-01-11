# MARL 交通十字路口环境 🚦

这是一个基于 **Pygame** 开发的轻量级多智能体强化学习 (MARL) 十字路口仿真环境。

项目实现了基于 **运动学自行车模型** 的车辆控制、**贝塞尔曲线** 导航、**线束激光雷达** 感知以及符合学术标准的 **RL 观测空间**。

![Screenshot](screenshot.png)

*(建议运行程序后将截图命名为 `screenshot.png` 放在仓库根目录，README 将自动展示)*

---

## 📂 文件结构

- `agent.py`：智能体类，包含物理更新、碰撞检测、RL 状态获取
- `road.py`：道路环境绘制、静态障碍 Mask 生成
- `sensor.py`：激光雷达逻辑
- `config.py`：全局参数配置（物理、RL 维度）
- `utils.py`：数学工具函数

---

## 📝 TODO

- [ ] 接入 Gym/PettingZoo 接口
- [ ] 使用 PPO / MADDPG 算法训练

---

## 许可证

本项目遵循 MIT License。


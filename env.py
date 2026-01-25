"""Fast C++-backed intersection environment with Python-compatible API.

Rendering is delegated to the C++ OpenGL/GLFW renderer.

Notes:
- Multi-agent mode: returns obs (N,127), rewards (N,), terminated, truncated, info.
- traffic_flow mode (single-agent): returns obs (127,), reward scalar, terminated, truncated, info.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    from . import cpp_backend
except ImportError:
    import cpp_backend  # type: ignore

# Handle both relative and absolute imports
try:
    from .utils import (
        DEFAULT_ROUTE_MAPPING_2LANES,
        DEFAULT_ROUTE_MAPPING_3LANES,
        build_lane_layout,
    )
except ImportError:
    from utils import (
        DEFAULT_ROUTE_MAPPING_2LANES,
        DEFAULT_ROUTE_MAPPING_3LANES,
        build_lane_layout,
    )

# Local default reward config (mirrors Intersection/config.py)
DEFAULT_REWARD_CONFIG = {
    "use_team_reward": False,
    "traffic_flow": False,
    "reward_config": {
        "progress_scale": 10.0,
        "stuck_speed_threshold": 1.0,
        "stuck_penalty": -0.01,
        "crash_vehicle_penalty": -10.0,
        "crash_object_penalty": -5.0,
        "success_reward": 10.0,
        "action_smoothness_scale": -0.02,
        "team_alpha": 0.2,
    },
}


def _apply_reward_config(env: Any, reward_cfg: Dict[str, Any]) -> None:
    if not hasattr(env, "reward_config"):
        return
    rc = env.reward_config

    if "progress_scale" in reward_cfg:
        rc.k_prog = float(reward_cfg["progress_scale"])
    if "stuck_speed_threshold" in reward_cfg:
        rc.v_min_ms = float(reward_cfg["stuck_speed_threshold"])
    if "stuck_penalty" in reward_cfg:
        rc.k_stuck = float(reward_cfg["stuck_penalty"])
    if "crash_vehicle_penalty" in reward_cfg:
        rc.k_cv = float(reward_cfg["crash_vehicle_penalty"])
    if "crash_object_penalty" in reward_cfg:
        rc.k_co = float(reward_cfg["crash_object_penalty"])
    if "success_reward" in reward_cfg:
        rc.k_succ = float(reward_cfg["success_reward"])
    if "action_smoothness_scale" in reward_cfg:
        rc.k_sm = float(reward_cfg["action_smoothness_scale"])
    if "team_alpha" in reward_cfg:
        rc.alpha = float(reward_cfg["team_alpha"])


class IntersectionEnv:
    def __init__(self, config: Dict[str, Any] | None = None):
        if config is None:
            config = {}

        self.traffic_flow = bool(config.get("traffic_flow", False))

        if self.traffic_flow:
            self.num_agents = 1
        else:
            self.num_agents = int(config.get("num_agents", 1))

        self.num_lanes = int(config.get("num_lanes", 3))
        self.render_mode = config.get("render_mode", None)
        self.show_lane_ids = bool(config.get("show_lane_ids", False))
        self.show_lidar = bool(config.get("show_lidar", False))

        use_team = bool(config.get("use_team_reward", DEFAULT_REWARD_CONFIG.get("use_team_reward", False)))
        if self.traffic_flow:
            use_team = False

        respawn = bool(config.get("respawn_enabled", True))
        max_steps = int(config.get("max_steps", 2000))

        self.ego_routes = config.get("ego_routes", None)
        if self.ego_routes is None:
            self.ego_routes = self._default_routes(self.num_agents, self.num_lanes)

        self.lane_layout = build_lane_layout(self.num_lanes)
        self.points = self.lane_layout["points"]

        self.env = cpp_backend.IntersectionEnv(self.num_lanes)
        self.env.configure(use_team, respawn, max_steps)

        # traffic flow (single ego + NPC)
        self.traffic_density = float(config.get("traffic_density", 0.5))
        try:
            self.env.configure_traffic(self.traffic_flow, self.traffic_density)
            mapping = DEFAULT_ROUTE_MAPPING_2LANES if self.num_lanes == 2 else DEFAULT_ROUTE_MAPPING_3LANES
            routes = []
            for start, ends in mapping.items():
                for end in ends:
                    routes.append((start, end))
            self.env.configure_routes(routes)
        except Exception:
            pass

        reward_cfg = config.get("reward_config", None)
        if reward_cfg is None:
            reward_cfg = DEFAULT_REWARD_CONFIG.get("reward_config", {})
        if isinstance(reward_cfg, dict):
            _apply_reward_config(self.env, reward_cfg)

        self.cars: List[cpp_backend.Car] = []
        self.traffic_cars: List[cpp_backend.Car] = []

        self.reset()

    @staticmethod
    def _default_routes(num_agents: int, num_lanes: int):
        mapping = DEFAULT_ROUTE_MAPPING_2LANES if num_lanes == 2 else DEFAULT_ROUTE_MAPPING_3LANES
        all_routes = []
        for start, ends in mapping.items():
            for end in ends:
                all_routes.append((start, end))
        return [all_routes[i % len(all_routes)] for i in range(num_agents)]

    def reset(self):
        self.env.reset()
        for i in range(self.num_agents):
            start_id, end_id = self.ego_routes[i]
            self.env.add_car_with_route(start_id, end_id)
        self.cars = self.env.cars
        if self.traffic_flow:
            try:
                self.traffic_cars = list(self.env.traffic_cars)
            except Exception:
                self.traffic_cars = []
        obs = self._collect_obs()
        if self.traffic_flow:
            return obs[0], {}
        return obs, {}

    def _collect_obs(self) -> np.ndarray:
        obs = self.env.get_observations()
        return np.asarray(obs, dtype=np.float32)

    def step(self, actions: Union[np.ndarray, List[List[float]], List[float]], dt: float = 1.0 / 60.0):
        actions = np.asarray(actions, dtype=np.float32)

        # Accept both (2,) and (1,2) for single-agent manual control
        if self.traffic_flow:
            actions = actions.reshape(1, 2)
        else:
            if actions.ndim == 1:
                if actions.size == 2 and self.num_agents == 1:
                    actions = actions.reshape(1, 2)
                else:
                    raise ValueError(f"Expected actions shape (N,2) for multi-agent, got {actions.shape}")

        res = self.env.step(actions[:, 0].tolist(), actions[:, 1].tolist(), float(dt))

        if self.traffic_flow:
            try:
                self.traffic_cars = list(self.env.traffic_cars)
            except Exception:
                self.traffic_cars = []

        obs = np.asarray(res.obs, dtype=np.float32)
        rewards = np.asarray(res.rewards, dtype=np.float32)
        terminated = bool(res.terminated)
        truncated = bool(res.truncated)

        collisions = {int(res.agent_ids[i]): str(res.status[i]) for i in range(len(res.status))}

        info = {
            "step": int(res.step),
            "rewards": rewards.tolist() if not self.traffic_flow else float(rewards[0]) if len(rewards) else 0.0,
            "collisions": collisions,
            "agents_alive": int(getattr(res, "agents_alive", 0)),
            "terminated": terminated,
            "truncated": truncated,
            "done": list(res.done),
            "status": list(res.status),
        }

        if self.traffic_flow:
            return obs[0], float(rewards[0]) if len(rewards) else 0.0, terminated, truncated, info
        return obs, rewards, terminated, truncated, info

    def render(self, show_lane_ids: bool | None = None, show_lidar: bool | None = None):
        if self.render_mode != "human":
            return
        if show_lane_ids is None:
            show_lane_ids = self.show_lane_ids
        if show_lidar is None:
            show_lidar = self.show_lidar
        self.env.render(bool(show_lane_ids), bool(show_lidar))

    def close(self):
        # C++ side owns the GLFW window; nothing to close here.
        pass


if __name__ == "__main__":
    env = IntersectionEnv({"num_agents": 6, "num_lanes": 3, "render_mode": "human", "respawn_enabled": True})
    env.reset()
    for _ in range(200):
        act = np.zeros((env.num_agents, 2), dtype=np.float32)
        env.step(act)
        env.render()

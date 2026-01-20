# --- env.py ---
# Standard RL Environment Interface with Integrated Reward and Traffic Flow

import pygame
import numpy as np
import math
import random
import os
from typing import Dict, List, Tuple, Optional, Any
# Support both relative and absolute imports
try:
    from .config import *
    from .agent import Car, build_lane_layout
    from .utils import calculate_heading_rad
    from .sensor import Lidar
except ImportError:
    from config import *
    from agent import Car, build_lane_layout
    from utils import calculate_heading_rad
    from sensor import Lidar

# Default route mapping presets for specific lane counts
DEFAULT_ROUTE_MAPPING_2LANES = {
    "IN_1": ["OUT_3"],
    "IN_2": ["OUT_6"],
    "IN_3": ["OUT_5"],
    "IN_4": ["OUT_8"],
    "IN_6": ["OUT_2"],
    "IN_7": ["OUT_1"],
    "IN_8": ["OUT_4"],
}

DEFAULT_ROUTE_MAPPING_3LANES = {
    "IN_1": ["OUT_4"],
    "IN_2": ["OUT_8"],
    "IN_3": ["OUT_12"],
    "IN_4": ["OUT_7"],
    "IN_5": ["OUT_11"],
    "IN_6": ["OUT_3"],
    "IN_7": ["OUT_10"],
    "IN_8": ["OUT_2"],
    "IN_9": ["OUT_6"],
    "IN_10": ["OUT_1"],
    "IN_11": ["OUT_5"],
    "IN_12": ["OUT_9"],
}

class Road:
    def __init__(self, num_lanes=None, points=None):
        """
        Initialize Road.
        
        Args:
            num_lanes: Number of lanes per direction. If None, defaults to 3.
            points: Dictionary of lane points {lane_id: (x, y)}. If None, builds default (3 lanes).
        """
        self.num_lanes = num_lanes if num_lanes is not None else 3
        if points is not None:
            self.points = points
        else:
            # Build default points for 3 lanes as fallback
            from .agent import build_lane_layout
            try:
                from .agent import build_lane_layout
            except ImportError:
                from agent import build_lane_layout
            default_layout = build_lane_layout(3)
            self.points = default_layout['points']
        self.width = WIDTH
        self.height = HEIGHT
        self.cx = WIDTH // 2
        self.cy = HEIGHT // 2
        # Calculate road half width based on num_lanes
        self.rw = self.num_lanes * LANE_WIDTH_PX
        self.cr = CORNER_RADIUS
        
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.collision_mask = self._generate_collision_mask()
        self.line_mask = self._generate_line_mask()

    def _generate_collision_mask(self):
        """White = obstacle, black = road"""
        mask_surf = pygame.Surface((self.width, self.height))
        mask_surf.fill((255, 255, 255))
        
        # Cut out intersection
        pygame.draw.rect(mask_surf, 0, (self.cx - self.rw, 0, self.rw * 2, self.height))
        pygame.draw.rect(mask_surf, 0, (0, self.cy - self.rw, self.width, self.rw * 2))
        
        # Fill dead corners
        pygame.draw.rect(mask_surf, 0, (self.cx - self.rw - self.cr, self.cy - self.rw - self.cr, self.cr, self.cr))
        pygame.draw.rect(mask_surf, 0, (self.cx + self.rw,      self.cy - self.rw - self.cr, self.cr, self.cr))
        pygame.draw.rect(mask_surf, 0, (self.cx - self.rw - self.cr, self.cy + self.rw,      self.cr, self.cr))
        pygame.draw.rect(mask_surf, 0, (self.cx + self.rw,      self.cy + self.rw,      self.cr, self.cr))

        # Draw back rounded corners
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
        # Base road surface
        pygame.draw.rect(screen, COLOR_ROAD, (self.cx - self.rw, 0, self.rw * 2, self.height))
        pygame.draw.rect(screen, COLOR_ROAD, (0, self.cy - self.rw, self.width, self.rw * 2))
        
        # Rounded corner handling
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
        center_gap = 2

        # Double yellow center lines (vertical & horizontal)
        for dx in (-center_gap, center_gap):
            pygame.draw.line(screen, COLOR_YELLOW, (self.cx+dx, 0), (self.cx+dx, self.cy-stop_offset), 2)
            pygame.draw.line(screen, COLOR_YELLOW, (self.cx+dx, self.height), (self.cx+dx, self.cy+stop_offset), 2)
        for dy in (-center_gap, center_gap):
            pygame.draw.line(screen, COLOR_YELLOW, (0, self.cy+dy), (self.cx-stop_offset, self.cy+dy), 2)
            pygame.draw.line(screen, COLOR_YELLOW, (self.width, self.cy+dy), (self.cx+stop_offset, self.cy+dy), 2)

        # Stop lines
        stop_w = 4
        pygame.draw.line(screen, COLOR_WHITE, (self.cx-self.rw, self.cy-stop_offset), (self.cx, self.cy-stop_offset), stop_w)
        pygame.draw.line(screen, COLOR_WHITE, (self.cx, self.cy+stop_offset), (self.cx+self.rw, self.cy+stop_offset), stop_w)
        pygame.draw.line(screen, COLOR_WHITE, (self.cx-stop_offset, self.cy), (self.cx-stop_offset, self.cy+self.rw), stop_w)
        pygame.draw.line(screen, COLOR_WHITE, (self.cx+stop_offset, self.cy), (self.cx+stop_offset, self.cy-self.rw), stop_w)

        # Dashed lane separators for multi-lane roads (exclude center double yellow)
        for i in range(1, self.num_lanes):
            offset = i * LANE_WIDTH_PX
            # Vertical lanes (top and bottom segments)
            self._draw_dash(screen, (self.cx - offset, 0), (self.cx - offset, self.cy - stop_offset))
            self._draw_dash(screen, (self.cx + offset, 0), (self.cx + offset, self.cy - stop_offset))
            self._draw_dash(screen, (self.cx - offset, self.height), (self.cx - offset, self.cy + stop_offset))
            self._draw_dash(screen, (self.cx + offset, self.height), (self.cx + offset, self.cy + stop_offset))
            # Horizontal lanes (left and right segments)
            self._draw_dash(screen, (0, self.cy - offset), (self.cx - stop_offset, self.cy - offset))
            self._draw_dash(screen, (0, self.cy + offset), (self.cx - stop_offset, self.cy + offset))
            self._draw_dash(screen, (self.width, self.cy - offset), (self.cx + stop_offset, self.cy - offset))
            self._draw_dash(screen, (self.width, self.cy + offset), (self.cx + stop_offset, self.cy + offset))

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
        """Draw lane IDs at their spawn points."""
        COLOR_IN, COLOR_OUT = (0,0,200), (200,0,0)

        def label(t, x, y, c):
            s = self.font.render(t, True, (255,255,255))
            r = s.get_rect(center=(x,y))
            pygame.draw.rect(screen, c, r.inflate(10,6), border_radius=4)
            screen.blit(s, r)

        # Use self.points instead of global POINTS
        for lane_id, (x, y) in self.points.items():
            color = COLOR_IN if lane_id.startswith("IN") else COLOR_OUT
            label(lane_id, x, y, color)


class IntersectionEnv:
    """
    Standard RL Environment for Intersection Navigation.
    
    Supports both single-agent (with traffic flow) and multi-agent modes.
    Follows gym-style interface: reset(), step(), render()
    
    Integrated features:
    - Reward calculation (individual and team rewards)
    - Traffic flow generation (for single-agent mode)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize environment.
        
        :param config: Configuration dictionary
            - traffic_flow: bool, if True -> single agent with NPCs (traffic flow enabled)
            - num_agents: int, number of agents (if traffic_flow=False)
            - num_lanes: int, number of lanes per direction (if None, defaults to 3)
            - traffic_density: float, traffic density (0.0-1.0) for traffic flow
            - use_team_reward: bool, use team reward mixing (only for multi-agent)
            - reward_config: dict, reward configuration
            - render_mode: str, 'human' or None
            - ego_routes: list of (start, end) tuples for ego vehicles
            - max_steps: int, maximum steps per episode
            - respawn_enabled: bool, if True, agents will respawn at their starting position when they crash or disappear (within max_steps)
        """
        if config is None:
            config = {}
        
        # Environment configuration
        self.traffic_flow = config.get('traffic_flow', True)  # Default: single agent with traffic
        self.num_agents = config.get('num_agents', 1)
        self.traffic_density = config.get('traffic_density', 0.5)
        self.render_mode = config.get('render_mode', None)
        self.ego_routes = config.get('ego_routes', None)
        self.max_steps = config.get('max_steps', 2000)
        self.respawn_enabled = config.get('respawn_enabled', False)  # Enable respawn for agents
        # Fast headless mode: used for MCTS rollouts where we only need coarse
        # physics & collision, not pixel-accurate rendering.
        self.fast_mode = config.get('fast_mode', False)
        
        # Handle num_lanes configuration (if provided, build custom lane layout)
        self.num_lanes = config.get('num_lanes', None)
        if self.num_lanes is not None:
            # Build custom lane layout based on num_lanes
            self.lane_layout = build_lane_layout(self.num_lanes)
            self.points = self.lane_layout['points']
        else:
            # Use default (3 lanes) as fallback
            default_num_lanes = 3
            self.num_lanes = default_num_lanes
            self.lane_layout = build_lane_layout(default_num_lanes)
            self.points = self.lane_layout['points']
        
        # Initialize pygame (needed for Road class font initialization)
        self.pygame_initialized = False
        try:
            pygame.init()
            self.pygame_initialized = True
        except:
            pass  # pygame might already be initialized
        
        # Initialize pygame display if rendering (disabled in fast_mode)
        if self.render_mode == 'human' and not self.fast_mode:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(TITLE)
            
            # Set window icon (default: assets/icon.png or assets/icon.ico)
            icon_path = config.get('icon_path', None)
            if icon_path is None:
                # Try to find assets in multiple locations:
                # 1. In package directory (when installed as package)
                # 2. In project root (when running from source)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                
                # Try package directory first (for installed package)
                package_assets_dir = os.path.join(current_dir, 'assets')
                # Try project root (for development)
                project_assets_dir = os.path.join(project_root, 'assets')
                
                # Try default paths in both locations
                default_paths = [
                    os.path.join(package_assets_dir, 'icon.png'),
                    os.path.join(project_assets_dir, 'icon.png'),
                    os.path.join(package_assets_dir, 'icon.ico'),
                    os.path.join(project_assets_dir, 'icon.ico'),
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        icon_path = path
                        break
                
                # Debug: print path if not found (can be removed later)
                if not icon_path:
                    print(f"[DEBUG] Icon not found. Searched in: {package_assets_dir}, {project_assets_dir}")
                    print(f"[DEBUG] Current file: {__file__}")
                    print(f"[DEBUG] Project root: {project_root}")
            
            if icon_path and os.path.exists(icon_path):
                # Load icon from file
                try:
                    icon = pygame.image.load(icon_path)
                    pygame.display.set_icon(icon)
                    print(f"[DEBUG] Icon loaded successfully from: {icon_path}")
                except Exception as e:
                    # If loading fails, create default icon
                    print(f"[DEBUG] Failed to load icon: {e}")
                    self._set_default_icon()
            else:
                # Create default icon (simple intersection symbol)
                self._set_default_icon()
            
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)
        
        # Initialize road (requires pygame to be initialized)
        # Pass num_lanes and points to Road so it renders correctly
        road_num_lanes = self.num_lanes if self.num_lanes is not None else 3
        self.road = Road(num_lanes=road_num_lanes, points=self.points)
        # lane_layout is already set above based on num_lanes config
        self.route_mapping = self._build_route_mapping(config.get('route_mapping'))
        
        # Initialize reward configuration
        self.use_team_reward = config.get('use_team_reward', False)
        # If traffic flow exists, force individual reward only
        if self.traffic_flow:
            self.use_team_reward = False
        
        reward_config = config.get('reward_config', {})
        self._load_reward_config(reward_config)
        
        # Initialize traffic flow (only for single-agent mode)
        self.traffic_cars = pygame.sprite.Group()
        self.spawn_timer = 0
        self._init_traffic_routes()
        self.default_route = self._get_default_route()
        
        # Agent storage (use Group so alive() works correctly)
        self.agents: List[Car] = []
        self.agent_group = pygame.sprite.Group()
        
        # Episode tracking
        self.step_count = 0
        
        # Reward tracking state
        self.prev_positions = {}
        self.prev_actions = {}
        
        # Observation and action spaces
        self.observation_space = {
            'shape': (OBS_DIM,),
            'dtype': np.float32,
            'low': -np.inf,
            'high': np.inf
        }
        
        self.action_space = {
            'shape': (2,),  # [throttle, steer]
            'dtype': np.float32,
            'low': np.array([-1.0, -1.0]),
            'high': np.array([1.0, 1.0])
        }

    def _build_route_mapping(self, custom_mapping: Optional[Dict[str, List[str]]]):
        """
        Build a lane mapping dict (IN_x -> [OUT_y,...]).
        Allows custom overrides; falls back to presets per lane count.
        """
        if custom_mapping:
            def normalize_lane(value, prefix):
                if isinstance(value, str):
                    lane = value.strip().upper()
                    if lane.startswith('IN_') or lane.startswith('OUT_'):
                        return lane
                    if lane.startswith('IN') or lane.startswith('OUT'):
                        name = lane.replace('IN', 'IN_').replace('OUT', 'OUT_')
                        return name
                    if lane.isdigit():
                        return f"{prefix}{lane}"
                    digits = ''.join(ch for ch in lane if ch.isdigit())
                    if digits:
                        return f"{prefix}{digits}"
                    return f"{prefix}{lane}"
                return f"{prefix}{int(value)}"

            mapping = {}
            for raw_start, raw_targets in custom_mapping.items():
                start_id = normalize_lane(raw_start, 'IN_')
                target_list = raw_targets if isinstance(raw_targets, list) else [raw_targets]
                mapping[start_id] = [normalize_lane(t, 'OUT_') for t in target_list]
            return mapping

        # Fallback to presets based on num_lanes (use self.num_lanes if set, otherwise default to 3)
        num_lanes = self.num_lanes if self.num_lanes is not None else 3
        if num_lanes == 2:
            return DEFAULT_ROUTE_MAPPING_2LANES
        if num_lanes == 3:
            return DEFAULT_ROUTE_MAPPING_3LANES
        return {}
    
    def _init_traffic_routes(self):
        """Initialize traffic routes for NPC generation based on mapping."""
        layout = self.lane_layout
        dir_order = layout['dir_order']
        opposite = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
        left_turn = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}  # 90 deg left (driver view)

        routes: List[Tuple[str, str]] = []
        for direction in dir_order:
            for start_id in layout['in_by_dir'][direction]:
                mapped_targets = self.route_mapping.get(start_id, [])
                if mapped_targets:
                    for end_id in mapped_targets:
                        if end_id in self.points:
                            routes.append((start_id, end_id))
                    continue

                # Fallback: straight + left using lane indices
                straight_out_lanes = layout['out_by_dir'][opposite[direction]]
                left_out_lanes = layout['out_by_dir'][left_turn[direction]]
                idx = layout['idx_of'][start_id]

                if straight_out_lanes:
                    straight_out = straight_out_lanes[min(idx, len(straight_out_lanes) - 1)]
                    routes.append((start_id, straight_out))
                if left_out_lanes:
                    left_out = left_out_lanes[min(idx, len(left_out_lanes) - 1)]
                    routes.append((start_id, left_out))

        self.traffic_routes = routes

    def _get_default_route(self):
        """Pick a stable default ego route based on mapping preference."""
        if self.route_mapping:
            # use first mapping entry with valid targets
            for start_id, targets in self.route_mapping.items():
                for end_id in targets:
                    if end_id in self.points:
                        return (start_id, end_id)
        if self.traffic_routes:
            return self.traffic_routes[0]

        # Fallback to south-to-north straight if possible
        layout = self.lane_layout
        in_south = layout['in_by_dir']['S']
        out_north = layout['out_by_dir']['N']
        if in_south and out_north:
            idx = min(1, len(in_south) - 1)
            return (in_south[idx], out_north[min(idx, len(out_north) - 1)])

        return ('IN_1', 'OUT_1')
    
    def _load_reward_config(self, reward_config):
        """Load reward coefficients from config dictionary."""
        # Progress reward coefficient
        self.k_prog = reward_config.get('progress_scale', 1.0)
        
        # Stuck detection
        self.v_min = reward_config.get('stuck_speed_threshold', 1.0)  # m/s
        self.v_min_px = self.v_min * SCALE / FPS  # Convert m/s to px/frame
        self.k_stuck = reward_config.get('stuck_penalty', -0.1)
        
        # Crash penalties
        self.k_cv = reward_config.get('crash_vehicle_penalty', -10.0)
        self.k_co = reward_config.get('crash_object_penalty', -5.0)
        # Note: out_of_road_penalty is not used separately as CRASH_WALL already covers off-road cases
        
        # Success reward
        self.k_succ = reward_config.get('success_reward', 10.0)
        
        # Action smoothness
        self.k_sm = reward_config.get('action_smoothness_scale', -0.02)
        
        # Team mixing coefficient
        self.alpha = reward_config.get('team_alpha', 0.2)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        :param seed: Random seed (optional)
        :return: Initial observation and info dict
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Clear existing agents
        for agent in self.agents:
            agent.kill()
        self.agents.clear()
        self.agent_group.empty()
        
        # Reset traffic flow (if enabled)
        if self.traffic_flow:
            self.traffic_cars.empty()
            self.spawn_timer = 0
        
        # Reset reward tracking
        self.prev_positions.clear()
        self.prev_actions.clear()
        
        # Create agents
        if self.traffic_flow:
            # Single agent mode
            route = self.ego_routes[0] if self.ego_routes else self.default_route
            agent = Car(route[0], route[1], respawn_enabled=self.respawn_enabled,
                       points=self.points, lane_layout=self.lane_layout,
                       graphics_enabled=not self.fast_mode)
            agent.image_orig.fill((255, 0, 0))  # Red for ego
            pygame.draw.rect(agent.image_orig, (200,200,200), 
                           (agent.length*0.7, 2, agent.length*0.25, agent.width-4))
            # Update image and rect after modifying image_orig
            agent.image = pygame.transform.rotate(agent.image_orig, math.degrees(agent.heading))
            agent.rect = agent.image.get_rect(center=(int(agent.pos_x), int(agent.pos_y)))
            agent.mask = pygame.mask.from_surface(agent.image)
            self.agents.append(agent)
            self.agent_group.add(agent)
        else:
            # Multi-agent mode
            default_routes = [
                ('IN_6', 'OUT_2'),  # South to North
                ('IN_4', 'OUT_8'),  # East to West
                ('IN_5', 'OUT_7'),  # South to West
                ('IN_2', 'OUT_6'),  # North to South
            ]
            routes = self.ego_routes if self.ego_routes else default_routes[:self.num_agents]
            
            for i, route in enumerate(routes):
                agent = Car(route[0], route[1], respawn_enabled=self.respawn_enabled,
                           points=self.points, lane_layout=self.lane_layout,
                           graphics_enabled=not self.fast_mode)
                # Different colors for different agents
                color = COLOR_CAR_LIST[i % len(COLOR_CAR_LIST)]
                agent.image_orig.fill(color)
                pygame.draw.rect(agent.image_orig, (200,200,200),
                               (agent.length*0.7, 2, agent.length*0.25, agent.width-4))
                # Update image and rect after modifying image_orig
                agent.image = pygame.transform.rotate(agent.image_orig, math.degrees(agent.heading))
                agent.rect = agent.image.get_rect(center=(int(agent.pos_x), int(agent.pos_y)))
                agent.mask = pygame.mask.from_surface(agent.image)
                self.agents.append(agent)
                self.agent_group.add(agent)
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'step': self.step_count,
            'agents_alive': len([a for a in self.agents if a.alive()])
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        :param action: Action array [throttle, steer] for single agent,
                      or list of actions for multi-agent
        :return: (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        
        # Handle action format
        if self.traffic_flow:
            # Single agent: action is [throttle, steer]
            actions = [action]
        else:
            # Multi-agent: action is list of [throttle, steer]
            if isinstance(action, np.ndarray) and action.ndim == 1:
                # Single action for all agents (broadcast)
                actions = [action] * len(self.agents)
            else:
                actions = action
        
        # Update traffic flow (if single-agent mode)
        if self.traffic_flow:
            self._update_traffic_flow()
        
        # Update agents and collect collision info
        collision_dict = {}
        all_vehicles = list(self.agents)
        if self.traffic_flow:
            all_vehicles.extend(list(self.traffic_cars))
        
        for i, agent in enumerate(self.agents):
            if not agent.alive():
                continue
            
            # Update agent with action
            agent_action = actions[i] if i < len(actions) else [0.0, 0.0]
            agent.update(agent_action)
            
            if self.fast_mode:
                # Fast/headless mode: skip lidar casting & pixel-level masks,
                # use cheap geometric collision checks instead.
                if self.traffic_flow:
                    obstacles = list(self.traffic_cars)
                else:
                    obstacles = [a for a in self.agents if a is not agent]
                collision_info = self._fast_check_collision(agent, obstacles)
            else:
                # Full mode: update lidar and use precise mask-based collision
                agent.lidar.update(self.road.collision_mask, all_vehicles)
                
                if self.traffic_flow:
                    # Single agent: check against traffic
                    obstacles = list(self.traffic_cars)
                else:
                    # Multi-agent: check against other agents
                    obstacles = [a for a in self.agents if a is not agent]
                
                collision_info = agent.check_collision(
                    self.road.collision_mask,
                    self.road.line_mask,
                    obstacles
                )
            collision_dict[agent] = collision_info
        
        # Compute rewards
        rewards = []
        for agent in self.agents:
            if not agent.alive():
                rewards.append(0.0)
                continue
            
            collision_info = collision_dict.get(agent, (False, "ALIVE"))
            reward = self._compute_reward(
                agent,
                collision_info,
                collision_dict if not self.traffic_flow else None
            )
            rewards.append(reward)
        
        # Handle respawn for crashed agents (if enabled and within max_steps)
        # Respawn only for specific crash types: CRASH_CAR, CRASH_LINE, CRASH_WALL
        if self.respawn_enabled and (self.max_steps is None or self.step_count < self.max_steps):
            for i, agent in enumerate(self.agents):
                if not agent.respawn_enabled:
                    continue
                
                # Get collision status
                collision_info = collision_dict.get(agent, (False, "ALIVE"))
                has_collision, status = collision_info if isinstance(collision_info, tuple) and len(collision_info) >= 2 else (False, "ALIVE")
                
                # Only respawn for specific crash types: CRASH_CAR, CRASH_LINE, CRASH_WALL
                # Do NOT respawn for SUCCESS or ALIVE
                if has_collision and status in ["CRASH_CAR", "CRASH_LINE", "CRASH_WALL"]:
                    
                    # Respawn agent
                    agent.respawn_count += 1
                    # Reinitialize agent at original position
                    start_pos = self.points[agent.original_start_id]
                    agent.pos_x = float(start_pos[0])
                    agent.pos_y = float(start_pos[1])
                    agent.speed = 0.0
                    agent.acc = 0.0
                    agent.steering = 0.0
                    
                    # Reinitialize path and navigation
                    agent.intention = agent._determine_intention(agent.original_start_id, agent.original_end_id)
                    agent.path = agent._generate_path(agent.original_start_id, agent.original_end_id)
                    agent.path_index = 0
                    agent.end_x, agent.end_y = agent.path[-1]
                    
                    # Reset heading
                    if len(agent.path) > 1:
                        agent.heading = calculate_heading_rad(agent.path[0], agent.path[1])
                    else:
                        agent.heading = 0.0
                    
                    # Reset sprite
                    agent.rect.center = (int(agent.pos_x), int(agent.pos_y))
                    agent.image = pygame.transform.rotate(agent.image_orig, math.degrees(agent.heading))
                    agent.rect = agent.image.get_rect(center=agent.rect.center)
                    agent.mask = pygame.mask.from_surface(agent.image)
                    
                    # Reinitialize lidar
                    agent.lidar = Lidar(agent)
                    
                    # Ensure agent is in the group (in case it was removed)
                    if agent not in self.agent_group:
                        self.agent_group.add(agent)
                    
                    # Update collision dict to mark agent as alive after respawn
                    collision_dict[agent] = (False, "ALIVE")
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Check if all agents are done (only terminate if respawn is disabled or all agents succeeded)
        # Count alive agents and successful agents
        alive_agents = [agent for agent in self.agents if agent.alive()]
        successful_agents = []
        
        for agent in alive_agents:
            collision_info = collision_dict.get(agent, (False, "ALIVE"))
            if collision_info[0]:
                # If respawn is disabled, any collision terminates
                if not self.respawn_enabled:
                    terminated = True
                    break
                # If agent succeeded, count it
                elif collision_info[1] == "SUCCESS":
                    successful_agents.append(agent)
        
        # Terminate only if all alive agents have succeeded
        if self.respawn_enabled and len(successful_agents) > 0:
            if len(successful_agents) == len(alive_agents):
                terminated = True
        
        # Check step limit (skip if max_steps is None for continuous running)
        if self.max_steps is not None and self.step_count >= self.max_steps:
            truncated = True
        
        # Get observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            'step': self.step_count,
            'rewards': rewards,
            'collisions': {id(agent): collision_dict.get(agent, (False, "ALIVE"))[1] 
                          for agent in self.agents},
            'agents_alive': len([a for a in self.agents if a.alive()]),
            'terminated': terminated,
            'truncated': truncated
        }
        
        # Return format: (obs, reward, terminated, truncated, info)
        # For single agent, return scalar reward
        if self.traffic_flow:
            reward = rewards[0] if rewards else 0.0
            return obs, reward, terminated, truncated, info
        else:
            # Multi-agent: return list of rewards
            return obs, rewards, terminated, truncated, info
    
    def _update_traffic_flow(self):
        """Update traffic flow: spawn NPCs and update their behavior."""
        # Spawn logic
        arrival_rate = self.traffic_density 
        dt = 1.0 / FPS 

        spawn_prob = 1.0 - math.exp(-arrival_rate * dt)
        
        if random.random() < spawn_prob:
            self._try_spawn_traffic_car()
        
        # Update NPCs
        # NPCs should not consider ego vehicle in their planning
        all_vehicles_for_npc = list(self.traffic_cars)
        
        for car in list(self.traffic_cars):
            # Update lidar (include ego vehicle for sensor detection)
            all_vehicles_for_lidar = list(self.traffic_cars)
            if len(self.agents) > 0 and self.agents[0].alive():
                all_vehicles_for_lidar.append(self.agents[0])
            car.lidar.update(self.road.collision_mask, all_vehicles_for_lidar)
            
            # Autonomous driving (exclude ego vehicle from planning)
            action = car.plan_autonomous_action(self.road.collision_mask, all_vehicles_for_npc)
            car.update(action)
            
            # Remove if arrived
            if self._is_arrived(car):
                car.kill()
                continue
            
            # Remove if out of screen
            if self._is_out_of_screen(car):
                car.kill()
                continue
            
            # NPC collision: remove both cars
            for other in list(self.traffic_cars):
                if other is car:
                    continue
                if not car.alive() or not other.alive():
                    continue
                if car.rect.colliderect(other.rect):
                    off_x = other.rect.x - car.rect.x
                    off_y = other.rect.y - car.rect.y
                    if car.mask.overlap(other.mask, (off_x, off_y)):
                        car.kill()
                        other.kill()
                        break
    
    def _try_spawn_traffic_car(self):
        """Try to spawn a traffic NPC car."""
        route = random.choice(self.traffic_routes)
        start_node = route[0]
        
        # Check if spawn point is blocked
        sx, sy = self.points[start_node]
        
        is_blocked = False
        check_list = list(self.traffic_cars)
        if len(self.agents) > 0 and self.agents[0].alive():
            check_list.append(self.agents[0])
        
        for car in check_list:
            dist = math.hypot(car.pos_x - sx, car.pos_y - sy)
            if dist < CAR_LENGTH * 2.5:
                is_blocked = True
                break
        
        if not is_blocked:
            new_car = Car(route[0], route[1], points=self.points, lane_layout=self.lane_layout)
            # Mark as NPC color (gray)
            new_car.image_orig.fill((150, 150, 150))
            pygame.draw.rect(new_car.image_orig, (0,0,0), 
                           (new_car.length*0.7, 2, new_car.length*0.25, new_car.width-4))
            new_car.image = pygame.transform.rotate(new_car.image_orig, math.degrees(new_car.heading))
            self.traffic_cars.add(new_car)
    
    def _is_out_of_screen(self, car):
        """Check if car is out of screen bounds."""
        m = 100
        if car.pos_x < -m or car.pos_x > WIDTH+m or car.pos_y < -m or car.pos_y > HEIGHT+m:
            return True
        return False
    
    def _fast_check_collision(self, agent, obstacles):
        """
        Cheap geometric collision check for fast/headless mode.
        
        - SUCCESS: based on distance to route end.
        - CRASH_WALL: vehicle corners outside road area (checked via road.collision_mask).
        - CRASH_CAR: circle-based proximity between vehicle centers.
        
        Ignores lane-line collisions to avoid any pixel/Mask ops.
        """
        # 1) Success: close enough to target end point
        if hasattr(agent, 'end_x') and hasattr(agent, 'end_y'):
            dist_goal = math.hypot(agent.pos_x - agent.end_x, agent.pos_y - agent.end_y)
            if dist_goal < 20.0:
                return True, "SUCCESS"

        # 2) Wall / off-road: check vehicle corners against road.collision_mask
        # road_mask: white (255) = obstacle/grass, black (0) = road
        road_mask = getattr(self.road, 'collision_mask', None)
        if road_mask is not None and hasattr(agent, 'get_corners'):
            w, h = road_mask.get_size()
            for x, y in agent.get_corners():
                ix, iy = int(x), int(y)
                # Out of mask bounds = off-road
                if ix < 0 or ix >= w or iy < 0 or iy >= h:
                    return True, "CRASH_WALL"
                # White pixel in mask = obstacle (grass/wall)
                if road_mask.get_at((ix, iy)):
                    return True, "CRASH_WALL"
        else:
            # Fallback: simple screen bounds check if mask unavailable
            margin = 20
            if (agent.pos_x < -margin or agent.pos_x > WIDTH + margin or
                    agent.pos_y < -margin or agent.pos_y > HEIGHT + margin):
                return True, "CRASH_WALL"

        # 3) Vehicle-vehicle: circle approximation using car footprint
        radius = max(CAR_LENGTH, CAR_WIDTH) * 0.5
        for other in obstacles:
            dx = other.pos_x - agent.pos_x
            dy = other.pos_y - agent.pos_y
            if dx * dx + dy * dy < (radius * 2) ** 2:
                return True, "CRASH_CAR"

        return False, "ALIVE"
    
    def _is_arrived(self, car, tol=20):
        """Check if car has arrived at destination."""
        if hasattr(car, 'end_x'):
            d = math.hypot(car.pos_x - car.end_x, car.pos_y - car.end_y)
            if d < tol:
                return True
        return False
    
    def _compute_reward(self, vehicle, collision_info, vehicle_collision_dict=None):
        """
        Compute reward for a vehicle.
        Automatically selects individual or team reward based on config.
        """
        if self.use_team_reward and vehicle_collision_dict is not None:
            return self._compute_team_reward(vehicle, collision_info, vehicle_collision_dict)
        else:
            return self._compute_individual_reward(vehicle, collision_info)
    
    def _compute_individual_reward(self, vehicle, collision_info):
        """
        Compute individual reward for a vehicle.
        Equation (8): r_i^ind(t) = r_i^prog(t) + r_i^stuck(t) + r_i^crashV(t) + 
                      r_i^crashO(t) + r_i^succ(t) + r_i^smooth(t)
        Note: r_i^oor is included in r_i^crashO (CRASH_WALL covers off-road cases)
        """
        done, info = collision_info
        
        # 1. Progress reward
        r_prog = self._compute_progress_reward(vehicle)
        
        # 2. Stuck penalty
        r_stuck = self._compute_stuck_penalty(vehicle)
        
        # 3. Crash penalties
        r_crash_v = 0.0
        r_crash_o = 0.0
        
        if done:
            if info == "CRASH_CAR":
                r_crash_v = self.k_cv
            elif info == "CRASH_WALL" or info == "CRASH_LINE":
                r_crash_o = self.k_co
        
        # 4. Success reward
        r_succ = self.k_succ if (done and info == "SUCCESS") else 0.0
        
        # 5. Action smoothness
        r_smooth = self._compute_smoothness_reward(vehicle)
        
        # Sum all components
        r_ind = r_prog + r_stuck + r_crash_v + r_crash_o + r_succ + r_smooth
        
        return r_ind
    
    def _compute_team_reward(self, vehicle, collision_info, vehicle_collision_dict):
        """
        Compute team reward using reward mixing.
        Equation (9): r_i^mix(t) = (1 - α) * r_i^ind(t) + α * r̄^ind(t)
        """
        # Compute individual reward for this vehicle
        r_ind = self._compute_individual_reward(vehicle, collision_info)
        
        # Compute individual rewards for all agents
        individual_rewards = []
        for agent in self.agents:
            if agent is vehicle:
                individual_rewards.append(r_ind)
            else:
                other_collision_info = vehicle_collision_dict.get(agent, (False, "ALIVE"))
                r_other = self._compute_individual_reward(agent, other_collision_info)
                individual_rewards.append(r_other)
        
        # Average individual reward
        r_avg = np.mean(individual_rewards) if len(individual_rewards) > 0 else 0.0
        
        # Mixed reward
        r_mix = (1 - self.alpha) * r_ind + self.alpha * r_avg
        
        return r_mix
    
    def _compute_progress_reward(self, vehicle):
        """Compute progress reward based on distance reduction."""
        if not hasattr(vehicle, 'end_x') or not hasattr(vehicle, 'end_y'):
            return 0.0
        
        current_dist = np.hypot(vehicle.pos_x - vehicle.end_x, 
                               vehicle.pos_y - vehicle.end_y)
        
        vehicle_id = id(vehicle)
        if vehicle_id in self.prev_positions:
            prev_dist = self.prev_positions[vehicle_id]
            progress = prev_dist - current_dist
            max_progress = np.hypot(WIDTH, HEIGHT)
            normalized_progress = progress / max_progress if max_progress > 0 else 0
            r_prog = self.k_prog * normalized_progress
        else:
            r_prog = 0.0
        
        self.prev_positions[vehicle_id] = current_dist
        return r_prog
    
    def _compute_stuck_penalty(self, vehicle):
        """Compute stuck penalty if vehicle speed is too low."""
        speed_ms = (vehicle.speed * FPS) / SCALE  # px/frame -> m/s
        if speed_ms < self.v_min:
            return self.k_stuck
        return 0.0
    
    def _compute_smoothness_reward(self, vehicle):
        """Compute action smoothness reward."""
        vehicle_id = id(vehicle)
        
        current_action = np.array([
            vehicle.acc / MAX_ACC,
            vehicle.steering / MAX_STEERING_ANGLE
        ])
        
        if vehicle_id in self.prev_actions:
            prev_action = self.prev_actions[vehicle_id]
            action_diff = np.linalg.norm(current_action - prev_action)
            r_smooth = self.k_sm * (action_diff ** 2)
        else:
            r_smooth = 0.0
        
        self.prev_actions[vehicle_id] = current_action
        return r_smooth
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        :return: Observation array
        """
        if self.traffic_flow:
            # Single agent: return observation of first agent
            if len(self.agents) > 0 and self.agents[0].alive():
                all_vehicles = list(self.agents)
                all_vehicles.extend(list(self.traffic_cars))
                return self.agents[0].get_observation(all_vehicles)
            else:
                return np.zeros(OBS_DIM, dtype=np.float32)
        else:
            # Multi-agent: return stacked observations
            observations = []
            for agent in self.agents:
                if agent.alive():
                    obs = agent.get_observation(self.agents)
                    observations.append(obs)
                else:
                    observations.append(np.zeros(OBS_DIM, dtype=np.float32))
            
            # Stack into (num_agents, obs_dim) array
            return np.stack(observations)
    
    def render(self, show_lane_ids=False, show_lidar=True):
        """
        Render the environment.
        
        :param show_lane_ids: If True, render lane ID labels
        :param show_lidar: If True, render lidar rays
        """
        if self.render_mode != 'human' or not self.pygame_initialized:
            return
        
        # Check if pygame is actually initialized (may have been closed by another process)
        if not pygame.get_init():
            # Try to reinitialize
            try:
                pygame.init()
                if self.render_mode == 'human':
                    self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                    pygame.display.set_caption(TITLE)
                    self.pygame_initialized = True
                else:
                    return
            except Exception as e:
                # If reinitialization fails, skip rendering
                return
        
        # Handle events
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.pygame_initialized = False
                    return
        except pygame.error as e:
            # If pygame is not initialized, skip event handling
            if "not initialized" in str(e).lower():
                self.pygame_initialized = False
                return
            raise
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw road with optional lane IDs
        self.road.draw(self.screen, show_lane_ids=show_lane_ids)
        
        # Draw traffic (if single-agent mode)
        if self.traffic_flow:
            self.traffic_cars.draw(self.screen)
        
        # Draw agents
        for agent in self.agents:
            if agent.alive():
                self.screen.blit(agent.image, agent.rect)
                # Draw lidar if enabled
                if show_lidar and self.render_mode == 'human':
                    agent.lidar.draw(self.screen)
        
        # Draw UI
        if self.pygame_initialized:
            info_text = f"Step: {self.step_count} | Agents: {len([a for a in self.agents if a.alive()])}"
            if self.traffic_flow:
                info_text += f" | Traffic: {len(self.traffic_cars)}"
            if len(self.agents) > 0 and self.agents[0].alive():
                speed_ms = (self.agents[0].speed * FPS) / SCALE
                info_text += f" | Speed: {speed_ms:.1f} m/s"
            txt = self.font.render(info_text, True, (255, 255, 255))
            self.screen.blit(txt, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def _render_with_lane_ids(self):
        """Convenience method: render with lane IDs and lidar enabled."""
        self.render(show_lane_ids=True, show_lidar=True)
    
    def _set_default_icon(self):
        """Create and set a default icon (simple intersection symbol)."""
        # Create a 32x32 icon surface
        icon_size = 32
        icon = pygame.Surface((icon_size, icon_size))
        icon.fill((34, 139, 34))  # Grass color background
        
        # Draw a simple intersection (cross shape)
        road_color = (60, 60, 60)  # Road gray
        road_width = 8
        
        # Horizontal road
        pygame.draw.rect(icon, road_color, 
                        (0, icon_size//2 - road_width//2, icon_size, road_width))
        # Vertical road
        pygame.draw.rect(icon, road_color, 
                        (icon_size//2 - road_width//2, 0, road_width, icon_size))
        
        # Draw a small car in the center
        car_color = (255, 0, 0)  # Red car
        car_size = 6
        pygame.draw.rect(icon, car_color,
                        (icon_size//2 - car_size//2, icon_size//2 - car_size//2,
                         car_size, car_size))
        
        pygame.display.set_icon(icon)
    
    def close(self):
        """
        Close the environment and cleanup resources.
        """
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()

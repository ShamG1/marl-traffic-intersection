# --- env.py ---
# Standard RL Environment Interface with Integrated Reward and Traffic Flow

import pygame
import numpy as np
import math
import random
import os
from typing import Dict, List, Tuple, Optional, Any
from config import *
from agent import Car, POINTS

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
        # Double yellow lines
        pygame.draw.line(screen, COLOR_YELLOW, (self.cx-2, 0), (self.cx-2, self.cy-stop_offset), 2)
        pygame.draw.line(screen, COLOR_YELLOW, (self.cx+2, 0), (self.cx+2, self.cy-stop_offset), 2)
        pygame.draw.line(screen, COLOR_YELLOW, (self.cx-2, self.height), (self.cx-2, self.cy+stop_offset), 2)
        pygame.draw.line(screen, COLOR_YELLOW, (self.cx+2, self.height), (self.cx+2, self.cy+stop_offset), 2)
        
        pygame.draw.line(screen, COLOR_YELLOW, (0, self.cy-2), (self.cx-stop_offset, self.cy-2), 2)
        pygame.draw.line(screen, COLOR_YELLOW, (0, self.cy+2), (self.cx-stop_offset, self.cy+2), 2)
        pygame.draw.line(screen, COLOR_YELLOW, (self.width, self.cy-2), (self.cx+stop_offset, self.cy-2), 2)
        pygame.draw.line(screen, COLOR_YELLOW, (self.width, self.cy+2), (self.cx+stop_offset, self.cy+2), 2)

        # Stop lines
        stop_w = 4
        pygame.draw.line(screen, COLOR_WHITE, (self.cx-self.rw, self.cy-stop_offset), (self.cx, self.cy-stop_offset), stop_w)
        pygame.draw.line(screen, COLOR_WHITE, (self.cx, self.cy+stop_offset), (self.cx+self.rw, self.cy+stop_offset), stop_w)
        pygame.draw.line(screen, COLOR_WHITE, (self.cx-stop_offset, self.cy), (self.cx-stop_offset, self.cy+self.rw), stop_w)
        pygame.draw.line(screen, COLOR_WHITE, (self.cx+stop_offset, self.cy), (self.cx+stop_offset, self.cy-self.rw), stop_w)

        # Dashed lines
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
            - traffic_density: float, traffic density (0.0-1.0) for traffic flow
            - use_team_reward: bool, use team reward mixing (only for multi-agent)
            - reward_config: dict, reward configuration
            - render_mode: str, 'human' or None
            - ego_routes: list of (start, end) tuples for ego vehicles
            - max_steps: int, maximum steps per episode
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
        
        # Initialize pygame (needed for Road class font initialization)
        self.pygame_initialized = False
        try:
            pygame.init()
            self.pygame_initialized = True
        except:
            pass  # pygame might already be initialized
        
        # Initialize pygame display if rendering
        if self.render_mode == 'human':
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(TITLE)
            
            # Set window icon (default: assets/icon.png or assets/icon.ico)
            icon_path = config.get('icon_path', None)
            if icon_path is None:
                # Try default paths in assets directory
                default_paths = ['assets/icon.png', 'assets/icon.ico']
                for path in default_paths:
                    if os.path.exists(path):
                        icon_path = path
                        break
            
            if icon_path and os.path.exists(icon_path):
                # Load icon from file
                try:
                    icon = pygame.image.load(icon_path)
                    pygame.display.set_icon(icon)
                except:
                    # If loading fails, create default icon
                    self._set_default_icon()
            else:
                # Create default icon (simple intersection symbol)
                self._set_default_icon()
            
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)
        
        # Initialize road (requires pygame to be initialized)
        self.road = Road()
        
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
    
    def _init_traffic_routes(self):
        """Initialize traffic routes for NPC generation."""
        self.traffic_routes = [
            ('IN_6', 'OUT_2'), # South straight
            ('IN_5', 'OUT_7'), # South left turn
            ('IN_4', 'OUT_8'), # East straight
            ('IN_3', 'OUT_5'), # East left turn
            ('IN_2', 'OUT_6'), # North straight
            ('IN_1', 'OUT_3'), # North left turn
            ('IN_8', 'OUT_4'), # West straight
            ('IN_7', 'OUT_1')  # West left turn
        ]
    
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
            route = self.ego_routes[0] if self.ego_routes else ('IN_6', 'OUT_2')
            agent = Car(route[0], route[1])
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
                agent = Car(route[0], route[1])
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
            
            # Update lidar
            agent.lidar.update(self.road.collision_mask, all_vehicles)
            
            # Check collision
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
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Check if any agent is done
        for agent in self.agents:
            if agent.alive():
                collision_info = collision_dict.get(agent, (False, "ALIVE"))
                if collision_info[0]:
                    terminated = True
                    break
        
        # Check step limit
        if self.step_count >= self.max_steps:
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
        self.spawn_timer += 1
        spawn_interval = int(90 / (self.traffic_density + 0.01))
        
        if self.spawn_timer >= spawn_interval:
            self._try_spawn_traffic_car()
            self.spawn_timer = 0
        
        # Update NPCs
        all_vehicles = list(self.traffic_cars)
        if len(self.agents) > 0 and self.agents[0].alive():
            all_vehicles.append(self.agents[0])
        
        for car in list(self.traffic_cars):
            # Update lidar
            car.lidar.update(self.road.collision_mask, all_vehicles)
            
            # Autonomous driving
            action = car.plan_autonomous_action(self.road.collision_mask, all_vehicles)
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
        sx, sy = POINTS[start_node]
        
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
            new_car = Car(route[0], route[1])
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
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
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

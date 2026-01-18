# --- train.py ---
# Multi-Agent MCTS Training Script

import os
import sys

# Suppress pygame messages in worker processes (set before any pygame imports)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from collections import deque
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Intersection.env import IntersectionEnv
from Intersection.config import DEFAULT_REWARD_CONFIG, OBS_DIM

# Import MCTS modules

# Global function for process pool (must be at module level for pickling)
def _search_agent_wrapper(args):
    """
    Wrapper function for process pool MCTS search.
    Must be at module level to be picklable.
    """
    agent_id, obs_data, obs_history_data, hidden_state_data, env_state_data, config = args
    
    # Suppress pygame initialization messages in worker processes
    import os
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    
    # Recreate MCTS instance in this process
    # Note: Rollout debug output is disabled - only episode-level logs are shown
    
    try:
        # Import necessary modules
        from MCTS.mcts import MCTS
        from MCTS.networks import DualNetwork
        from Intersection.config import OBS_DIM
        from Intersection.env import IntersectionEnv
        from Intersection.config import DEFAULT_REWARD_CONFIG
        
        # Recreate network from state dict
        import torch
        device = torch.device(config['device'])
        network = DualNetwork(
            obs_dim=OBS_DIM,
            action_dim=2,
            hidden_dim=config['hidden_dim'],
            lstm_hidden_dim=config['lstm_hidden_dim'],
            use_lstm=config['use_lstm'],
            sequence_length=config['sequence_length']
        ).to(device)
        network.load_state_dict(config['network_state_dict'])
        network.eval()
        
        # Recreate env_factory in this process
        def create_env_copy_local():
            """Create environment copy in this process."""
            env_config_local = {
                'traffic_flow': config['env_config'].get('traffic_flow', False),
                'num_agents': config['env_config']['num_agents'],
                'num_lanes': config['env_config']['num_lanes'],
                'use_team_reward': config['env_config']['use_team_reward'],
                'render_mode': None,
                'max_steps': config['env_config']['max_steps'],
                'respawn_enabled': config['env_config']['respawn_enabled'],
                'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
            }
            
            if env_config_local['traffic_flow']:
                env_config_local['traffic_density'] = config['env_config'].get('traffic_density', 0.5)
            else:
                # Generate ego routes inline (simplified version for worker process)
                from Intersection.env import DEFAULT_ROUTE_MAPPING_2LANES, DEFAULT_ROUTE_MAPPING_3LANES
                num_agents_local = config['env_config']['num_agents']
                num_lanes_local = config['env_config']['num_lanes']
                
                if num_lanes_local == 2:
                    route_mapping = DEFAULT_ROUTE_MAPPING_2LANES
                elif num_lanes_local == 3:
                    route_mapping = DEFAULT_ROUTE_MAPPING_3LANES
                else:
                    route_mapping = DEFAULT_ROUTE_MAPPING_2LANES
                
                all_routes = []
                for start, ends in route_mapping.items():
                    for end in ends:
                        all_routes.append((start, end))
                
                # Simple selection: take first num_agents routes
                ego_routes_copy = all_routes[:num_agents_local]
                env_config_local['ego_routes'] = ego_routes_copy
            
            return IntersectionEnv(env_config_local)
        
        # Recreate MCTS instance
        mcts = MCTS(
            network=network,
            action_space=None,
            num_simulations=config['num_simulations'],
            c_puct=config['c_puct'],
            temperature=config['temperature'],
            device=device,
            rollout_depth=config['rollout_depth'],
            env_factory=create_env_copy_local,
            all_networks=None,  # Will be recreated in rollout if needed
            agent_id=agent_id,
            num_action_samples=config['num_action_samples']
        )
        
        # Convert hidden state back to tensor if needed
        hidden_state_tensor = None
        if hidden_state_data is not None:
            if isinstance(hidden_state_data, tuple):
                hidden_state_tensor = tuple(torch.tensor(h, device=device) for h in hidden_state_data)
            else:
                hidden_state_tensor = torch.tensor(hidden_state_data, device=device)
        
        # Perform search
        action, search_stats = mcts.search(
            root_state=obs_data,
            obs_history=obs_history_data,
            hidden_state=hidden_state_tensor,
            env_state=env_state_data
        )
        
        # MCTS search completed (no output to reduce verbosity)
        
        # Ensure action is numpy array with exactly 2 elements
        import numpy as np
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        action = action.flatten()
        
        # Ensure action has exactly 2 elements
        if action.size != 2:
            if action.size == 1:
                action = np.array([action[0], 0.0])
            elif action.size == 0:
                action = np.zeros(2)
            else:
                action = action[:2]
        
        return agent_id, action
    except Exception as e:
        # Only log errors, not detailed traceback (to reduce verbosity)
        import sys
        import traceback
        # Only output critical errors
        sys.stdout.write(f"[Process {agent_id}] Error: {e}\n")
        sys.stdout.flush()
        import numpy as np
        return agent_id, np.zeros(2)
try:
    from .networks import DualNetwork
    from .mcts import MCTS
except ImportError:
    from networks import DualNetwork
    from mcts import MCTS


def generate_ego_routes(num_agents: int, num_lanes: int):
    """Generate routes for agents based on num_agents and num_lanes."""
    # Import route mappings from env
    from Intersection.env import DEFAULT_ROUTE_MAPPING_2LANES, DEFAULT_ROUTE_MAPPING_3LANES
    
    if num_lanes == 2:
        route_mapping = DEFAULT_ROUTE_MAPPING_2LANES
    elif num_lanes == 3:
        route_mapping = DEFAULT_ROUTE_MAPPING_3LANES
    else:
        # Fallback to 2 lanes
        route_mapping = DEFAULT_ROUTE_MAPPING_2LANES
    
    # Get all available routes
    all_routes = []
    for start, ends in route_mapping.items():
        for end in ends:
            all_routes.append((start, end))
    
    # Select routes for agents (balanced distribution, ensuring uniqueness)
    selected_routes = []
    agents_per_dir = num_agents // 4
    extra_agents = num_agents % 4
    
    # Track used routes to avoid duplicates
    used_routes = set()
    
    for i in range(4):
        count = agents_per_dir + (1 if i < extra_agents else 0)
        # Select routes for this direction
        # For 3 lanes: direction 0 uses IN_1, IN_2, IN_3; direction 1 uses IN_4, IN_5, IN_6; etc.
        if num_lanes == 3:
            start_idx = i * num_lanes + 1
            dir_routes = [r for r in all_routes 
                         if any(r[0].startswith(f'IN_{j}') for j in range(start_idx, start_idx + num_lanes))]
        else:  # 2 lanes
            start_idx = i * num_lanes + 1
            dir_routes = [r for r in all_routes 
                         if any(r[0].startswith(f'IN_{j}') for j in range(start_idx, start_idx + num_lanes))]
        
        if len(dir_routes) == 0:
            # Fallback: use all routes
            dir_routes = all_routes
        
        # Filter out already used routes
        available_routes = [r for r in dir_routes if r not in used_routes]
        if len(available_routes) == 0:
            # If all routes in this direction are used, allow reuse but try to avoid exact duplicates
            available_routes = dir_routes
        
        # Select routes for agents in this direction
        dir_route_idx = 0
        for _ in range(count):
            if available_routes:
                # Cycle through available routes
                route = available_routes[dir_route_idx % len(available_routes)]
                selected_routes.append(route)
                used_routes.add(route)
                dir_route_idx += 1
            elif dir_routes:
                # Fallback: use any route from this direction
                route = dir_routes[dir_route_idx % len(dir_routes)]
                selected_routes.append(route)
                dir_route_idx += 1
    
    # If we need more routes, use remaining available routes
    remaining_routes = [r for r in all_routes if r not in used_routes]
    while len(selected_routes) < num_agents:
        if remaining_routes:
            route = remaining_routes.pop(0)
            selected_routes.append(route)
            used_routes.add(route)
        else:
            # If all routes are used, cycle through all routes
            route = all_routes[(len(selected_routes) - len(used_routes)) % len(all_routes)]
            selected_routes.append(route)
    
    return selected_routes[:num_agents]


class MCTSTrainer:
    """Multi-Agent MCTS Trainer."""
    
    def __init__(
        self,
        num_agents: int = 1,  # Single agent for traffic flow mode
        num_lanes: int = 3,
        max_episodes: int = 10000,
        max_steps_per_episode: int = 2000,
        mcts_simulations: int = 50,
        rollout_depth: int = 3,
        num_action_samples: int = 5,
        save_frequency: int = 100,
        log_frequency: int = 10,
        device: str = 'cpu',
        use_team_reward: bool = False,  # Not used in traffic flow mode
        render: bool = False,
        show_lane_ids: bool = False,
        show_lidar: bool = False,
        respawn_enabled: bool = False,  # Default disabled for single-agent traffic flow mode
        save_dir: str = 'MCTS_SINGLE/checkpoints',
        parallel_mcts: bool = False,  # Not needed for single agent
        max_workers: int = None,  # Number of processes for parallel MCTS (None = num_agents)
        traffic_flow: bool = True,  # Enable traffic flow mode (single agent with NPCs)
        traffic_density: float = 0.5  # Traffic density (0.0-1.0)
    ):
        """
        Initialize MCTS Trainer for single-agent traffic flow mode.
        
        Args:
            num_agents: Number of agents (should be 1 for traffic flow mode)
            num_lanes: Number of lanes per direction
            max_episodes: Maximum training episodes
            mcts_simulations: Number of MCTS simulations per step
            save_frequency: Episodes before saving checkpoint
            log_frequency: Episodes before logging stats
            device: Device to use ('cpu' or 'cuda')
            use_team_reward: Whether to use team reward (not used in traffic flow mode)
            render: Whether to render environment
            show_lane_ids: Whether to show lane IDs in render
            show_lidar: Whether to show lidar in render
            respawn_enabled: Whether to enable respawn
            save_dir: Directory to save checkpoints
            parallel_mcts: Whether to use parallel MCTS (not needed for single agent)
            max_workers: Number of processes for parallel MCTS
            traffic_flow: Enable traffic flow mode (single agent with NPCs)
            traffic_density: Traffic density (0.0-1.0)
        """
        # Force single agent for traffic flow mode
        if traffic_flow:
            num_agents = 1
            use_team_reward = False  # Traffic flow mode doesn't support team reward
        
        self.num_agents = num_agents
        self.num_lanes = num_lanes
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.mcts_simulations = mcts_simulations
        self.rollout_depth = rollout_depth
        self.num_action_samples = num_action_samples
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.device = torch.device(device)
        self.use_team_reward = use_team_reward
        self.render = render
        self.show_lane_ids = show_lane_ids
        self.show_lidar = show_lidar
        self.respawn_enabled = respawn_enabled
        self.parallel_mcts = parallel_mcts and num_agents > 1  # Only parallel if multiple agents
        self.max_workers = max_workers if max_workers is not None else num_agents
        self._process_pool = None  # Initialize process pool to None
        self.traffic_flow = traffic_flow
        self.traffic_density = traffic_density
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        
        # Generate routes (only needed if not using traffic flow)
        if not traffic_flow:
            ego_routes = generate_ego_routes(num_agents, num_lanes)
            # Debug: Check for duplicate routes
            route_counts = {}
            for route in ego_routes:
                route_counts[route] = route_counts.get(route, 0) + 1
            duplicates = {r: c for r, c in route_counts.items() if c > 1}
            if duplicates:
                print(f"WARNING: Found duplicate routes: {duplicates}")
                print(f"All routes: {ego_routes}")
        else:
            ego_routes = None  # Traffic flow mode doesn't need ego_routes
        
        # Store routes for debugging
        self.ego_routes = ego_routes
        
        # Initialize environment
        env_config = {
            'traffic_flow': traffic_flow,
            'num_agents': num_agents,
            'num_lanes': num_lanes,
            'use_team_reward': use_team_reward,
            'render_mode': 'human' if render else None,
            'max_steps': max_steps_per_episode,
            'respawn_enabled': respawn_enabled,
            'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
        }
        
        if traffic_flow:
            env_config['traffic_density'] = traffic_density
        else:
            env_config['ego_routes'] = ego_routes
        
        self.env = IntersectionEnv(env_config)
        
        # Initialize dual networks for each agent
        # Each network is initialized independently, so they will have different random weights
        self.networks = torch.nn.ModuleList([
            DualNetwork(
                obs_dim=OBS_DIM,
                action_dim=2,
                hidden_dim=256,
                lstm_hidden_dim=128,
                use_lstm=True,
                sequence_length=5
            ).to(self.device)
            for _ in range(num_agents)
        ])
        
        # Verify networks are different (for debugging)
        if num_agents > 1:
            # Check if first layer weights are different
            first_net_first_weight = list(self.networks[0].parameters())[0].data[0, 0].item()
            second_net_first_weight = list(self.networks[1].parameters())[0].data[0, 0].item()
            if abs(first_net_first_weight - second_net_first_weight) < 1e-6:
                print(f"WARNING: Networks may have identical initialization!")
            else:
                print(f"Networks initialized with different weights (verified)")
        
        # Create environment factory function for MCTS rollouts
        # Note: With process pool, each process will create its own environment copy
        # No need for locks since processes don't share memory
        def create_env_copy():
            """Create a copy of the environment for MCTS rollouts."""
            try:
                # Create a new environment with same configuration
                rollout_config = {
                    'traffic_flow': traffic_flow,
                    'num_agents': num_agents,
                    'num_lanes': num_lanes,
                    'use_team_reward': use_team_reward,
                    'render_mode': None,  # Rollouts run in background without rendering
                    'max_steps': max_steps_per_episode,
                    'respawn_enabled': respawn_enabled,
                    'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
                }
                
                if traffic_flow:
                    rollout_config['traffic_density'] = traffic_density
                else:
                    ego_routes_copy = generate_ego_routes(num_agents, num_lanes)
                    rollout_config['ego_routes'] = ego_routes_copy
                
                env_copy = IntersectionEnv(rollout_config)
                return env_copy
            except Exception as e:
                import sys
                sys.stdout.write(f"Error creating env copy: {e}\n")
                sys.stdout.flush()
                import traceback
                traceback.print_exc()
                raise
        
        # Initialize MCTS for each agent with real environment rollouts
        # Note: Rollouts run in background without rendering
        self.mcts_instances = [
            MCTS(
                network=network,
                action_space=None,  # Continuous actions
                num_simulations=mcts_simulations,
                c_puct=1.0,
                temperature=1.0,
                device=device,
                rollout_depth=self.rollout_depth,
                num_action_samples=self.num_action_samples,
                env_factory=create_env_copy,
                all_networks=self.networks,  # All networks for multi-agent rollouts
                agent_id=i  # Agent ID
            )
            for i, network in enumerate(self.networks)
        ]
        
        # Enable rollout debugging for first few episodes
        self.enable_rollout_debug = False  # Set to True to enable detailed rollout logs
        self.enable_debug = False  # Set to True to enable all debug output
        
        # Process pool for parallel MCTS (created on first use)
        self._process_pool = None
        
        # Optimizers for each network
        self.optimizers = [
            optim.Adam(network.parameters(), lr=3e-4)
            for network in self.networks
        ]
        
        # Observation history for LSTM
        self.obs_history = [deque(maxlen=5) for _ in range(num_agents)]
        self.hidden_states = [None for _ in range(num_agents)]
        
        # Training statistics
        self.stats = {
            'episode': 0,
            'total_steps': 0,
            'episode_rewards': [],
            'episode_lengths': [],
            'success_count': 0,
            'crash_count': 0,
        }
        
        # Log file
        self.log_file = os.path.join(save_dir, 'training_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"Traffic flow mode: {traffic_flow}\n")
            f.write(f"Num agents: {num_agents}\n")
            f.write(f"Num lanes: {num_lanes}\n")
            f.write(f"Use team reward: {use_team_reward}\n")
            f.write(f"Respawn enabled: {respawn_enabled}\n")
            f.write(f"MCTS simulations: {mcts_simulations}\n")
            if traffic_flow:
                f.write(f"Traffic density: {traffic_density}\n")
            else:
                f.write("Generated routes:\n")
                for i, route in enumerate(ego_routes):
                    f.write(f"  Agent {i}: {route[0]} -> {route[1]}\n")
            f.write("=" * 80 + "\n")
        
        # CSV file for episode rewards
        self.csv_file = os.path.join(save_dir, 'episode_rewards.csv')
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Total_Reward', 'Mean_Reward', 'Episode_Length'])
    
    def log(self, message: str):
        """Log message to file and console."""
        print(message)
        # Use UTF-8 encoding to handle Unicode characters
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
    
    def train(self):
        """Main training loop."""
        self.log("=" * 80)
        self.log("Starting Multi-Agent MCTS Training")
        self.log("=" * 80)
        
        episode = 0
        step_count = 0
        
        while episode < self.max_episodes:
            episode += 1
            self.stats['episode'] = episode
            
            try:
                # Reset environment
                self.log(f"Resetting environment for episode {episode}...")
                obs, info = self.env.reset()
                self.log(f"Environment reset successful. Starting episode {episode}...")
            except Exception as e:
                self.log(f"Error resetting environment at episode {episode}: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # Debug: Check initial agent positions for first few episodes
            if self.enable_debug and episode <= 3:
                if hasattr(self.env, 'agents'):
                    print(f"\n[Episode {episode}] Initial agent positions:")
                    positions = {}
                    for i, agent in enumerate(self.env.agents):
                        pos_key = (round(agent.pos_x, 1), round(agent.pos_y, 1))
                        if pos_key not in positions:
                            positions[pos_key] = []
                        positions[pos_key].append(i)
                        route_info = self.ego_routes[i] if i < len(self.ego_routes) else 'N/A'
                        print(f"  Agent {i}: pos=({agent.pos_x:.1f}, {agent.pos_y:.1f}), heading={agent.heading:.2f}, route={route_info}")
                    # Check for overlapping positions
                    overlaps = {pos: agents for pos, agents in positions.items() if len(agents) > 1}
                    if overlaps:
                        print(f"  WARNING: Agents with overlapping positions: {overlaps}")
            
            # Reset observation history and hidden states
            for i in range(self.num_agents):
                self.obs_history[i].clear()
                self.hidden_states[i] = None
                # Initialize history with first observation
                self.obs_history[i].append(obs[i])
            
            # Reset buffer at start of episode
            if hasattr(self, 'buffer'):
                self.buffer = {
                    'obs': [[] for _ in range(self.num_agents)],
                    'actions': [[] for _ in range(self.num_agents)],
                    'rewards': [[] for _ in range(self.num_agents)],
                    'values': [[] for _ in range(self.num_agents)],
                    'dones': [[] for _ in range(self.num_agents)],
                }
            
            episode_reward = np.zeros(self.num_agents)
            episode_length = 0
            done = False
            
            self.log(f"Starting episode {episode} loop (max_steps: {self.max_steps_per_episode})...")
            
            # Create progress bar for this episode
            pbar = tqdm(
                total=self.max_steps_per_episode,
                desc=f"Episode {episode}",
                unit="step",
                leave=False,  # Don't leave progress bar after completion
                ncols=100,  # Width of progress bar
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            # Episode loop
            while not done and episode_length < self.max_steps_per_episode:
                if episode_length == 0:
                    self.log(f"Episode {episode}, Step 0: Starting MCTS search...")
                # Render current state before MCTS search (so user can see what's happening)
                if self.render:
                    try:
                        self.env.render(show_lane_ids=self.show_lane_ids, show_lidar=self.show_lidar)
                        # Process events to keep window responsive
                        import pygame
                        if pygame.get_init():
                            pygame.event.pump()
                    except Exception:
                        pass
                
                # Select actions using MCTS for each agent
                # Prepare environment state for rollouts (shared by all agents)
                # Only include serializable data (no pygame objects)
                env_state = {
                    'step_count': getattr(self.env, 'step_count', 0),
                    'obs': obs.copy() if isinstance(obs, np.ndarray) else obs,  # Current observations for all agents
                    # Extract only serializable agent data (no pygame surfaces)
                    'agent_states': []
                }
                if hasattr(self.env, 'agents') and self.env.agents:
                    for agent in self.env.agents:
                        # Extract only basic attributes (no pygame objects)
                        # Use getattr with safe defaults to avoid AttributeError
                        agent_state = {
                            'pos_x': getattr(agent, 'pos_x', 0.0),
                            'pos_y': getattr(agent, 'pos_y', 0.0),
                            'heading': getattr(agent, 'heading', 0.0),
                            'speed': getattr(agent, 'speed', 0.0),  # Car uses 'speed', not 'velocity'
                            'target_lane': getattr(agent, 'target_lane', None),
                            'route': getattr(agent, 'route', None)
                        }
                        env_state['agent_states'].append(agent_state)
                
                # Update observation history for all agents
                for i in range(self.num_agents):
                    if episode_length > 0:
                        self.obs_history[i].append(obs[i])
                
                # Enable debug output for first episode to diagnose hanging issue
                for i in range(self.num_agents):
                    # Enable debug for first episode only
                    self.mcts_instances[i].debug_rollout = (episode == 1 and episode_length == 0)
                
                # Handle pygame events before MCTS search
                if self.render:
                    try:
                        import pygame
                        if pygame.get_init() and hasattr(self.env, 'screen') and self.env.screen is not None:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    return
                    except Exception:
                        pass
                
                # Perform MCTS search: parallel or sequential
                try:
                    if self.parallel_mcts and self.num_agents > 1:
                        # Parallel MCTS: all agents search simultaneously
                        if episode_length == 0:
                            self.log(f"Using parallel MCTS with {self.max_workers} process workers...")
                        actions = self._parallel_mcts_search(obs, env_state)
                    else:
                        # Sequential MCTS: one agent at a time (original behavior)
                        if episode_length == 0:
                            self.log(f"Using sequential MCTS...")
                        actions = self._sequential_mcts_search(obs, env_state)
                    
                    # Ensure actions have correct shape
                    if not isinstance(actions, np.ndarray):
                        actions = np.array(actions)
                    
                    # Fix shape issues
                    if actions.ndim == 0:
                        # Scalar - convert to (num_agents, 2) with zeros
                        actions = np.zeros((self.num_agents, 2))
                    elif actions.ndim == 1:
                        # Single action - reshape to (1, 2)
                        if actions.size == 2:
                            actions = actions.reshape(1, 2)
                        elif actions.size == 1:
                            # Single value - pad with zero
                            actions = np.array([[actions[0], 0.0]])
                        else:
                            # Multiple values - take first 2
                            actions = actions[:2].reshape(1, 2)
                    elif actions.ndim == 2:
                        # 2D array - check each row
                        if actions.shape[0] != self.num_agents or actions.shape[1] != 2:
                            # Fix shape
                            fixed_actions = []
                            for i in range(self.num_agents):
                                if i < actions.shape[0]:
                                    row = actions[i].flatten()
                                    if row.size >= 2:
                                        fixed_actions.append(row[:2])
                                    elif row.size == 1:
                                        fixed_actions.append(np.array([row[0], 0.0]))
                                    else:
                                        fixed_actions.append(np.zeros(2))
                                else:
                                    fixed_actions.append(np.zeros(2))
                            actions = np.array(fixed_actions)
                    else:
                        # Higher dimensions - flatten and reshape
                        actions_flat = actions.flatten()
                        if actions_flat.size >= self.num_agents * 2:
                            actions = actions_flat[:self.num_agents * 2].reshape(self.num_agents, 2)
                        else:
                            actions = np.zeros((self.num_agents, 2))
                    
                    # Final validation: ensure each row has exactly 2 elements
                    if actions.shape != (self.num_agents, 2):
                        # Last resort: rebuild from scratch
                        fixed_actions = []
                        for i in range(self.num_agents):
                            if i < len(actions):
                                row = np.asarray(actions[i]).flatten()
                                if row.size >= 2:
                                    fixed_actions.append(row[:2])
                                elif row.size == 1:
                                    fixed_actions.append(np.array([row[0], 0.0]))
                                else:
                                    fixed_actions.append(np.zeros(2))
                            else:
                                fixed_actions.append(np.zeros(2))
                        actions = np.array(fixed_actions)
                    
                    # For single agent, convert (1, 2) to (2,) for env.step() compatibility
                    # The env.step() expects different formats for single vs multi-agent
                    if self.num_agents == 1:
                        # Single agent: extract the first row as 1D array
                        actions = actions[0]  # Shape: (2,)
                    # For multi-agent, keep as (num_agents, 2)
                    
                    # MCTS search completed (output disabled - only episode-level logs are shown)
                except Exception as e:
                    self.log(f"Error in MCTS search at episode {episode}, step {episode_length}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Use zero actions as fallback
                    if self.num_agents == 1:
                        actions = np.zeros(2)  # Single agent: (2,)
                    else:
                        actions = np.zeros((self.num_agents, 2))  # Multi-agent: (num_agents, 2)
                
                # Debug: print actions for first few episodes
                if self.enable_debug and episode <= 10 and episode_length == 0:
                    print(f"\n[Episode {episode}, Step 0] Actions selected:")
                    if self.num_agents == 1:
                        # Single agent: actions is 1D array
                        print(f"  Agent 0: {actions} (shape: {actions.shape}, dtype: {actions.dtype})")
                        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
                            print(f"  WARNING: Invalid actions detected (NaN or Inf)!")
                        if np.any(np.abs(actions) > 1.0):
                            print(f"  WARNING: Actions out of range [-1, 1]!")
                            print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
                    else:
                        # Multi-agent: actions is 2D array
                        for i, act in enumerate(actions):
                            print(f"  Agent {i}: {act} (shape: {act.shape}, dtype: {act.dtype})")
                        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
                            print(f"  WARNING: Invalid actions detected (NaN or Inf)!")
                        if np.any(np.abs(actions) > 1.0):
                            print(f"  WARNING: Actions out of range [-1, 1]!")
                            print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
                
                # Step environment
                try:
                    # Environment step (output disabled - only episode-level logs are shown)
                    next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                    # Environment step completed (output disabled - only episode-level logs are shown)
                except Exception as e:
                    self.log(f"Error stepping environment at episode {episode}, step {episode_length}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                
                # Handle reward format
                if isinstance(rewards, (list, np.ndarray)):
                    rewards = np.array(rewards)
                else:
                    rewards = np.array([rewards] * self.num_agents)
                
                done = terminated or truncated
                
                # Debug: check why episode ended (moved after done is set)
                if self.enable_debug and done and episode_length == 0 and episode <= 10:
                    collisions = info.get('collisions', {})
                    print(f"\n[Episode {episode}] Ended at step 0!")
                    print(f"  Terminated: {terminated}, Truncated: {truncated}")
                    print(f"  Rewards: {rewards}")
                    print(f"  Mean reward: {rewards.mean():.2f}")
                    print(f"  Collisions: {collisions}")
                    if hasattr(self.env, 'agents'):
                        for i, agent in enumerate(self.env.agents):
                            agent_id = id(agent)
                            if agent_id in collisions:
                                print(f"  Agent {i} (id={agent_id}): {collisions[agent_id]}")
                    # Check agent positions
                    if hasattr(self.env, 'agents'):
                        print(f"  Agent positions:")
                        for i, agent in enumerate(self.env.agents):
                            print(f"    Agent {i}: pos=({agent.pos_x:.1f}, {agent.pos_y:.1f}), heading={agent.heading:.2f}, speed={agent.speed:.2f}")
                
                # Store transitions and update in batches
                self._update_networks(obs, actions, rewards, next_obs, done)
                
                episode_reward += rewards
                episode_length += 1
                step_count += 1
                
                # Update progress bar
                pbar.update(1)
                # Update progress bar description with current reward
                avg_reward = episode_reward.mean()
                pbar.set_postfix({
                    'reward': f'{avg_reward:.2f}',
                    'done': done
                })
                
                obs = next_obs
                
                # Render if enabled (after action execution)
                if self.render:
                    self.env.render(show_lane_ids=self.show_lane_ids, show_lidar=self.show_lidar)
                    # Add small delay to make rendering visible and keep window responsive
                    import time
                    time.sleep(0.1)  # 100ms delay to make rendering visible
                    # Also handle events to keep window responsive
                    try:
                        import pygame
                        if pygame.get_init():
                            pygame.event.pump()
                    except:
                        pass
            
            # Close progress bar
            pbar.close()
            
            # Final update if buffer has remaining data
            if hasattr(self, 'buffer') and len(self.buffer['obs'][0]) > 0:
                self._batch_update_networks(obs, done)
            
            # Update statistics
            self.stats['total_steps'] = step_count
            self.stats['episode_rewards'].append(episode_reward.mean())
            self.stats['episode_lengths'].append(episode_length)
            
            # Debug: print detailed info for early episodes
            if self.enable_debug and episode <= 10:
                print(f"[Episode {episode}] Summary:")
                print(f"  Episode length: {episode_length}")
                print(f"  Total reward: {episode_reward.sum():.2f}")
                print(f"  Mean reward: {episode_reward.mean():.2f}")
                print(f"  Reward per agent: {episode_reward}")
                collisions = info.get('collisions', {})
                if collisions:
                    print(f"  Collisions: {collisions}")
            
            # Write episode rewards to CSV
            total_reward = episode_reward.sum()
            mean_reward = episode_reward.mean()
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([episode, total_reward, mean_reward, episode_length])
            
            # Check for success/crash
            collisions = info.get('collisions', {})
            has_success = any(
                status == 'SUCCESS' for status in collisions.values()
            )
            has_crash = any(
                status in ['CRASH_CAR', 'CRASH_WALL', 'CRASH_LINE']
                for status in collisions.values()
            )
            
            if has_success:
                self.stats['success_count'] += 1
            if has_crash:
                self.stats['crash_count'] += 1
            
            # Logging
            if episode % self.log_frequency == 0:
                avg_reward = np.mean(self.stats['episode_rewards'][-self.log_frequency:])
                avg_length = np.mean(self.stats['episode_lengths'][-self.log_frequency:])
                success_rate = self.stats['success_count'] / self.log_frequency
                crash_rate = self.stats['crash_count'] / self.log_frequency
                
                # Get rollout statistics from MCTS instances (always enabled now)
                rollout_info = ""
                total_rollouts = sum(mcts.get_rollout_stats()['total_rollouts'] for mcts in self.mcts_instances)
                total_env_steps = sum(mcts.get_rollout_stats()['total_env_steps'] for mcts in self.mcts_instances)
                successful_rollouts = sum(mcts.get_rollout_stats()['successful_rollouts'] for mcts in self.mcts_instances)
                if total_rollouts > 0:
                    rollout_info = f" | Rollouts: {total_rollouts}({successful_rollouts}✓) | EnvSteps: {total_env_steps}"
                
                self.log(
                    f"Episode {episode:5d} | "
                    f"Reward: {avg_reward:7.2f} | "
                    f"Length: {avg_length:5.1f} | "
                    f"Success: {success_rate:.2%} | "
                    f"Crash: {crash_rate:.2%} | "
                    f"Steps: {step_count}{rollout_info}"
                )
                
                # Reset rollout stats after logging
                for mcts in self.mcts_instances:
                    mcts.reset_rollout_stats()
                
                # Reset counters
                self.stats['success_count'] = 0
                self.stats['crash_count'] = 0
            
            # Save checkpoint
            if episode % self.save_frequency == 0:
                checkpoint_path = os.path.join(
                    self.save_dir, f"mcts_episode_{episode}.pth"
                )
                self.save_checkpoint(checkpoint_path, episode)
                self.log(f"Checkpoint saved: {checkpoint_path}")
        
        self.log("Training completed!")
    
    def _parallel_mcts_search(self, obs, env_state):
        """
        Perform MCTS search for all agents in parallel using process pool.
        
        Args:
            obs: Current observations for all agents
            env_state: Environment state for rollouts
            
        Returns:
            actions: Array of actions for all agents
        """
        # Create process pool if not exists
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Submit searches (output disabled - only episode-level logs are shown)
        
        # Note: For process pool, we need to pass serializable data
        # Networks and MCTS instances cannot be passed directly, so we'll need to
        # recreate them in each process using the module-level wrapper function
        # Prepare arguments for each agent (must be serializable)
        search_args = []
        for i in range(self.num_agents):
            # Get network state dict
            network_state_dict = {k: v.cpu() for k, v in self.networks[i].state_dict().items()}
            
            config = {
                'hidden_dim': 256,
                'lstm_hidden_dim': 128,
                'use_lstm': True,
                'sequence_length': 5,
                'device': str(self.device),
                'num_simulations': self.mcts_simulations,
                'c_puct': 1.0,
                'temperature': 1.0,
                'rollout_depth': self.rollout_depth,
                'num_action_samples': self.num_action_samples,
                'network_state_dict': network_state_dict,
                'env_config': {  # Pass config instead of factory
                    'traffic_flow': self.traffic_flow,
                    'num_agents': self.num_agents,
                    'num_lanes': self.num_lanes,
                    'use_team_reward': self.use_team_reward,
                    'max_steps': self.max_steps_per_episode,
                    'respawn_enabled': self.respawn_enabled
                }
            }
            
            # Add traffic_density if using traffic flow
            if self.traffic_flow:
                config['env_config']['traffic_density'] = self.traffic_density
            
            # Convert hidden state to CPU and numpy
            hidden_state_cpu = None
            if self.hidden_states[i] is not None:
                if isinstance(self.hidden_states[i], tuple):
                    hidden_state_cpu = tuple(h.cpu().numpy() for h in self.hidden_states[i])
                else:
                    hidden_state_cpu = self.hidden_states[i].cpu().numpy()
            
            # Ensure env_state is fully serializable (deep copy and convert to basic types)
            import copy as cp
            env_state_serializable = {
                'step_count': int(env_state.get('step_count', 0)),
                'obs': [obs_arr.copy() if isinstance(obs_arr, np.ndarray) else obs_arr for obs_arr in env_state.get('obs', [])],
                'agent_states': env_state.get('agent_states', [])
            }
            
            search_args.append((
                i,
                obs[i].copy(),
                [h.copy() if isinstance(h, np.ndarray) else h for h in list(self.obs_history[i])],
                hidden_state_cpu,
                env_state_serializable,  # Fully serializable
                config
            ))
        
        # Submit all agent searches to process pool
        # Use module-level function for process pool (must be picklable)
        futures = {self._process_pool.submit(_search_agent_wrapper, args): i for i, args in enumerate(search_args)}
        
        # Collect results as they complete
        actions = [None] * self.num_agents
        completed = 0
        errors = []
        start_time = datetime.now()
        
        try:
            # Wait for results (output disabled - only episode-level logs are shown)
            for future in as_completed(futures, timeout=600):  # 10 minute total timeout
                completed += 1
                
                try:
                    i, action = future.result(timeout=120)  # 2 minute timeout per result
                    # Ensure action is a numpy array with exactly 2 elements
                    action = np.asarray(action).flatten()
                    if action.size != 2:
                        if action.size == 1:
                            action = np.array([action[0], 0.0])
                        elif action.size == 0:
                            action = np.zeros(2)
                        else:
                            action = action[:2]
                    actions[i] = action
                except Exception as e:
                    agent_id = futures[future]
                    error_msg = f"Error getting result for agent {agent_id}: {e}"
                    errors.append(error_msg)
                    self.log(error_msg)
                    import traceback
                    self.log(traceback.format_exc())
                    # Use zero action as fallback
                    actions[agent_id] = np.zeros(2)
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            self.log(f"Fatal error in as_completed after {elapsed:.1f}s: {e}")
            import traceback
            self.log(traceback.format_exc())
            # Fill remaining actions with zeros
            for i in range(self.num_agents):
                if actions[i] is None:
                    self.log(f"Warning: Agent {i} action is None, using zero action")
                    actions[i] = np.zeros(2)
        
        # Check if all actions are collected and ensure correct format
        for i in range(self.num_agents):
            if actions[i] is None:
                self.log(f"Warning: Agent {i} action is None, using zero action")
                actions[i] = np.zeros(2)
            else:
                # Ensure each action is a numpy array with exactly 2 elements
                action = np.asarray(actions[i]).flatten()
                if action.size != 2:
                    if action.size == 1:
                        actions[i] = np.array([action[0], 0.0])
                    elif action.size == 0:
                        actions[i] = np.zeros(2)
                    else:
                        actions[i] = action[:2]
                else:
                    actions[i] = action
        
        if errors:
            self.log(f"Total errors during parallel MCTS: {len(errors)}")
        
        # Convert to numpy array and ensure shape is (num_agents, 2)
        actions_array = np.array(actions)
        if actions_array.shape != (self.num_agents, 2):
            # Rebuild to ensure correct shape
            fixed_actions = []
            for i in range(self.num_agents):
                if i < len(actions):
                    row = np.asarray(actions[i]).flatten()
                    if row.size >= 2:
                        fixed_actions.append(row[:2])
                    elif row.size == 1:
                        fixed_actions.append(np.array([row[0], 0.0]))
                    else:
                        fixed_actions.append(np.zeros(2))
                else:
                    fixed_actions.append(np.zeros(2))
            actions_array = np.array(fixed_actions)
        
        return actions_array
    
    def _sequential_mcts_search(self, obs, env_state):
        """
        Perform MCTS search for all agents sequentially (one at a time).
        
        Args:
            obs: Current observations for all agents
            env_state: Environment state for rollouts
            
        Returns:
            actions: Array of actions for all agents
        """
        actions = []
        
        # Execute sequentially for each agent
        for i in range(self.num_agents):
            # Handle pygame events during MCTS search to keep window responsive
            if self.render:
                try:
                    import pygame
                    if pygame.get_init() and hasattr(self.env, 'screen') and self.env.screen is not None:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                return np.zeros((self.num_agents, 2))
                except Exception:
                    pass
            
            try:
                action, search_stats = self.mcts_instances[i].search(
                    root_state=obs[i],
                    obs_history=list(self.obs_history[i]),
                    hidden_state=self.hidden_states[i],
                    env_state=env_state
                )
            except Exception as e:
                # If search fails, use zero action
                action = np.zeros(2)
            
            # Ensure action is numpy array with correct shape (2 elements)
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            
            # Ensure action is 1D with 2 elements
            action = action.flatten()
            if action.size != 2:
                # Wrong size - use zeros
                action = np.zeros(2)
            
            # Update hidden state after MCTS search
            obs_seq_list = list(self.obs_history[i])
            if len(obs_seq_list) > 0:
                try:
                    # Use the helper function to prepare sequence
                    # Import from mcts module (same directory)
                    import sys
                    import os
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    from mcts import _prepare_obs_sequence
                    obs_seq = _prepare_obs_sequence(obs_seq_list, 5)
                    if obs_seq is not None:
                        obs_tensor = torch.FloatTensor(obs_seq).unsqueeze(0).to(self.device)
                        _, _, _, self.hidden_states[i] = self.networks[i](obs_tensor, self.hidden_states[i])
                except Exception:
                    # If update fails, continue without updating hidden state
                    pass
            
            # Add exploration noise in early training
            if self.stats['episode'] < 1000:
                noise_scale = 0.3 * (1.0 - self.stats['episode'] / 1000.0)
                action = action + np.random.normal(0, noise_scale, size=action.shape)
                action = np.clip(action, -1.0, 1.0)
            
            actions.append(action)
        
        # Ensure return shape is (num_agents, 2) and each action has exactly 2 elements
        # First, ensure each action in the list has exactly 2 elements
        for i in range(len(actions)):
            action = np.asarray(actions[i]).flatten()
            if action.size != 2:
                if action.size == 1:
                    actions[i] = np.array([action[0], 0.0])
                elif action.size == 0:
                    actions[i] = np.zeros(2)
                else:
                    actions[i] = action[:2]
            else:
                actions[i] = action
        
        # Convert to numpy array
        actions_array = np.array(actions)
        
        # Final validation and fix if needed
        if actions_array.shape != (self.num_agents, 2):
            # Rebuild to ensure correct shape
            fixed_actions = []
            for i in range(self.num_agents):
                if i < len(actions):
                    row = np.asarray(actions[i]).flatten()
                    if row.size >= 2:
                        fixed_actions.append(row[:2])
                    elif row.size == 1:
                        fixed_actions.append(np.array([row[0], 0.0]))
                    else:
                        fixed_actions.append(np.zeros(2))
                else:
                    fixed_actions.append(np.zeros(2))
            actions_array = np.array(fixed_actions)
        
        return actions_array
    
    def _batched_mcts_search(self, obs, env_state):
        """
        Perform MCTS search for all agents with batched rollouts.
        Collects rollout requests from all agents and executes them in batches.
        
        Args:
            obs: Current observations for all agents
            env_state: Environment state for rollouts
            
        Returns:
            actions: Array of actions for all agents
        """
        # Initialize a shared rollout queue for batch processing
        # This allows all agents to submit rollout requests that can be executed together
        actions = []
        
        # For now, execute sequentially but with shared environment state
        # This is a stepping stone - we can optimize further by batching rollout execution
        for i in range(self.num_agents):
            # Handle pygame events during MCTS search to keep window responsive
            if self.render:
                try:
                    import pygame
                    if pygame.get_init() and hasattr(self.env, 'screen') and self.env.screen is not None:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                return np.array(actions + [np.zeros(2)] * (self.num_agents - len(actions)))
                except Exception:
                    pass
            
            action, search_stats = self.mcts_instances[i].search(
                root_state=obs[i],
                obs_history=list(self.obs_history[i]),
                hidden_state=self.hidden_states[i],
                env_state=env_state
            )
            
            # Ensure action is numpy array with correct shape
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            if action.ndim == 0:
                action = np.array([action])
            action = action.flatten()
            
            # Update hidden state after MCTS search
            obs_seq = np.array(list(self.obs_history[i]))
            if len(obs_seq) < 5:
                obs_seq = np.array([obs_seq[0]] * (5 - len(obs_seq)) + list(obs_seq))
            obs_tensor = torch.FloatTensor(obs_seq).unsqueeze(0).to(self.device)
            _, _, _, self.hidden_states[i] = self.networks[i](obs_tensor, self.hidden_states[i])
            
            # Add exploration noise in early training
            if self.stats['episode'] < 1000:
                noise_scale = 0.3 * (1.0 - self.stats['episode'] / 1000.0)
                action = action + np.random.normal(0, noise_scale, size=action.shape)
                action = np.clip(action, -1.0, 1.0)
            
            actions.append(action)
        
        return np.array(actions)
    
    def _update_networks(self, obs, actions, rewards, next_obs, done):
        """
        Update networks using experience buffer.
        Store transitions and update in batches for stability.
        """
        # Store transitions in buffer
        for i in range(self.num_agents):
            # Build observation sequence
            obs_list = list(self.obs_history[i])
            
            # Ensure all observations are numpy arrays with correct shape
            obs_list_processed = []
            for obs_item in obs_list:
                obs_arr = np.asarray(obs_item)
                if obs_arr.ndim == 0:
                    # Scalar - this shouldn't happen, but handle it
                    obs_arr = np.zeros(118)  # OBS_DIM
                elif obs_arr.ndim == 1:
                    # 1D array - ensure it has correct size
                    if obs_arr.size != 118:
                        if obs_arr.size == 0:
                            obs_arr = np.zeros(118)
                        elif obs_arr.size < 118:
                            # Pad with zeros
                            obs_arr = np.pad(obs_arr, (0, 118 - obs_arr.size), 'constant')
                        else:
                            # Take first 118 elements
                            obs_arr = obs_arr[:118]
                    obs_list_processed.append(obs_arr)
                else:
                    # Multi-dimensional - flatten and take first 118
                    obs_arr = obs_arr.flatten()[:118]
                    if obs_arr.size < 118:
                        obs_arr = np.pad(obs_arr, (0, 118 - obs_arr.size), 'constant')
                    obs_list_processed.append(obs_arr)
            
            # Build sequence with correct length (5)
            if len(obs_list_processed) == 0:
                # No history - use zeros
                obs_seq = np.zeros((5, 118))
            elif len(obs_list_processed) < 5:
                # Pad with first observation
                if len(obs_list_processed) > 0:
                    first_obs = obs_list_processed[0]
                    padding = [first_obs] * (5 - len(obs_list_processed))
                    obs_seq = np.array(padding + obs_list_processed)
                else:
                    obs_seq = np.zeros((5, 118))
            else:
                # Take last 5 observations
                obs_seq = np.array(obs_list_processed[-5:])
            
            # Ensure obs_seq has correct shape (5, 118)
            if obs_seq.shape != (5, 118):
                if obs_seq.ndim == 1:
                    # If 1D, reshape to (1, 118) and pad
                    obs_seq = obs_seq.reshape(1, -1)
                    if obs_seq.shape[1] < 118:
                        obs_seq = np.pad(obs_seq, ((0, 0), (0, 118 - obs_seq.shape[1])), 'constant')
                    # Repeat to get 5 timesteps
                    obs_seq = np.repeat(obs_seq, 5, axis=0)
                elif obs_seq.ndim == 2:
                    # Fix shape
                    if obs_seq.shape[0] < 5:
                        # Pad with first row
                        padding = np.repeat(obs_seq[0:1], 5 - obs_seq.shape[0], axis=0)
                        obs_seq = np.vstack([padding, obs_seq])
                    elif obs_seq.shape[0] > 5:
                        # Take last 5
                        obs_seq = obs_seq[-5:]
                    
                    if obs_seq.shape[1] < 118:
                        # Pad columns
                        obs_seq = np.pad(obs_seq, ((0, 0), (0, 118 - obs_seq.shape[1])), 'constant')
                    elif obs_seq.shape[1] > 118:
                        # Take first 118 columns
                        obs_seq = obs_seq[:, :118]
                else:
                    # Unexpected shape - create zeros
                    obs_seq = np.zeros((5, 118))
            
            # Get value estimate
            obs_tensor = torch.FloatTensor(obs_seq).unsqueeze(0).to(self.device)  # Shape: (1, 5, 118)
            _, _, value_pred, _ = self.networks[i](obs_tensor, self.hidden_states[i])
            
            # Store transition (we'll update in batches)
            if not hasattr(self, 'buffer'):
                self.buffer = {
                    'obs': [[] for _ in range(self.num_agents)],
                    'actions': [[] for _ in range(self.num_agents)],
                    'rewards': [[] for _ in range(self.num_agents)],
                    'values': [[] for _ in range(self.num_agents)],
                    'dones': [[] for _ in range(self.num_agents)],
                }
            
            self.buffer['obs'][i].append(obs_seq)
            self.buffer['actions'][i].append(actions[i])
            self.buffer['rewards'][i].append(rewards[i])
            self.buffer['values'][i].append(value_pred.item())
            self.buffer['dones'][i].append(done)
        
        # Update networks when buffer reaches threshold 
        buffer_size = len(self.buffer['obs'][0])
        if buffer_size >= 64:  # Update every 64 steps
            self._batch_update_networks(next_obs, done)
    
    def _batch_update_networks(self, next_obs, done):
        """Batch update networks using stored transitions."""
        gamma = 0.99  # Discount factor
        
        for i in range(self.num_agents):
            if len(self.buffer['obs'][i]) == 0:
                continue
            
            # Get buffer data
            obs_seqs = np.array(self.buffer['obs'][i])  # (T, seq_len, obs_dim)
            actions = np.array(self.buffer['actions'][i])  # (T, action_dim)
            rewards = np.array(self.buffer['rewards'][i])  # (T,)
            values = np.array(self.buffer['values'][i])  # (T,)
            dones = np.array(self.buffer['dones'][i])  # (T,)
            
            # Compute returns (discounted rewards)
            returns = np.zeros_like(rewards)
            if not done:
                # Get next value estimate - use same logic as _update_networks
                obs_list = list(self.obs_history[i])
                
                # Ensure all observations are numpy arrays with correct shape
                obs_list_processed = []
                for obs_item in obs_list:
                    obs_arr = np.asarray(obs_item)
                    if obs_arr.ndim == 0:
                        obs_arr = np.zeros(118)  # OBS_DIM
                    elif obs_arr.ndim == 1:
                        if obs_arr.size != 118:
                            if obs_arr.size == 0:
                                obs_arr = np.zeros(118)
                            elif obs_arr.size < 118:
                                obs_arr = np.pad(obs_arr, (0, 118 - obs_arr.size), 'constant')
                            else:
                                obs_arr = obs_arr[:118]
                        obs_list_processed.append(obs_arr)
                    else:
                        obs_arr = obs_arr.flatten()[:118]
                        if obs_arr.size < 118:
                            obs_arr = np.pad(obs_arr, (0, 118 - obs_arr.size), 'constant')
                        obs_list_processed.append(obs_arr)
                
                # Build sequence with correct length (5)
                if len(obs_list_processed) == 0:
                    next_obs_seq = np.zeros((5, 118))
                elif len(obs_list_processed) < 5:
                    if len(obs_list_processed) > 0:
                        first_obs = obs_list_processed[0]
                        padding = [first_obs] * (5 - len(obs_list_processed))
                        next_obs_seq = np.array(padding + obs_list_processed)
                    else:
                        next_obs_seq = np.zeros((5, 118))
                else:
                    next_obs_seq = np.array(obs_list_processed[-5:])
                
                # Ensure next_obs_seq has correct shape (5, 118)
                if next_obs_seq.shape != (5, 118):
                    if next_obs_seq.ndim == 1:
                        next_obs_seq = next_obs_seq.reshape(1, -1)
                        if next_obs_seq.shape[1] < 118:
                            next_obs_seq = np.pad(next_obs_seq, ((0, 0), (0, 118 - next_obs_seq.shape[1])), 'constant')
                        next_obs_seq = np.repeat(next_obs_seq, 5, axis=0)
                    elif next_obs_seq.ndim == 2:
                        if next_obs_seq.shape[0] < 5:
                            padding = np.repeat(next_obs_seq[0:1], 5 - next_obs_seq.shape[0], axis=0)
                            next_obs_seq = np.vstack([padding, next_obs_seq])
                        elif next_obs_seq.shape[0] > 5:
                            next_obs_seq = next_obs_seq[-5:]
                        
                        if next_obs_seq.shape[1] < 118:
                            next_obs_seq = np.pad(next_obs_seq, ((0, 0), (0, 118 - next_obs_seq.shape[1])), 'constant')
                        elif next_obs_seq.shape[1] > 118:
                            next_obs_seq = next_obs_seq[:, :118]
                    else:
                        next_obs_seq = np.zeros((5, 118))
                
                next_obs_tensor = torch.FloatTensor(next_obs_seq).unsqueeze(0).to(self.device)  # Shape: (1, 5, 118)
                _, _, next_value, _ = self.networks[i](next_obs_tensor, self.hidden_states[i])
                returns[-1] = rewards[-1] + gamma * next_value.item() * (1 - done)
            else:
                returns[-1] = rewards[-1]
            
            # Compute discounted returns backwards
            for t in reversed(range(len(rewards) - 1)):
                returns[t] = rewards[t] + gamma * returns[t + 1] * (1 - dones[t])
            
            # Compute advantages
            advantages = returns - values
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Convert to tensors
            obs_tensor = torch.FloatTensor(obs_seqs).to(self.device)  # (T, seq_len, obs_dim)
            actions_tensor = torch.FloatTensor(actions).to(self.device)  # (T, action_dim)
            returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(self.device)  # (T, 1)
            advantages_tensor = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)  # (T, 1)
            
            # Forward pass
            policy_mean, policy_std, value_pred, _ = self.networks[i](obs_tensor, None)  # Reset hidden state for batch
            
            # Value loss
            value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(), returns_tensor.squeeze())
            
            # Policy loss (REINFORCE with advantage)
            dist = torch.distributions.Normal(policy_mean, policy_std)
            log_probs = dist.log_prob(actions_tensor).sum(dim=1, keepdim=True)  # (T, 1)
            policy_loss = -(log_probs * advantages_tensor).mean()
            
            # Total loss
            total_loss = value_loss + 0.5 * policy_loss
            
            # Backward pass
            self.optimizers[i].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.networks[i].parameters(), 0.5)
            self.optimizers[i].step()
        
        # Clear buffer
        self.buffer = {
            'obs': [[] for _ in range(self.num_agents)],
            'actions': [[] for _ in range(self.num_agents)],
            'rewards': [[] for _ in range(self.num_agents)],
            'values': [[] for _ in range(self.num_agents)],
            'dones': [[] for _ in range(self.num_agents)],
        }
    
    def save_checkpoint(self, path: str, episode: int):
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'networks_state_dict': [net.state_dict() for net in self.networks],
            'optimizers_state_dict': [opt.state_dict() for opt in self.optimizers],
            'stats': self.stats,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, net in enumerate(self.networks):
            net.load_state_dict(checkpoint['networks_state_dict'][i])
        
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(checkpoint['optimizers_state_dict'][i])
        
        self.stats = checkpoint['stats']
        self.log(f"Checkpoint loaded: {path}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Single-Agent MCTS Training with Traffic Flow')
    parser.add_argument('--num-agents', type=int, default=1, help='Number of agents (default: 1 for traffic flow)')
    parser.add_argument('--num-lanes', type=int, default=3, help='Number of lanes per direction')
    parser.add_argument('--max-episodes', type=int, default=10000, help='Max training episodes')
    parser.add_argument('--max-steps', type=int, default=2000, help='Max steps per episode')
    parser.add_argument('--mcts-simulations', type=int, default=5, help='Number of MCTS simulations per step (default: 5)')
    parser.add_argument('--rollout-depth', type=int, default=3, help='Number of steps to rollout in environment (default: 3)')
    parser.add_argument('--num-action-samples', type=int, default=3, help='Number of actions to sample per node expansion (default: 3)')
    parser.add_argument('--save-frequency', type=int, default=100, help='Episodes before saving checkpoint')
    parser.add_argument('--log-frequency', type=int, default=10, help='Episodes before logging')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--use-team-reward', action='store_true', help='Use team reward (not used in traffic flow mode)')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--show-lane-ids', action='store_true', help='Show lane IDs in render')
    parser.add_argument('--show-lidar', action='store_true', help='Show lidar in render')
    parser.add_argument('--respawn', action='store_true', help='Enable respawn (disabled by default for single-agent mode)')
    parser.add_argument('--save-dir', type=str, default='MCTS_SINGLE/checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--load-checkpoint', type=str, default=None, help='Path to checkpoint to load')
    parser.add_argument('--no-parallel-mcts', action='store_true', help='Disable parallel MCTS (use sequential)')
    parser.add_argument('--max-workers', type=int, default=1, help='Max processes for parallel MCTS (default: 1 for single agent)')
    parser.add_argument('--no-traffic-flow', action='store_false', dest='traffic_flow', default=True, help='Disable traffic flow mode (use multi-agent mode)')
    parser.add_argument('--traffic-density', type=float, default=10, help='Traffic density (0.0-1.0) for traffic flow mode (default: 0.5)')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print(f"CUDA available, using CUDA")
            print(f"{torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
    else:
        device = args.device
    
    # Create trainer
    trainer = MCTSTrainer(
        num_agents=args.num_agents,
        num_lanes=args.num_lanes,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps,
        mcts_simulations=args.mcts_simulations,
        rollout_depth=args.rollout_depth,
        num_action_samples=args.num_action_samples,
        save_frequency=args.save_frequency,
        log_frequency=args.log_frequency,
        device=device,
        use_team_reward=args.use_team_reward,
        render=args.render,
        show_lane_ids=args.show_lane_ids,
        show_lidar=args.show_lidar,
        respawn_enabled=args.respawn,  # Disabled by default for single-agent mode
        save_dir=args.save_dir,
        parallel_mcts=not args.no_parallel_mcts,  # Enable by default
        max_workers=args.max_workers,
        traffic_flow=args.traffic_flow,
        traffic_density=args.traffic_density
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Start training
    try:
        print("Calling trainer.train()...")
        trainer.train()
        print("Training completed normally.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save final checkpoint
        final_checkpoint = os.path.join(args.save_dir, 'mcts_interrupted.pth')
        trainer.save_checkpoint(final_checkpoint, trainer.stats['episode'])
        print(f"Final checkpoint saved: {final_checkpoint}")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        # Save final checkpoint if possible
        try:
            final_checkpoint = os.path.join(args.save_dir, 'mcts_error.pth')
            trainer.save_checkpoint(final_checkpoint, trainer.stats.get('episode', 0))
            print(f"Error checkpoint saved: {final_checkpoint}")
        except:
            pass
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        # Save final checkpoint if possible
        try:
            final_checkpoint = os.path.join(args.save_dir, 'mcts_error.pth')
            trainer.save_checkpoint(final_checkpoint, trainer.stats.get('episode', 0))
            print(f"Error checkpoint saved: {final_checkpoint}")
        except:
            pass


if __name__ == '__main__':
    main()
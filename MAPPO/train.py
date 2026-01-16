# --- train.py ---
# Training script for MAPPO with 6 agents

import os
import sys
import numpy as np
import torch
import time
from datetime import datetime
import json

# Add parent directory to path to import env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Intersection.env import IntersectionEnv, DEFAULT_ROUTE_MAPPING_2LANES, DEFAULT_ROUTE_MAPPING_3LANES
from Intersection.config import DEFAULT_REWARD_CONFIG, OBS_DIM

# Import MAPPO - handle both absolute and relative imports
try:
    from .mappo import MAPPO
except ImportError:
    from mappo import MAPPO


def generate_ego_routes(num_agents: int, num_lanes: int):
    """
    Generate ego routes for agents based on default route mappings, ensuring balanced distribution.
    
    Args:
        num_agents: Number of agents
        num_lanes: Number of lanes per direction (2 or 3)
        
    Returns:
        List of (start_id, end_id) tuples for ego vehicles
    """
    # Select the appropriate route mapping based on num_lanes
    if num_lanes == 2:
        route_mapping = DEFAULT_ROUTE_MAPPING_2LANES
    elif num_lanes == 3:
        route_mapping = DEFAULT_ROUTE_MAPPING_3LANES
    else:
        # Fallback: generate routes based on lane calculation
        route_mapping = {}
        for dir_idx in range(4):  # 4 directions: N, E, S, W
            for lane_idx in range(num_lanes):
                in_id = dir_idx * num_lanes + lane_idx + 1
                # Default: straight route
                opposite_dir_idx = (dir_idx + 2) % 4
                out_id = opposite_dir_idx * num_lanes + lane_idx + 1
                route_mapping[f"IN_{in_id}"] = [f"OUT_{out_id}"]
    
    # Generate all possible routes from the mapping
    all_routes = []
    for in_id, out_ids in route_mapping.items():
        for out_id in out_ids:
            all_routes.append((in_id, out_id))
    
    # Group routes by starting direction for balanced distribution
    # Direction calculation: (IN_id - 1) // num_lanes
    routes_by_dir = {i: [] for i in range(4)}
    for route in all_routes:
        in_id = route[0]
        in_num = int(in_id.split('_')[1])
        dir_idx = (in_num - 1) // num_lanes
        routes_by_dir[dir_idx].append(route)
    
    # Strategy: distribute agents evenly across directions
    selected_routes = []
    agents_per_dir = num_agents // 4
    extra_agents = num_agents % 4
    
    # First pass: ensure at least one agent from each direction
    for dir_idx in range(4):
        count = agents_per_dir + (1 if dir_idx < extra_agents else 0)
        if count > 0 and routes_by_dir[dir_idx]:
            # Select routes from this direction, cycling through available routes
            for i in range(count):
                route_idx = i % len(routes_by_dir[dir_idx])
                selected_routes.append(routes_by_dir[dir_idx][route_idx])
    
    # If we still need more routes, fill from remaining
    if len(selected_routes) < num_agents:
        all_remaining = []
        used = set(selected_routes)
        for dir_idx in range(4):
            remaining = [r for r in routes_by_dir[dir_idx] if r not in used]
            all_remaining.extend(remaining)
        
        remaining_needed = num_agents - len(selected_routes)
        if all_remaining:
            # Distribute remaining routes evenly
            step = max(1, len(all_remaining) // remaining_needed) if remaining_needed > 0 else 1
            selected_routes.extend(all_remaining[::step][:remaining_needed])
    
    return selected_routes[:num_agents]


class Trainer:
    """MAPPO Trainer for multi-agent intersection navigation."""
    
    def __init__(
        self,
        num_agents: int = 6,
        num_lanes: int = 2,
        max_episodes: int = 10000,
        max_steps_per_episode: int = 2000,
        update_frequency: int = 2048,  # Steps before update
        save_frequency: int = 100,  # Episodes before saving
        log_frequency: int = 10,  # Episodes before logging
        device: str = 'cpu',
        use_team_reward: bool = True,
        render: bool = False,
        show_lane_ids: bool = False,
        show_lidar: bool = False,
        respawn_enabled: bool = False,
        save_dir: str = 'policy/checkpoints'
    ):
        """
        Initialize trainer.
        
        Args:
            num_agents: Number of agents
            num_lanes: Number of lanes per direction (default: 2)
            max_episodes: Maximum training episodes
            max_steps_per_episode: Maximum steps per episode
            update_frequency: Steps before policy update
            save_frequency: Episodes before saving checkpoint
            log_frequency: Episodes before logging stats
            device: Device to use ('cpu' or 'cuda')
            use_team_reward: Whether to use team reward mixing
            render: Whether to render environment
            save_dir: Directory to save checkpoints
        """
        self.num_agents = num_agents
        self.num_lanes = num_lanes
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.device = device
        self.use_team_reward = use_team_reward
        self.render = render
        self.show_lane_ids = show_lane_ids
        self.show_lidar = show_lidar
        self.respawn_enabled = respawn_enabled
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate ego routes dynamically based on num_agents and num_lanes
        ego_routes = generate_ego_routes(num_agents, num_lanes)
        
        # Log generated routes
        print(f"Generated {len(ego_routes)} routes for {num_agents} agents with {num_lanes} lanes:")
        for i, route in enumerate(ego_routes):
            print(f"  Agent {i}: {route[0]} -> {route[1]}")
        
        # Initialize environment
        self.env = IntersectionEnv({
            'traffic_flow': False,  # Multi-agent mode
            'num_agents': num_agents,
            'num_lanes': num_lanes,
            'use_team_reward': use_team_reward,
            'render_mode': 'human' if render else None,
            'max_steps': max_steps_per_episode,
            'respawn_enabled': respawn_enabled,
            'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
            # Dynamically generated routes for balanced distribution
            'ego_routes': ego_routes
        })
        
        # Initialize MAPPO
        self.mappo = MAPPO(
            num_agents=num_agents,
            obs_dim=OBS_DIM,
            action_dim=2,
            hidden_dim=256,
            lr_actor=3e-4,
            lr_critic=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_clip=True,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            device=device
        )
        
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
            f.write(f"Num agents: {num_agents}\n")
            f.write(f"Num lanes: {num_lanes}\n")
            f.write(f"Use team reward: {use_team_reward}\n")
            f.write(f"Respawn enabled: {respawn_enabled}\n")
            f.write("Generated routes:\n")
            for i, route in enumerate(ego_routes):
                f.write(f"  Agent {i}: {route[0]} -> {route[1]}\n")
            f.write("=" * 80 + "\n")
    
    def log(self, message: str):
        """Log message to file and console."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
    def train(self):
        """Main training loop."""
        self.log("=" * 80)
        self.log("Starting MAPPO Training")
        self.log("=" * 80)
        
        episode = 0
        step_count = 0
        
        while episode < self.max_episodes:
            episode += 1
            self.stats['episode'] = episode
            
            # Reset environment
            obs, info = self.env.reset()
            episode_reward = np.zeros(self.num_agents)
            episode_length = 0
            done = False
            
            # Episode loop
            while not done and episode_length < self.max_steps_per_episode:
                # Select actions
                actions, log_probs = self.mappo.select_actions(obs)
                
                # Get value estimates
                values = self.mappo.get_values(obs)
                
                # Step environment
                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                
                # Handle reward format (could be scalar or list)
                if isinstance(rewards, (list, np.ndarray)):
                    rewards = np.array(rewards)
                else:
                    rewards = np.array([rewards] * self.num_agents)
                
                done = terminated or truncated
                dones = np.array([done] * self.num_agents)
                
                # Store transition
                self.mappo.store_transition(
                    obs, actions, rewards, values, log_probs, dones
                )
                
                episode_reward += rewards
                episode_length += 1
                step_count += 1
                
                # Update policy if buffer reaches update_frequency
                # This ensures updates happen even if episode is very long (e.g., with respawn)
                buffer_size = len(self.mappo.buffer['obs'][0])
                if buffer_size >= self.update_frequency:
                    self.mappo.update(next_obs, epochs=10, batch_size=64)
                    self.log(f"Policy updated at step {step_count} (buffer size: {buffer_size})")
                
                obs = next_obs
                
                # Render if enabled
                if self.render:
                    self.env.render(show_lane_ids=self.show_lane_ids, show_lidar=self.show_lidar)
            
            # Final update at end of episode if buffer has remaining data
            buffer_size = len(self.mappo.buffer['obs'][0])
            if buffer_size > 0:
                self.mappo.update(obs, epochs=10, batch_size=64)
                self.log(f"Final update at episode end (buffer size: {buffer_size})")
            
            # Update statistics
            self.stats['total_steps'] = step_count
            self.stats['episode_rewards'].append(episode_reward.mean())
            self.stats['episode_lengths'].append(episode_length)
            
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
                
                self.log(
                    f"Episode {episode:5d} | "
                    f"Reward: {avg_reward:7.2f} | "
                    f"Length: {avg_length:5.1f} | "
                    f"Success: {success_rate:.2%} | "
                    f"Crash: {crash_rate:.2%} | "
                    f"Steps: {step_count}"
                )
                
                # Reset counters
                self.stats['success_count'] = 0
                self.stats['crash_count'] = 0
            
            # Save checkpoint
            if episode % self.save_frequency == 0:
                checkpoint_path = os.path.join(
                    self.save_dir, f"mappo_episode_{episode}.pth"
                )
                self.mappo.save(checkpoint_path)
                self.log(f"Checkpoint saved: {checkpoint_path}")
                
                # Save training stats
                stats_path = os.path.join(self.save_dir, 'training_stats.json')
                with open(stats_path, 'w') as f:
                    json.dump(self.stats, f, indent=2)
        
        # Final save
        final_path = os.path.join(self.save_dir, 'mappo_final.pth')
        self.mappo.save(final_path)
        self.log(f"Training completed! Final model saved: {final_path}")
        
        # Save final stats
        stats_path = os.path.join(self.save_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        self.env.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MAPPO for intersection navigation')
    parser.add_argument('--num-agents', type=int, default=6, help='Number of agents')
    parser.add_argument('--num-lanes', type=int, default=3, help='Number of lanes per direction')
    parser.add_argument('--max-episodes', type=int, default=10000, help='Max training episodes')
    parser.add_argument('--update-frequency', type=int, default=2048, help='Steps before update')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--use-team-reward', action='store_true', help='Use team reward mixing')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--show-lane-ids', action='store_true', help='Show lane IDs in render')
    parser.add_argument('--show-lidar', action='store_true', help='Show lidar rays in render')
    parser.add_argument('--respawn', action='store_true', help='Enable respawn for agents (agents will respawn at start when they crash)')
    parser.add_argument('--save-dir', type=str, default='policy/checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # Check for CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    else:
        print("CUDA available, using CUDA")
        print(torch.cuda.get_device_name(0))
    # Create trainer
    trainer = Trainer(
        num_agents=args.num_agents,
        num_lanes=args.num_lanes,
        max_episodes=args.max_episodes,
        update_frequency=args.update_frequency,
        device=args.device,
        use_team_reward=args.use_team_reward,
        render=args.render,
        show_lane_ids=args.show_lane_ids,
        show_lidar=args.show_lidar,
        respawn_enabled=args.respawn,
        save_dir=args.save_dir
    )
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.mappo.save(os.path.join(args.save_dir, 'mappo_interrupted.pth'))
        trainer.env.close()


if __name__ == '__main__':
    main()

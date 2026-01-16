# --- test_mappo.py ---
# Quick test script to verify MAPPO implementation

import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks import Actor, Critic
from mappo import MAPPO
from Intersection.config import OBS_DIM

def test_networks():
    """Test Actor and Critic networks."""
    print("Testing networks...")
    
    obs_dim = OBS_DIM
    action_dim = 2
    hidden_dim = 256
    
    # Test Actor
    actor = Actor(obs_dim, action_dim, hidden_dim)
    obs = torch.randn(1, obs_dim)
    mean, std = actor(obs)
    action, log_prob = actor.get_action(obs)
    
    print(f"Actor test passed:")
    print(f"  Input shape: {obs.shape}")
    print(f"  Mean shape: {mean.shape}, Range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"  Std shape: {std.shape}, Min: {std.min():.2f}")
    print(f"  Action shape: {action.shape}, Range: [{action.min():.2f}, {action.max():.2f}]")
    print(f"  Log prob shape: {log_prob.shape}")
    
    # Test Critic
    critic = Critic(obs_dim, hidden_dim)
    value = critic(obs)
    
    print(f"Critic test passed:")
    print(f"  Value shape: {value.shape}, Value: {value.item():.2f}")
    
    print("✓ Networks test passed!\n")


def test_mappo():
    """Test MAPPO class."""
    print("Testing MAPPO...")
    
    num_agents = 6
    obs_dim = OBS_DIM
    action_dim = 2
    
    mappo = MAPPO(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device='cpu'
    )
    
    # Test action selection
    obs = np.random.randn(num_agents, obs_dim)
    actions, log_probs = mappo.select_actions(obs)
    
    print(f"MAPPO action selection test passed:")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Actions shape: {actions.shape}, Range: [{actions.min():.2f}, {actions.max():.2f}]")
    print(f"  Log probs shape: {log_probs.shape}")
    
    # Test value estimation
    values = mappo.get_values(obs)
    print(f"  Values shape: {values.shape}")
    
    # Test storing transitions
    rewards = np.random.randn(num_agents)
    dones = np.zeros(num_agents, dtype=bool)
    mappo.store_transition(obs, actions, rewards, values, log_probs, dones)
    
    print(f"  Buffer size: {len(mappo.buffer['obs'][0])}")
    
    # Test update (with dummy data)
    for _ in range(10):
        mappo.store_transition(
            np.random.randn(num_agents, obs_dim),
            np.random.randn(num_agents, action_dim),
            np.random.randn(num_agents),
            np.random.randn(num_agents, 1),
            np.random.randn(num_agents, 1),
            np.zeros(num_agents, dtype=bool)
        )
    
    next_obs = np.random.randn(num_agents, obs_dim)
    mappo.update(next_obs, epochs=1, batch_size=4)
    
    print(f"  Update completed successfully")
    print("✓ MAPPO test passed!\n")


def test_integration():
    """Test integration with environment."""
    print("Testing integration with environment...")
    
    try:
        from Intersection.env import IntersectionEnv
        from Intersection.config import DEFAULT_REWARD_CONFIG
        
        env = IntersectionEnv({
            'traffic_flow': False,
            'num_agents': 6,
            'use_team_reward': True,
            'render_mode': None,
            'max_steps': 100,
            'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
            'ego_routes': [
                ('IN_6', 'OUT_2'),
                ('IN_4', 'OUT_8'),
                ('IN_5', 'OUT_7'),
                ('IN_2', 'OUT_6'),
                ('IN_8', 'OUT_4'),
                ('IN_1', 'OUT_3'),
            ]
        })
        
        mappo = MAPPO(num_agents=6, device='cpu')
        
        obs, info = env.reset()
        print(f"  Environment reset successful")
        print(f"  Obs shape: {obs.shape}")
        
        # Run a few steps
        for step in range(5):
            actions, log_probs = mappo.select_actions(obs)
            values = mappo.get_values(obs)
            
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            if isinstance(rewards, (list, np.ndarray)):
                rewards = np.array(rewards)
            else:
                rewards = np.array([rewards] * 6)
            
            dones = np.array([terminated or truncated] * 6)
            mappo.store_transition(obs, actions, rewards, values, log_probs, dones)
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        print(f"  Integration test completed successfully")
        print("✓ Integration test passed!\n")
        
        env.close()
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("=" * 60)
    print("MAPPO Implementation Test")
    print("=" * 60)
    print()
    
    test_networks()
    test_mappo()
    test_integration()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

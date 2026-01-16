# --- mappo.py ---
# Multi-Agent Proximal Policy Optimization (MAPPO) Implementation

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List, Dict, Tuple

# Import networks - handle both absolute and relative imports
try:
    from .networks import Actor, Critic
except ImportError:
    from networks import Actor, Critic


class MAPPO:
    """Multi-Agent Proximal Policy Optimization algorithm."""
    
    def __init__(
        self,
        num_agents: int = 6,
        obs_dim: int = 118,
        action_dim: int = 2,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_clip: bool = True,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initialize MAPPO.
        
        Args:
            num_agents: Number of agents
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip epsilon
            value_clip: Whether to clip value function
            entropy_coef: Entropy coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use ('cpu' or 'cuda')
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        
        # Create actor and critic networks for each agent
        self.actors = nn.ModuleList([
            Actor(obs_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(num_agents)
        ])
        
        self.critics = nn.ModuleList([
            Critic(obs_dim, hidden_dim).to(self.device)
            for _ in range(num_agents)
        ])
        
        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]
        
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=lr_critic)
            for critic in self.critics
        ]
        
        # Experience buffer
        self.reset_buffer()
    
    def reset_buffer(self):
        """Reset experience buffer."""
        self.buffer = {
            'obs': [[] for _ in range(self.num_agents)],
            'actions': [[] for _ in range(self.num_agents)],
            'rewards': [[] for _ in range(self.num_agents)],
            'values': [[] for _ in range(self.num_agents)],
            'log_probs': [[] for _ in range(self.num_agents)],
            'dones': [[] for _ in range(self.num_agents)],
        }
    
    def select_actions(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select actions for all agents.
        
        Args:
            obs: Observations for all agents (num_agents, obs_dim)
            deterministic: Whether to use deterministic policy
        
        Returns:
            actions: Selected actions (num_agents, action_dim)
            log_probs: Log probabilities of actions (num_agents, 1)
        """
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions = []
        log_probs = []
        
        for i in range(self.num_agents):
            if deterministic:
                action = self.actors[i].get_action(obs_tensor[i:i+1], deterministic=True)
                actions.append(action.detach().cpu().numpy()[0])
                log_probs.append(np.array([0.0]))  # Dummy log prob for deterministic
            else:
                action, log_prob = self.actors[i].get_action(obs_tensor[i:i+1])
                actions.append(action.detach().cpu().numpy()[0])
                log_probs.append(log_prob.detach().cpu().numpy()[0])
        
        return np.array(actions), np.array(log_probs)
    
    def get_values(self, obs: np.ndarray) -> np.ndarray:
        """
        Get value estimates for all agents.
        
        Args:
            obs: Observations for all agents (num_agents, obs_dim)
        
        Returns:
            values: Value estimates (num_agents, 1)
        """
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        values = []
        
        for i in range(self.num_agents):
            value = self.critics[i](obs_tensor[i:i+1])
            values.append(value.detach().cpu().numpy()[0])
        
        return np.array(values)
    
    def store_transition(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        dones: np.ndarray
    ):
        """
        Store transition in buffer.
        
        Args:
            obs: Observations (num_agents, obs_dim)
            actions: Actions (num_agents, action_dim)
            rewards: Rewards (num_agents,)
            values: Value estimates (num_agents, 1)
            log_probs: Log probabilities (num_agents, 1)
            dones: Done flags (num_agents,)
        """
        for i in range(self.num_agents):
            self.buffer['obs'][i].append(obs[i])
            self.buffer['actions'][i].append(actions[i])
            self.buffer['rewards'][i].append(rewards[i])
            self.buffer['values'][i].append(values[i][0])
            self.buffer['log_probs'][i].append(log_probs[i][0])
            self.buffer['dones'][i].append(dones[i])
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_value: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards (T,)
            values: Value estimates (T,)
            next_value: Next value estimate (scalar)
            dones: Done flags (T,)
        
        Returns:
            advantages: GAE advantages (T,)
            returns: Returns (T,)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value_t = next_value * (1 - dones[t])
            else:
                next_value_t = values[t + 1] * (1 - dones[t])
            
            delta = rewards[t] + self.gamma * next_value_t - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self, next_obs: np.ndarray, epochs: int = 10, batch_size: int = 64):
        """
        Update policy and value networks using PPO.
        
        Args:
            next_obs: Next observations for computing next values (num_agents, obs_dim)
            epochs: Number of update epochs
            batch_size: Batch size for updates
        """
        # Get next values
        next_values = self.get_values(next_obs)
        
        # Process buffer for each agent
        for agent_id in range(self.num_agents):
            # Convert buffer to numpy arrays
            obs = np.array(self.buffer['obs'][agent_id])
            actions = np.array(self.buffer['actions'][agent_id])
            rewards = np.array(self.buffer['rewards'][agent_id])
            values = np.array(self.buffer['values'][agent_id])
            old_log_probs = np.array(self.buffer['log_probs'][agent_id])
            dones = np.array(self.buffer['dones'][agent_id])
            
            # Compute GAE
            advantages, returns = self.compute_gae(
                rewards, values, next_values[agent_id][0], dones
            )
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Convert to tensors
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device).unsqueeze(1)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
            returns_tensor = torch.FloatTensor(returns).to(self.device).unsqueeze(1)
            
            # Update for multiple epochs
            dataset_size = len(obs)
            indices = np.arange(dataset_size)
            
            for epoch in range(epochs):
                np.random.shuffle(indices)
                
                for start in range(0, dataset_size, batch_size):
                    end = start + batch_size
                    batch_indices = indices[start:end]
                    
                    # Get batch data
                    batch_obs = obs_tensor[batch_indices]
                    batch_actions = actions_tensor[batch_indices]
                    batch_old_log_probs = old_log_probs_tensor[batch_indices]
                    batch_advantages = advantages_tensor[batch_indices]
                    batch_returns = returns_tensor[batch_indices]
                    
                    # Update actor
                    new_log_probs, entropy = self.actors[agent_id].evaluate_actions(
                        batch_obs, batch_actions
                    )
                    
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                    
                    self.actor_optimizers[agent_id].zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actors[agent_id].parameters(), self.max_grad_norm
                    )
                    self.actor_optimizers[agent_id].step()
                    
                    # Update critic
                    new_values = self.critics[agent_id](batch_obs)
                    
                    if self.value_clip:
                        old_values = self.critics[agent_id](batch_obs).detach()
                        value_clipped = old_values + torch.clamp(
                            new_values - old_values, -self.clip_epsilon, self.clip_epsilon
                        )
                        value_loss1 = (new_values - batch_returns).pow(2)
                        value_loss2 = (value_clipped - batch_returns).pow(2)
                        value_loss = torch.max(value_loss1, value_loss2).mean()
                    else:
                        value_loss = (new_values - batch_returns).pow(2).mean()
                    
                    self.critic_optimizers[agent_id].zero_grad()
                    value_loss = self.value_coef * value_loss
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critics[agent_id].parameters(), self.max_grad_norm
                    )
                    self.critic_optimizers[agent_id].step()
        
        # Reset buffer after update
        self.reset_buffer()
    
    def save(self, filepath: str):
        """Save model checkpoints."""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers],
        }, filepath)
    
    def load(self, filepath: str):
        """Load model checkpoints."""
        checkpoint = torch.load(filepath, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(checkpoint['critics'][i])
        for i, opt in enumerate(self.actor_optimizers):
            opt.load_state_dict(checkpoint['actor_optimizers'][i])
        for i, opt in enumerate(self.critic_optimizers):
            opt.load_state_dict(checkpoint['critic_optimizers'][i])

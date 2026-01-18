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
        device: str = 'cpu',
        use_lstm: bool = True,
        sequence_length: int = 5,
        lstm_hidden_dim: int = 128
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
        self.use_lstm = use_lstm
        self.sequence_length = sequence_length
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # Create actor and critic networks for each agent
        self.actors = nn.ModuleList([
            Actor(obs_dim, action_dim, hidden_dim, lstm_hidden_dim, use_lstm, sequence_length).to(self.device)
            for _ in range(num_agents)
        ])
        
        self.critics = nn.ModuleList([
            Critic(obs_dim, hidden_dim, lstm_hidden_dim, use_lstm, sequence_length).to(self.device)
            for _ in range(num_agents)
        ])
        
        # Initialize LSTM hidden states for each agent (if using LSTM)
        if self.use_lstm:
            self.actor_hidden_states = [None for _ in range(num_agents)]
            self.critic_hidden_states = [None for _ in range(num_agents)]
        
        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]
        
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=lr_critic)
            for critic in self.critics
        ]
        
        # Observation history buffer for LSTM (maintains last sequence_length observations)
        # Initialize before reset_buffer() since reset_buffer() uses it
        if self.use_lstm:
            self.obs_history = [deque(maxlen=sequence_length) for _ in range(num_agents)]
        else:
            self.obs_history = None
        
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
        
        # Reset LSTM hidden states
        if self.use_lstm:
            self.actor_hidden_states = [None for _ in range(self.num_agents)]
            self.critic_hidden_states = [None for _ in range(self.num_agents)]
            # Clear observation history
            for i in range(self.num_agents):
                self.obs_history[i].clear()
    
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
        actions = []
        log_probs = []
        
        for i in range(self.num_agents):
            if self.use_lstm:
                # Update observation history
                self.obs_history[i].append(obs[i])
                
                # Build sequence tensor
                if len(self.obs_history[i]) < self.sequence_length:
                    # Pad with first observation if history is not full
                    seq_obs = list(self.obs_history[i])
                    while len(seq_obs) < self.sequence_length:
                        seq_obs.insert(0, seq_obs[0] if seq_obs else obs[i])
                    seq_obs = np.array(seq_obs)
                else:
                    seq_obs = np.array(list(self.obs_history[i]))
                
                seq_obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(self.device)  # (1, seq_len, obs_dim)
                
                if deterministic:
                    action, new_hidden = self.actors[i].get_action(seq_obs_tensor, deterministic=True, hidden_state=self.actor_hidden_states[i])
                    self.actor_hidden_states[i] = new_hidden
                    actions.append(action.detach().cpu().numpy()[0])
                    log_probs.append(np.array([0.0]))  # Dummy log prob for deterministic
                else:
                    action, log_prob, new_hidden = self.actors[i].get_action(seq_obs_tensor, deterministic=False, hidden_state=self.actor_hidden_states[i])
                    self.actor_hidden_states[i] = new_hidden
                    actions.append(action.detach().cpu().numpy()[0])
                    log_probs.append(log_prob.detach().cpu().numpy()[0])
            else:
                # Standard MLP
                obs_tensor = torch.FloatTensor(obs[i:i+1]).to(self.device)
                if deterministic:
                    action = self.actors[i].get_action(obs_tensor, deterministic=True)
                    actions.append(action.detach().cpu().numpy()[0])
                    log_probs.append(np.array([0.0]))
                else:
                    action, log_prob = self.actors[i].get_action(obs_tensor)
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
        values = []
        
        for i in range(self.num_agents):
            if self.use_lstm:
                # Use current observation history
                if len(self.obs_history[i]) < self.sequence_length:
                    # Pad with first observation if history is not full
                    seq_obs = list(self.obs_history[i])
                    while len(seq_obs) < self.sequence_length:
                        seq_obs.insert(0, seq_obs[0] if seq_obs else obs[i])
                    seq_obs = np.array(seq_obs)
                else:
                    seq_obs = np.array(list(self.obs_history[i]))
                
                seq_obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(self.device)  # (1, seq_len, obs_dim)
                value, new_hidden = self.critics[i](seq_obs_tensor, hidden_state=self.critic_hidden_states[i])
                self.critic_hidden_states[i] = new_hidden
                values.append(value.detach().cpu().numpy()[0])
            else:
                # Standard MLP
                obs_tensor = torch.FloatTensor(obs[i:i+1]).to(self.device)
                value = self.critics[i](obs_tensor)
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
            if self.use_lstm:
                # Build sequence observations for LSTM
                # For each timestep, we need the previous sequence_length observations
                seq_obs_list = []
                for t in range(len(obs)):
                    if t < self.sequence_length:
                        # Pad with first observation
                        seq = [obs[0]] * (self.sequence_length - t - 1) + list(obs[:t+1])
                    else:
                        seq = list(obs[t - self.sequence_length + 1:t + 1])
                    seq_obs_list.append(np.array(seq))
                obs_tensor = torch.FloatTensor(np.array(seq_obs_list)).to(self.device)  # (dataset_size, sequence_length, obs_dim)
            else:
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
                    if self.use_lstm:
                        new_log_probs, entropy, _ = self.actors[agent_id].evaluate_actions(
                            batch_obs, batch_actions, hidden_state=None
                        )
                    else:
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
                    if self.use_lstm:
                        new_values, _ = self.critics[agent_id](batch_obs, hidden_state=None)
                    else:
                        new_values = self.critics[agent_id](batch_obs)
                    
                    if self.value_clip:
                        if self.use_lstm:
                            old_values, _ = self.critics[agent_id](batch_obs, hidden_state=None)
                            old_values = old_values.detach()
                        else:
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

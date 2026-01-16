# --- networks.py ---
# Actor-Critic Networks for MAPPO

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Intersection.config import OBS_DIM


class Actor(nn.Module):
    """Actor network (policy network) for MAPPO."""
    
    def __init__(self, obs_dim=OBS_DIM, action_dim=2, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output: mean and std for continuous actions
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs):
        """Forward pass through actor network."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = torch.tanh(self.mean_head(x))  # Mean in [-1, 1]
        std = F.softplus(self.std_head(x)) + 1e-5  # Std > 0
        
        return mean, std
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy distribution."""
        mean, std = self.forward(obs)
        
        if deterministic:
            return mean
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Clip action to [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        
        return action, log_prob
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions and return log probs and entropy."""
        mean, std = self.forward(obs)
        
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, entropy


class Critic(nn.Module):
    """Critic network (value network) for MAPPO."""
    
    def __init__(self, obs_dim=OBS_DIM, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs):
        """Forward pass through critic network."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        
        return value

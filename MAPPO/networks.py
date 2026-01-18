# --- networks.py ---
# Actor-Critic Networks for MAPPO

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Intersection.config import OBS_DIM


class Actor(nn.Module):
    """Actor network (policy network) for MAPPO with LSTM."""
    
    def __init__(self, obs_dim=OBS_DIM, action_dim=2, hidden_dim=256, lstm_hidden_dim=128, use_lstm=True, sequence_length=5):
        super(Actor, self).__init__()
        
        self.use_lstm = use_lstm
        self.sequence_length = sequence_length
        
        # Input projection
        self.fc_input = nn.Linear(obs_dim, hidden_dim)
        
        if self.use_lstm:
            # LSTM layer
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
            # Post-LSTM layers
            self.fc2 = nn.Linear(lstm_hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        else:
            # Standard MLP layers
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
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1)
    
    def forward(self, obs, hidden_state=None):
        """
        Forward pass through actor network.
        
        Args:
            obs: Observation tensor
                - If use_lstm=False: (batch_size, obs_dim)
                - If use_lstm=True: (batch_size, sequence_length, obs_dim) or (batch_size, obs_dim) for single step
            hidden_state: LSTM hidden state tuple (h, c) if use_lstm=True
        
        Returns:
            mean: Action mean (batch_size, action_dim)
            std: Action std (batch_size, action_dim)
            hidden_state: Updated LSTM hidden state (if use_lstm=True)
        """
        if self.use_lstm:
            # Handle both sequence and single step inputs
            if len(obs.shape) == 2:
                # Single step: (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
                obs = obs.unsqueeze(1)
            
            batch_size = obs.shape[0]
            seq_len = obs.shape[1]
            
            # Project input
            x = F.relu(self.fc_input(obs))  # (batch_size, seq_len, hidden_dim)
            
            # LSTM
            if hidden_state is None:
                lstm_out, new_hidden = self.lstm(x)
            else:
                lstm_out, new_hidden = self.lstm(x, hidden_state)
            
            # Use last timestep output
            x = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_dim)
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        else:
            # Standard MLP
            x = F.relu(self.fc_input(obs))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            new_hidden = None
        
        mean = torch.tanh(self.mean_head(x))  # Mean in [-1, 1]
        std = F.softplus(self.std_head(x)) + 1e-5  # Std > 0
        
        if self.use_lstm:
            return mean, std, new_hidden
        else:
            return mean, std
    
    def get_action(self, obs, deterministic=False, hidden_state=None):
        """Sample action from policy distribution."""
        if self.use_lstm:
            mean, std, new_hidden = self.forward(obs, hidden_state)
        else:
            mean, std = self.forward(obs)
            new_hidden = None
        
        if deterministic:
            if self.use_lstm:
                return mean, new_hidden
            else:
                return mean
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Clip action to [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        
        if self.use_lstm:
            return action, log_prob, new_hidden
        else:
            return action, log_prob
    
    def evaluate_actions(self, obs, actions, hidden_state=None):
        """Evaluate actions and return log probs and entropy."""
        if self.use_lstm:
            mean, std, new_hidden = self.forward(obs, hidden_state)
        else:
            mean, std = self.forward(obs)
            new_hidden = None
        
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        if self.use_lstm:
            return log_probs, entropy, new_hidden
        else:
            return log_probs, entropy


class Critic(nn.Module):
    """Critic network (value network) for MAPPO with LSTM."""
    
    def __init__(self, obs_dim=OBS_DIM, hidden_dim=256, lstm_hidden_dim=128, use_lstm=True, sequence_length=5):
        super(Critic, self).__init__()
        
        self.use_lstm = use_lstm
        self.sequence_length = sequence_length
        
        # Input projection
        self.fc_input = nn.Linear(obs_dim, hidden_dim)
        
        if self.use_lstm:
            # LSTM layer
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
            # Post-LSTM layers
            self.fc2 = nn.Linear(lstm_hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        else:
            # Standard MLP layers
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
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1)
    
    def forward(self, obs, hidden_state=None):
        """
        Forward pass through critic network.
        
        Args:
            obs: Observation tensor
                - If use_lstm=False: (batch_size, obs_dim)
                - If use_lstm=True: (batch_size, sequence_length, obs_dim) or (batch_size, obs_dim) for single step
            hidden_state: LSTM hidden state tuple (h, c) if use_lstm=True
        
        Returns:
            value: State value (batch_size, 1)
            hidden_state: Updated LSTM hidden state (if use_lstm=True)
        """
        if self.use_lstm:
            # Handle both sequence and single step inputs
            if len(obs.shape) == 2:
                # Single step: (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
                obs = obs.unsqueeze(1)
            
            batch_size = obs.shape[0]
            seq_len = obs.shape[1]
            
            # Project input
            x = F.relu(self.fc_input(obs))  # (batch_size, seq_len, hidden_dim)
            
            # LSTM
            if hidden_state is None:
                lstm_out, new_hidden = self.lstm(x)
            else:
                lstm_out, new_hidden = self.lstm(x, hidden_state)
            
            # Use last timestep output
            x = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_dim)
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        else:
            # Standard MLP
            x = F.relu(self.fc_input(obs))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            new_hidden = None
        
        value = self.fc4(x)
        
        if self.use_lstm:
            return value, new_hidden
        else:
            return value

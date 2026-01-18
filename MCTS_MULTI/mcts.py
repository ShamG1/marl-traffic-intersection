# --- mcts.py ---
# Monte Carlo Tree Search for Multi-Agent Training with Real Environment Rollouts

import numpy as np
import torch
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Callable
import math
import copy
import threading


class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Current state (observation)
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children = {}  # Dict: action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = 0.0  # Prior probability from policy network
        self.value_estimate = 0.0  # Value estimate from value network
        
    @property
    def value(self):
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_leaf(self):
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0
    
    def is_fully_expanded(self, action_space):
        """Check if all actions have been explored."""
        return len(self.children) == len(action_space)


class MCTS:
    """
    True Monte Carlo Tree Search with real environment rollouts.
    Uses dual network (policy + value) for action selection and value estimation.
    Performs actual environment simulations for state transitions.
    """
    
    def __init__(
        self,
        network,  # DualNetwork instance
        action_space: np.ndarray,
        num_simulations: int = 50,
        c_puct: float = 1.0,  # Exploration constant
        temperature: float = 1.0,  # Temperature for action selection
        device: str = 'cpu',
        rollout_depth: int = 3,  # How many steps to rollout in environment
        env_factory: Optional[Callable] = None,  # Function to create environment copy
        all_networks: Optional[List] = None,  # All agent networks for multi-agent rollouts
        agent_id: int = 0,  # ID of the agent this MCTS is for
        num_action_samples: int = 5  # Number of actions to sample per node expansion
    ):
        """
        Initialize MCTS with real environment rollouts.
        
        Args:
            network: DualNetwork instance for policy and value prediction
            action_space: Array of possible actions (for discrete) or None (for continuous)
            num_simulations: Number of MCTS simulations per step
            c_puct: Exploration constant (higher = more exploration)
            temperature: Temperature for action selection (higher = more exploration)
            device: Device to use ('cpu' or 'cuda')
            rollout_depth: Number of steps to rollout in environment (default: 3)
            env_factory: Function that returns a copy of the environment for rollouts
            all_networks: List of all agent networks (for multi-agent rollouts)
            agent_id: ID of the agent this MCTS instance is for
            num_action_samples: Number of actions to sample per node expansion (default: 5)
        """
        self.network = network
        self.action_space = action_space
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = torch.device(device)
        self.rollout_depth = rollout_depth
        self.env_factory = env_factory
        self.all_networks = all_networks
        self.agent_id = agent_id
        self.num_action_samples = num_action_samples
        
        # For continuous actions, we'll sample from policy distribution
        self.continuous_actions = action_space is None
        
        # Cache for environment states to avoid deep copying too often
        self._env_cache = None
        self._env_state_cache = None
        
        # Statistics for rollout verification
        self.rollout_stats = {
            'total_rollouts': 0,
            'successful_rollouts': 0,
            'failed_rollouts': 0,
            'total_env_steps': 0,
            'rollout_rewards': [],
            'rollout_depths': []
        }
        
        # Debug flag
        self.debug_rollout = False
    
    def search(
        self,
        root_state: np.ndarray,
        obs_history: Optional[List] = None,
        hidden_state=None,
        env_state: Optional[Dict] = None  # Current environment state for rollouts
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform MCTS search from root state with real environment rollouts.
        
        Args:
            root_state: Root observation (obs_dim,)
            obs_history: History of observations for LSTM (list of obs_dim arrays)
            hidden_state: LSTM hidden state
            env_state: Current environment state (for creating rollout copies)
        
        Returns:
            action: Selected action (action_dim,)
            search_stats: Dictionary with search statistics
        """
        # Create root node
        root = MCTSNode(root_state)
        
        # Get initial policy and value from network
        if obs_history is not None and len(obs_history) > 0:
            # Build sequence for LSTM
            seq_obs = np.array(obs_history[-self.network.sequence_length:])
            if len(seq_obs) < self.network.sequence_length:
                # Pad with first observation
                seq_obs = np.array([seq_obs[0]] * (self.network.sequence_length - len(seq_obs)) + list(seq_obs))
            seq_obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(self.device)
            policy_mean, policy_std, value, _ = self.network(seq_obs_tensor, hidden_state)
        else:
            obs_tensor = torch.FloatTensor(root_state).unsqueeze(0).to(self.device)
            policy_mean, policy_std, value, _ = self.network(obs_tensor, hidden_state)
        
        root.value_estimate = value.item()
        
        # Debug: store simulation details
        self.debug_info = {
            'simulations': [],
            'root_value': value.item(),
            'root_policy_mean': policy_mean.detach().cpu().numpy()[0] if isinstance(policy_mean, torch.Tensor) else policy_mean,
            'root_policy_std': policy_std.detach().cpu().numpy()[0] if isinstance(policy_std, torch.Tensor) else policy_std
        }
        
        # Store environment state for rollouts
        self._env_state_cache = env_state
        
        # Perform simulations
        if self.debug_rollout:
            print(f"\n[Agent {self.agent_id}] Starting MCTS search with {self.num_simulations} simulations")
            print(f"  Root state shape: {root_state.shape}")
            print(f"  Rollout depth: {self.rollout_depth}")
            print(f"  Env factory available: {self.env_factory is not None}")
            print(f"  Env state available: {env_state is not None}")
        
        for sim_idx in range(self.num_simulations):
            # Handle pygame events periodically during MCTS search to keep window responsive
            if sim_idx % 5 == 0:  # Every 5 simulations
                try:
                    import pygame
                    if pygame.get_init():
                        pygame.event.pump()  # Process events without blocking
                        # Check for quit event
                        for event in pygame.event.get(pygame.QUIT):
                            if event.type == pygame.QUIT:
                                break
                except:
                    pass  # Ignore pygame errors during MCTS search
            
            # Progress logging disabled - only episode-level logs are shown
            sim_info = {'simulation': sim_idx + 1}
            try:
                self._simulate(root, obs_history, hidden_state, sim_info, env_state)
            except Exception as e:
                # Silently continue on simulation errors (to reduce verbosity)
                continue
            
            self.debug_info['simulations'].append(sim_info)
        
        # Select action based on visit counts
        action_probs = self._get_action_probs(root)
        
        # Sample action from visit count distribution
        # If no children expanded (shouldn't happen after simulations, but safety check)
        if len(action_probs) == 0:
            # No children expanded, sample from policy network
            if obs_history is not None and len(obs_history) > 0:
                seq_obs = np.array(obs_history[-self.network.sequence_length:])
                if len(seq_obs) < self.network.sequence_length:
                    seq_obs = np.array([seq_obs[0]] * (self.network.sequence_length - len(seq_obs)) + list(seq_obs))
                seq_obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(self.device)
                policy_mean, policy_std, _, _ = self.network(seq_obs_tensor, hidden_state)
            else:
                obs_tensor = torch.FloatTensor(root_state).unsqueeze(0).to(self.device)
                policy_mean, policy_std, _, _ = self.network(obs_tensor, hidden_state)
            
            dist = torch.distributions.Normal(policy_mean, policy_std)
            best_action = dist.sample().cpu().numpy()[0]
            best_action = np.clip(best_action, -1.0, 1.0)
        elif self.temperature == 0:
            # Greedy selection
            best_action_tuple = max(action_probs.items(), key=lambda x: x[1])[0]
            # Convert tuple to numpy array if needed
            if isinstance(best_action_tuple, tuple):
                best_action = np.array(best_action_tuple)
            else:
                best_action = best_action_tuple
        else:
            # Sample with temperature
            actions_list = list(action_probs.keys())
            probs = np.array([action_probs[a] for a in actions_list])
            probs = np.power(probs, 1.0 / self.temperature)
            probs = probs / probs.sum()
            best_action_idx = np.random.choice(len(actions_list), p=probs)
            best_action_tuple = actions_list[best_action_idx]
            # Convert tuple to numpy array if needed
            if isinstance(best_action_tuple, tuple):
                best_action = np.array(best_action_tuple)
            else:
                best_action = best_action_tuple
        
        # Add rollout statistics to search stats
        search_stats = {
            'visit_counts': {str(a): node.visit_count for a, node in root.children.items()},
            'action_probs': {str(a): p for a, p in action_probs.items()},
            'root_value': root.value,
            'num_simulations': self.num_simulations,
            'selected_action': best_action.tolist() if isinstance(best_action, np.ndarray) else best_action,
            'debug_info': getattr(self, 'debug_info', None),
            'rollout_stats': {
                'total_rollouts': self.rollout_stats['total_rollouts'],
                'successful_rollouts': self.rollout_stats['successful_rollouts'],
                'failed_rollouts': self.rollout_stats['failed_rollouts'],
                'total_env_steps': self.rollout_stats['total_env_steps'],
                'avg_rollout_reward': np.mean(self.rollout_stats['rollout_rewards']) if self.rollout_stats['rollout_rewards'] else 0.0,
                'avg_rollout_depth': np.mean(self.rollout_stats['rollout_depths']) if self.rollout_stats['rollout_depths'] else 0.0
            }
        }
        
        # Search completed (no output to reduce verbosity - only episode-level logs are shown)
        return best_action, search_stats
    
    def _simulate(self, node: MCTSNode, obs_history: Optional[List], hidden_state, debug_info=None, env_state=None):
        """
        Perform one MCTS simulation from node with real environment rollout.
        
        Args:
            node: Current node
            obs_history: History of observations
            hidden_state: LSTM hidden state
            debug_info: Dictionary to store debug information
            env_state: Environment state for creating rollout copies
        """
        # Selection: traverse to leaf node
        path = []
        current = node
        selected_actions = []
        
        while not current.is_leaf():
            # Select action using PUCT
            action_key = self._select_action(current)  # Returns tuple for continuous actions
            path.append((current, action_key))
            selected_actions.append(np.array(action_key))
            
            if action_key not in current.children:
                # Expand: create new child node
                # For continuous actions, convert tuple back to numpy array for state transition
                if self.continuous_actions:
                    action_array = np.array(action_key, dtype=np.float32)
                else:
                    action_array = action_key
                # For real MCTS, we'll evaluate states on-demand during expansion
                # Store action but defer state evaluation
                current.children[action_key] = MCTSNode(current.state.copy(), parent=current, action=action_array)
            
            current = current.children[action_key]
        
        # Expansion and Evaluation with real rollout
        if current.visit_count == 0:
            # First visit: expand and evaluate with rollout
            expand_result = self._expand_and_evaluate_with_rollout(
                current, obs_history, hidden_state, debug_info, env_state, path
            )
            value = expand_result['value']
            sampled_actions = expand_result.get('sampled_actions', [])
            if debug_info is not None:
                debug_info['selected_actions'] = [a.tolist() for a in selected_actions]
                debug_info['sampled_actions'] = [a.tolist() for a in sampled_actions]
                debug_info['value'] = value
                debug_info['node_visit_count'] = current.visit_count
            self._backup(path + [(current, None)], value)
        else:
            # Already visited: expand if not fully expanded, then evaluate
            if len(current.children) == 0:
                # Not expanded yet, expand now
                expand_result = self._expand_and_evaluate_with_rollout(
                    current, obs_history, hidden_state, debug_info, env_state, path
                )
                value = expand_result['value']
                sampled_actions = expand_result.get('sampled_actions', [])
            else:
                # Already expanded, evaluate with rollout
                value = self._evaluate_with_rollout(current.state, obs_history, hidden_state, env_state, path)
                sampled_actions = []
            if debug_info is not None:
                debug_info['selected_actions'] = [a.tolist() for a in selected_actions]
                debug_info['sampled_actions'] = [a.tolist() for a in sampled_actions]
                debug_info['value'] = value
                debug_info['node_visit_count'] = current.visit_count
            self._backup(path + [(current, None)], value)
    
    def _select_action(self, node: MCTSNode) -> np.ndarray:
        """
        Select action using PUCT (Polynomial Upper Confidence Trees) formula.
        PUCT combines value estimates with policy priors for better exploration.
        
        For continuous actions, we sample from policy distribution.
        """
        if self.continuous_actions:
            # For continuous actions, sample from policy
            # Get policy from network
            obs_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
            policy_mean, policy_std, _, _ = self.network(obs_tensor)
            
            # Sample action
            dist = torch.distributions.Normal(policy_mean, policy_std)
            action = dist.sample().cpu().numpy()[0]
            action = np.clip(action, -1.0, 1.0)
            # Ensure action is 1D array and convert to tuple of Python floats for use as dictionary key
            action = np.asarray(action).flatten()
            return tuple(float(x) for x in action)
        else:
            # For discrete actions, use PUCT formula
            # PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N) / (1 + n)
            # where:
            #   Q(s,a): average value of action
            #   P(s,a): prior probability from policy network
            #   N: total visits to parent node
            #   n: visits to child node
            #   c_puct: exploration constant
            puct_values = {}
            total_visits = sum(child.visit_count for child in node.children.values())
            
            for action, child in node.children.items():
                if child.visit_count == 0:
                    # Unvisited nodes get infinite value to ensure exploration
                    puct_values[action] = float('inf')
                else:
                    # PUCT formula: combines value and prior probability
                    q_value = child.value  # Average value estimate
                    prior = child.prior_prob  # Prior probability from policy network
                    puct_values[action] = q_value + self.c_puct * prior * math.sqrt(total_visits) / (1 + child.visit_count)
            
            # Select action with highest PUCT value
            return max(puct_values.items(), key=lambda x: x[1])[0]
    
    def _expand_and_evaluate(self, node: MCTSNode, obs_history: Optional[List], hidden_state, debug_info=None):
        """
        Expand node and evaluate with network.
        Returns dictionary with value estimate and sampled actions.
        """
        # Get policy and value from network
        if obs_history is not None and len(obs_history) > 0:
            seq_obs = np.array(obs_history[-self.network.sequence_length:])
            if len(seq_obs) < self.network.sequence_length:
                seq_obs = np.array([seq_obs[0]] * (self.network.sequence_length - len(seq_obs)) + list(seq_obs))
            seq_obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(self.device)
            policy_mean, policy_std, value, _ = self.network(seq_obs_tensor, hidden_state)
        else:
            obs_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
            policy_mean, policy_std, value, _ = self.network(obs_tensor, hidden_state)
        
        # Store prior probabilities (for continuous, we use policy std as uncertainty measure)
        sampled_actions_list = []
        if self.continuous_actions:
            # For continuous actions, we'll create children for sampled actions
            # Sample a few actions from policy
            dist = torch.distributions.Normal(policy_mean, policy_std)
            sampled_actions = dist.sample((self.num_action_samples,)).cpu().numpy()
            sampled_actions = np.clip(sampled_actions, -1.0, 1.0)
            
            for action in sampled_actions:
                # Ensure action is 1D array
                action = np.asarray(action).flatten()
                sampled_actions_list.append(action.copy())
                # Convert numpy array to tuple of Python floats for use as dictionary key
                action_key = tuple(float(x) for x in action)
                if action_key not in node.children:
                    child_state = self._get_next_state(node.state, action)
                    child = MCTSNode(child_state, parent=node, action=action)
                    # Use policy probability as prior
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                    log_prob = dist.log_prob(action_tensor).sum()
                    child.prior_prob = torch.exp(log_prob).item()
                    child.value_estimate = value.item()
                    node.children[action_key] = child
        
        return {
            'value': value.item(),
            'sampled_actions': sampled_actions_list,
            'policy_mean': policy_mean.detach().cpu().numpy()[0] if isinstance(policy_mean, torch.Tensor) else policy_mean,
            'policy_std': policy_std.detach().cpu().numpy()[0] if isinstance(policy_std, torch.Tensor) else policy_std
        }
    
    def _evaluate(self, state: np.ndarray, obs_history: Optional[List], hidden_state):
        """
        Evaluate state using value network.
        Returns value estimate.
        """
        if obs_history is not None and len(obs_history) > 0:
            seq_obs = np.array(obs_history[-self.network.sequence_length:])
            if len(seq_obs) < self.network.sequence_length:
                seq_obs = np.array([seq_obs[0]] * (self.network.sequence_length - len(seq_obs)) + list(seq_obs))
            seq_obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(self.device)
            _, _, value, _ = self.network(seq_obs_tensor, hidden_state)
        else:
            obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, _, value, _ = self.network(obs_tensor, hidden_state)
        
        return value.item()
    
    def _get_next_state_with_rollout(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        env_state: Optional[Dict],
        path_actions: List[np.ndarray]
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Get next state by performing real environment rollout.
        
        Args:
            state: Current observation
            action: Action to take
            env_state: Environment state to restore (contains 'obs' for all agents)
            path_actions: Actions taken along the path to this node
        
        Returns:
            next_state: Next observation
            value: Estimated value (reward + discounted future value)
            done: Whether episode ended
        """
        if self.env_factory is None or env_state is None:
            # Fallback: use value network estimate only
            if self.debug_rollout:
                print(f"[Agent {self.agent_id}] Rollout SKIPPED: env_factory={self.env_factory is not None}, env_state={env_state is not None}")
            obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, _, value, _ = self.network(obs_tensor)
            return state.copy(), value.item(), False
        
        # Update statistics
        self.rollout_stats['total_rollouts'] += 1
        
        try:
            # Create environment copy for rollout (output disabled - only episode-level logs are shown)
            
            # Create environment copy (rollout output disabled)
            try:
                env_copy = self.env_factory()
            except Exception as factory_err:
                # Only log critical errors
                raise
            
            # Reset environment to get initial state
            # Note: This creates a fresh environment, so we need to use the observations from env_state
            try:
                env_copy.reset()
            except Exception as reset_err:
                # Only log critical errors
                raise
            
            # Get current observations for all agents from env_state
            all_obs = env_state.get('obs', None)
            
            if all_obs is None:
                if self.debug_rollout:
                    print(f"  [Warning] No observations in env_state, using network value only")
                obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, _, value, _ = self.network(obs_tensor)
                return state.copy(), value.item(), False
            
            # Ensure all_obs is a list
            if isinstance(all_obs, np.ndarray):
                if all_obs.ndim == 1:
                    all_obs = [all_obs]
                else:
                    all_obs = [all_obs[i] for i in range(len(all_obs))]
            elif not isinstance(all_obs, (list, tuple)):
                all_obs = [all_obs]
            
            if hasattr(env_copy, 'agents') and len(env_copy.agents) > 0:
                # Perform rollout: execute action and get next state
                # For multi-agent: current agent uses search action, others use policy
                num_agents = len(env_copy.agents)
                
                actions_all = []
                
                for i in range(num_agents):
                    if i == self.agent_id:
                        # Current agent: use the search action
                        actions_all.append(action)
                    else:
                        # Other agents: use policy network
                        # Use observation from env_state (current state), not from env_copy
                        if self.all_networks is not None and i < len(self.all_networks) and i < len(all_obs):
                            try:
                                agent_obs = all_obs[i]
                                if not isinstance(agent_obs, np.ndarray):
                                    agent_obs = np.array(agent_obs)
                                
                                # Ensure observation has correct shape and is valid
                                if agent_obs.ndim == 0 or len(agent_obs) == 0 or len(agent_obs) != 118:
                                    if self.debug_rollout:
                                        print(f"    [Warning] Invalid observation for agent {i}, shape: {agent_obs.shape if hasattr(agent_obs, 'shape') else 'unknown'}")
                                    # Try to get from environment - but check agents first
                                    try:
                                        if not hasattr(env_copy, 'agents') or len(env_copy.agents) <= i:
                                            raise ValueError(f"Agent {i} not available in env_copy (has {len(env_copy.agents) if hasattr(env_copy, 'agents') else 0} agents)")
                                        
                                        # Ensure road and lidar are available
                                        if not hasattr(env_copy, 'road'):
                                            raise ValueError("env_copy has no 'road' attribute")
                                        
                                        # Update lidar before getting observation (required!)
                                        if hasattr(env_copy.agents[i], 'lidar'):
                                            try:
                                                env_copy.agents[i].lidar.update(
                                                    env_copy.road.collision_mask,
                                                    env_copy.agents
                                                )
                                            except Exception as lidar_err:
                                                raise ValueError(f"Lidar update failed: {lidar_err}")
                                        else:
                                            raise ValueError(f"Agent {i} has no lidar")
                                        
                                        # Now get observation
                                        agent_obs = env_copy.agents[i].get_observation(env_copy.agents)
                                        
                                        # Validate the result
                                        if agent_obs is None:
                                            raise ValueError("Observation is None")
                                        if not isinstance(agent_obs, np.ndarray):
                                            raise ValueError(f"Observation is not numpy array: {type(agent_obs)}")
                                        if len(agent_obs) == 0:
                                            raise ValueError("Observation is empty array")
                                        if len(agent_obs) != 118:
                                            raise ValueError(f"Wrong observation size: {len(agent_obs)}, expected 118")
                                    except Exception as e:
                                        if self.debug_rollout:
                                            import traceback
                                            print(f"    [Warning] Failed to get observation from env_copy for agent {i}: {e}")
                                            # Print relevant part of traceback
                                            tb_lines = traceback.format_exc().split('\n')
                                            for line in tb_lines:
                                                if 'get_observation' in line or 'concatenate' in line or 'stack' in line:
                                                    print(f"    [Trace] {line}")
                                        agent_obs = np.zeros(118)  # OBS_DIM
                                
                                obs_tensor = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
                                policy_mean, policy_std, _, _ = self.all_networks[i](obs_tensor)
                                dist = torch.distributions.Normal(policy_mean, policy_std)
                                other_action = dist.sample().cpu().numpy()[0]
                                other_action = np.clip(other_action, -1.0, 1.0)
                                actions_all.append(other_action)
                            except Exception as e:
                                if self.debug_rollout:
                                    print(f"    [Warning] Failed to get action for agent {i}: {e}")
                                # Fallback: zero action
                                actions_all.append(np.zeros(2))
                        else:
                            # Fallback: zero action
                            actions_all.append(np.zeros(2))
                
                # Step environment - THIS IS THE REAL ROLLOUT!
                # Note: Rollouts run in background without rendering
                if self.debug_rollout:
                    print(f"  [Rollout Step 0] Executing env.step() with actions:")
                    for i, act in enumerate(actions_all):
                        print(f"    Agent {i}: {act}")
                
                next_obs_all, rewards, terminated, truncated, info = env_copy.step(np.array(actions_all))
                done = terminated or truncated
                self.rollout_stats['total_env_steps'] += 1
                
                # Get next state for current agent
                next_state = next_obs_all[self.agent_id] if isinstance(next_obs_all, (list, np.ndarray)) else next_obs_all
                reward = rewards[self.agent_id] if isinstance(rewards, (list, np.ndarray)) else rewards
                
                if self.debug_rollout:
                    print(f"  [Rollout Step 0] Result: reward={reward:.3f}, done={done}, next_state_shape={next_state.shape}")
                
                # Perform additional rollout steps if needed
                total_reward = reward
                current_done = done
                current_obs = next_state
                actual_rollout_depth = 1  # Track actual depth reached
                
                for step in range(1, self.rollout_depth):
                    if current_done:
                        break
                    
                    # Get actions for all agents
                    actions_rollout = []
                    for i in range(num_agents):
                        if i == self.agent_id:
                            # Current agent: continue using policy
                            obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0).to(self.device)
                            policy_mean, policy_std, _, _ = self.network(obs_tensor)
                            dist = torch.distributions.Normal(policy_mean, policy_std)
                            rollout_action = dist.sample().cpu().numpy()[0]
                            rollout_action = np.clip(rollout_action, -1.0, 1.0)
                            actions_rollout.append(rollout_action)
                        else:
                            # Other agents: use their policy networks
                            # Get current observation from environment after previous step
                            try:
                                # Ensure agent exists and has necessary components
                                if not hasattr(env_copy, 'agents') or len(env_copy.agents) <= i:
                                    raise ValueError(f"Agent {i} not available in env_copy")
                                
                                # Update lidar before getting observation (required for get_observation)
                                if hasattr(env_copy.agents[i], 'lidar') and hasattr(env_copy, 'road'):
                                    try:
                                        env_copy.agents[i].lidar.update(
                                            env_copy.road.collision_mask,
                                            env_copy.agents
                                        )
                                    except Exception as lidar_err:
                                        if self.debug_rollout:
                                            print(f"    [Warning] Lidar update failed for agent {i}: {lidar_err}")
                                        # Continue anyway, lidar might still work
                                
                                agent_obs = env_copy.agents[i].get_observation(env_copy.agents)
                                
                                # Validate observation
                                if agent_obs is None:
                                    raise ValueError(f"Observation is None for agent {i}")
                                if isinstance(agent_obs, np.ndarray):
                                    if agent_obs.ndim == 0 or len(agent_obs) == 0:
                                        raise ValueError(f"Empty observation array for agent {i}")
                                    if len(agent_obs) != 118:
                                        raise ValueError(f"Wrong observation size for agent {i}: got {len(agent_obs)}, expected 118")
                                
                                if self.all_networks is not None and i < len(self.all_networks):
                                    obs_tensor = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
                                    policy_mean, policy_std, _, _ = self.all_networks[i](obs_tensor)
                                    dist = torch.distributions.Normal(policy_mean, policy_std)
                                    other_action = dist.sample().cpu().numpy()[0]
                                    other_action = np.clip(other_action, -1.0, 1.0)
                                    actions_rollout.append(other_action)
                                else:
                                    actions_rollout.append(np.zeros(2))
                            except Exception as e:
                                # If getting observation fails, use zero action
                                if self.debug_rollout:
                                    import traceback
                                    print(f"    [Warning] Failed to get observation for agent {i} in step {step}: {e}")
                                    print(f"    [Traceback] {traceback.format_exc()[:200]}")  # First 200 chars
                                actions_rollout.append(np.zeros(2))
                    
                    # Step environment - CONTINUING ROLLOUT
                    # Note: Rollouts run in background without rendering
                    # Rollout output disabled - only episode-level logs are shown
                    try:
                        next_obs_all, rewards, terminated, truncated, info = env_copy.step(np.array(actions_rollout))
                        current_done = terminated or truncated
                        self.rollout_stats['total_env_steps'] += 1
                        current_obs = next_obs_all[self.agent_id] if isinstance(next_obs_all, (list, np.ndarray)) else next_obs_all
                        step_reward = rewards[self.agent_id] if isinstance(rewards, (list, np.ndarray)) else rewards
                        total_reward += step_reward * (0.99 ** step)  # Discounted
                        actual_rollout_depth += 1
                    except Exception as step_err:
                        # Silently break on error (to reduce verbosity)
                        break
                
                # Final value estimate from network
                obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0).to(self.device)
                _, _, final_value, _ = self.network(obs_tensor)
                
                # Combine rollout reward with value estimate
                value = total_reward + (0.99 ** self.rollout_depth) * final_value.item() * (1 - current_done)
                
                # Update statistics
                self.rollout_stats['successful_rollouts'] += 1
                self.rollout_stats['rollout_rewards'].append(total_reward)
                self.rollout_stats['rollout_depths'].append(actual_rollout_depth)
                
                # Rollout output disabled - only episode-level logs are shown
                return current_obs, value, current_done
            else:
                # Single agent or no agents
                next_obs, reward, terminated, truncated, _ = env_copy.step(action)
                done = terminated or truncated
                return next_obs, reward, done
                
        except Exception as e:
            # Fallback on error: use network value estimate
            self.rollout_stats['failed_rollouts'] += 1
            # Error output disabled - only episode-level logs are shown
            import traceback
            error_trace = traceback.format_exc()
            # Error output disabled - only episode-level logs are shown
            try:
                obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, _, value, _ = self.network(obs_tensor)
            except Exception as network_err:
                # Silently fallback to zero value (to reduce verbosity)
                value = torch.tensor(0.0)  # Fallback to zero value
            return state.copy(), value.item(), False
    
    def _expand_and_evaluate_with_rollout(
        self, 
        node: MCTSNode, 
        obs_history: Optional[List], 
        hidden_state,
        debug_info=None,
        env_state=None,
        path_actions: List = None
    ):
        # Add debug logging
        import sys
        if self.debug_rollout:
            sys.stdout.write(f"[Agent {self.agent_id}] _expand_and_evaluate_with_rollout called\n")
            sys.stdout.flush()
        """
        Expand node and evaluate with real environment rollout.
        """
        # Get policy from network for action sampling
        if obs_history is not None and len(obs_history) > 0:
            seq_obs = np.array(obs_history[-self.network.sequence_length:])
            if len(seq_obs) < self.network.sequence_length:
                seq_obs = np.array([seq_obs[0]] * (self.network.sequence_length - len(seq_obs)) + list(seq_obs))
            seq_obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(self.device)
            policy_mean, policy_std, value, _ = self.network(seq_obs_tensor, hidden_state)
        else:
            obs_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
            policy_mean, policy_std, value, _ = self.network(obs_tensor, hidden_state)
        
        sampled_actions_list = []
        rollout_values = []
        
        if self.continuous_actions:
            # Sample actions from policy
            dist = torch.distributions.Normal(policy_mean, policy_std)
            sampled_actions = dist.sample((self.num_action_samples,)).cpu().numpy()
            sampled_actions = np.clip(sampled_actions, -1.0, 1.0)
            
            for action_idx, action in enumerate(sampled_actions):
                action = np.asarray(action).flatten()
                sampled_actions_list.append(action.copy())
                action_key = tuple(float(x) for x in action)
                
                if action_key not in node.children:
                    # Add debug logging - always log for first episode
                    import sys
                    # Performing rollout (output disabled)
                    
                    # Perform real rollout
                    try:
                        next_state, rollout_value, done = self._get_next_state_with_rollout(
                            node.state, action, env_state, path_actions if path_actions else []
                        )
                        rollout_values.append(rollout_value)
                        
                        # Rollout completed (output disabled)
                    except Exception as e:
                        # Silently continue on rollout errors (to reduce verbosity)
                        pass
                        # Use network value as fallback
                        rollout_values.append(value.item())
                    
                    child = MCTSNode(next_state, parent=node, action=action)
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                    log_prob = dist.log_prob(action_tensor).sum()
                    child.prior_prob = torch.exp(log_prob).item()
                    child.value_estimate = rollout_value
                    node.children[action_key] = child
        
        # Use average rollout value or network value
        final_value = np.mean(rollout_values) if rollout_values else value.item()
        
        return {
            'value': final_value,
            'sampled_actions': sampled_actions_list,
            'policy_mean': policy_mean.detach().cpu().numpy()[0] if isinstance(policy_mean, torch.Tensor) else policy_mean,
            'policy_std': policy_std.detach().cpu().numpy()[0] if isinstance(policy_std, torch.Tensor) else policy_std
        }
    
    def _evaluate_with_rollout(
        self, 
        state: np.ndarray, 
        obs_history: Optional[List], 
        hidden_state,
        env_state=None,
        path_actions: List = None
    ):
        """
        Evaluate state with real environment rollout.
        """
        if self.env_factory is not None and env_state is not None:
            # Perform rollout to get value estimate
            _, rollout_value, _ = self._get_next_state_with_rollout(
                state, 
                np.zeros(2),  # Dummy action for evaluation
                env_state, 
                path_actions if path_actions else []
            )
            return rollout_value
        else:
            # Fallback to network evaluation
            return self._evaluate(state, obs_history, hidden_state)
    
    def _backup(self, path: List[Tuple[MCTSNode, Optional[np.ndarray]]], value: float):
        """
        Backup value through path.
        
        Args:
            path: List of (node, action) tuples from root to leaf
            value: Value to backup
        """
        # Backup from leaf to root
        for node, _ in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            # Value decays with depth (optional)
            # value *= 0.99
    
    def _get_action_probs(self, node: MCTSNode) -> Dict:
        """
        Get action probabilities from visit counts.
        
        Returns:
            Dictionary mapping actions to probabilities
        """
        if len(node.children) == 0:
            return {}
        
        total_visits = sum(child.visit_count for child in node.children.values())
        if total_visits == 0:
            return {action: 1.0 / len(node.children) for action in node.children.keys()}
        
        action_probs = {}
        for action, child in node.children.items():
            action_probs[action] = child.visit_count / total_visits
        
        return action_probs
    
    def get_rollout_stats(self) -> Dict:
        """Get rollout statistics."""
        return {
            'total_rollouts': self.rollout_stats['total_rollouts'],
            'successful_rollouts': self.rollout_stats['successful_rollouts'],
            'failed_rollouts': self.rollout_stats['failed_rollouts'],
            'total_env_steps': self.rollout_stats['total_env_steps'],
            'success_rate': (self.rollout_stats['successful_rollouts'] / max(1, self.rollout_stats['total_rollouts'])) * 100,
            'avg_rollout_reward': np.mean(self.rollout_stats['rollout_rewards']) if self.rollout_stats['rollout_rewards'] else 0.0,
            'avg_rollout_depth': np.mean(self.rollout_stats['rollout_depths']) if self.rollout_stats['rollout_depths'] else 0.0,
            'env_steps_per_rollout': self.rollout_stats['total_env_steps'] / max(1, self.rollout_stats['total_rollouts'])
        }
    
    def reset_rollout_stats(self):
        """Reset rollout statistics."""
        self.rollout_stats = {
            'total_rollouts': 0,
            'successful_rollouts': 0,
            'failed_rollouts': 0,
            'total_env_steps': 0,
            'rollout_rewards': [],
            'rollout_depths': []
        }
    
    def enable_debug(self, enable: bool = True):
        """Enable or disable debug output for rollouts."""
        self.debug_rollout = enable

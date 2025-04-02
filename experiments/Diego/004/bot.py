import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math


# Simplified Rainbow DQN - focused on speed and efficiency
class SimplifiedRainbowDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimplifiedRainbowDQN, self).__init__()

        # Simpler architecture with fewer parameters
        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Dueling architecture with simplified structure
        self.advantage_stream = nn.Linear(64, output_dim)
        self.value_stream = nn.Linear(64, 1)

        # Initialize weights with Xavier
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state_dict):
        # Combine all inputs into a single tensor
        location = state_dict['location']
        status = state_dict['status']
        rays = state_dict['rays']
        relative_pos = state_dict.get('relative_pos', torch.zeros_like(location))
        time_features = state_dict.get('time_features', torch.zeros((location.shape[0], 2), device=location.device))

        # Concatenate all inputs
        combined = torch.cat([location, status, rays, relative_pos, time_features], dim=1)

        # Process through feature layers
        features = self.feature_layer(combined)

        # Dueling architecture
        advantage = self.advantage_stream(features)
        value = self.value_stream(features).expand(-1, advantage.size(1))

        # Combine value and advantage
        q_values = value + advantage - advantage.mean(1, keepdim=True)

        return q_values


# Simpler Experience Replay Buffer with prioritization
class PrioritizedBuffer:
    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        # Use max priority for new experiences
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return [], [], []

        # Calculate sampling probabilities
        probabilities = self.priorities[:len(self.buffer)] ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        weights = torch.FloatTensor(weights)

        batch = [self.buffer[idx] for idx in indices]

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


# Main Agent class
class SimplifiedRainbowAgent:
    def __init__(self, action_size=16, input_dim=38):
        self.action_size = action_size
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else
                                   "cpu")

        print(f"Using device: {self.device}")

        # Hyperparameters optimized for faster learning
        self.gamma = 0.95  # Slightly reduced discount factor for faster value propagation
        self.learning_rate = 0.0005  # Increased learning rate
        self.batch_size = 32  # Smaller batch size for faster updates
        self.min_memory_size = 500  # Reduced memory requirement before training
        self.target_update_freq = 500  # More frequent target updates
        self.train_freq = 1  # Train after every step
        self.steps = 0

        # Create networks
        self.model = SimplifiedRainbowDQN(input_dim, action_size).to(self.device)
        self.target_model = SimplifiedRainbowDQN(input_dim, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Add aliases for compatibility with main.py
        self.online_net = self.model
        self.target_net = self.target_model

        # Optimizer with higher learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Epsilon-greedy exploration (for simplicity instead of noisy nets)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Memory with prioritization
        self.memory = PrioritizedBuffer(capacity=20000)  # Smaller memory for efficiency

        # State tracking
        self.last_state = None
        self.last_action = None
        self.training_started = False

        # Time tracking features
        self.time_since_last_shot = 0
        self.time_alive = 0

    def normalize_state(self, info):
        """Normalize state values to improve learning stability"""
        try:
            # Regular state components
            state = {
                'location': torch.tensor([
                    info['location'][0] / 1280.0,  # Normalize by world width
                    info['location'][1] / 1280.0  # Normalize by world height
                ], dtype=torch.float32),

                'status': torch.tensor([
                    info['rotation'] / 360.0,  # Normalize rotation
                    info['current_ammo'] / 30.0  # Normalize ammo
                ], dtype=torch.float32),

                'rays': []
            }

            # Process rays
            ray_data = []
            for ray in info.get('rays', []):
                if isinstance(ray, list) and len(ray) == 3:
                    start_pos, end_pos = ray[0]
                    distance = ray[1] if ray[1] is not None else 1500  # Max vision distance
                    hit_type = ray[2]

                    # Normalize positions and distance
                    ray_data.extend([
                        start_pos[0] / 1280.0,
                        start_pos[1] / 1280.0,
                        end_pos[0] / 1280.0,
                        end_pos[1] / 1280.0,
                        distance / 1500.0,  # Normalize by max vision distance
                        1.0 if hit_type == "player" else 0.5 if hit_type == "object" else 0.0
                    ])

            # Pad rays if necessary
            while len(ray_data) < 30:  # 5 rays * 6 features
                ray_data.extend([0.0] * 6)

            state['rays'] = torch.tensor(ray_data[:30], dtype=torch.float32)

            # Add relative position to opponent if available
            if 'closest_opponent' in info:
                opponent_pos = info['closest_opponent']
                rel_x = (opponent_pos[0] - info['location'][0]) / 1280.0
                rel_y = (opponent_pos[1] - info['location'][1]) / 1280.0
                state['relative_pos'] = torch.tensor([rel_x, rel_y], dtype=torch.float32)
            else:
                state['relative_pos'] = torch.tensor([0.0, 0.0], dtype=torch.float32)

            # Add time-based features
            state['time_features'] = torch.tensor([
                self.time_since_last_shot / 100.0,  # Normalize time since last shot
                self.time_alive / 2400.0  # Normalize time alive by max episode length
            ], dtype=torch.float32)

            # Update time tracking
            self.time_alive += 1
            if info.get('shot_fired', False):
                self.time_since_last_shot = 0
            else:
                self.time_since_last_shot += 1

            return state

        except Exception as e:
            print(f"Error in normalize_state: {e}")
            print(f"Info received: {info}")
            raise

    def action_to_dict(self, action):
        """Enhanced action space with more granular rotation"""
        movement_directions = ["forward", "right", "down", "left"]
        rotation_angles = [-30, -5, -1, 0, 1, 5, 30]

        # Basic movement commands
        commands = {
            "forward": False,
            "right": False,
            "down": False,
            "left": False,
            "rotate": 0,
            "shoot": False
        }

        # determine block (no-shoot vs shoot)
        if action < 28:
            shoot = False
            local_action = action  # 0..27
        else:
            shoot = True
            local_action = action - 28  # 0..27

        movement_idx = local_action // 7  # 0..3
        angle_idx = local_action % 7  # 0..6

        direction = movement_directions[movement_idx]
        commands[direction] = True
        commands["rotate"] = rotation_angles[angle_idx]
        commands["shoot"] = shoot

        return commands

    def act(self, info):
        try:
            state = self.normalize_state(info)

            # Convert state dict to tensors and add batch dimension
            state_tensors = {
                k: v.unsqueeze(0).to(self.device) for k, v in state.items()
            }

            # Epsilon-greedy action selection
            if random.random() <= self.epsilon:
                action = random.randrange(self.action_size)
            else:
                with torch.no_grad():
                    q_values = self.model(state_tensors)
                    action = torch.argmax(q_values).item()

            self.last_state = state
            self.last_action = action
            return self.action_to_dict(action)

        except Exception as e:
            print(f"Error in act: {e}")
            # Return safe default action
            return {"forward": False, "right": False, "down": False, "left": False, "rotate": 0, "shoot": False}

    def remember(self, reward, next_info, done):
        try:
            next_state = self.normalize_state(next_info)

            # Store experience in memory
            self.memory.add(self.last_state, self.last_action, reward, next_state, done)

            # Increment step counter
            self.steps += 1

            # Check if training should start
            if len(self.memory) >= self.min_memory_size and not self.training_started:
                print(f"Starting training with {len(self.memory)} samples in memory")
                self.training_started = True

            # Perform learning step
            if self.training_started and self.steps % self.train_freq == 0:
                self.learn()

                # Print training progress periodically
                if self.steps % 1000 == 0:
                    print(f"Step {self.steps}, epsilon: {self.epsilon:.4f}")

            # Update target network
            if self.steps > 0 and self.steps % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                # Update target_net alias too
                self.target_net = self.target_model
                print(f"Updated target network at step {self.steps}")

            # Reset time alive if episode done
            if done:
                self.time_alive = 0

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        except Exception as e:
            print(f"Error in remember: {e}")

    def learn(self):
        """Simplified learning step with Double Q-learning and PER"""
        if len(self.memory) < self.batch_size:
            return

        try:
            # Sample batch from memory
            batch, indices, weights = self.memory.sample(self.batch_size)

            # Prepare batch data
            states = {
                'location': torch.stack([t[0]['location'] for t in batch]).to(self.device),
                'status': torch.stack([t[0]['status'] for t in batch]).to(self.device),
                'rays': torch.stack([t[0]['rays'] for t in batch]).to(self.device),
                'relative_pos': torch.stack([t[0].get('relative_pos', torch.zeros(2)) for t in batch]).to(self.device),
                'time_features': torch.stack([t[0].get('time_features', torch.zeros(2)) for t in batch]).to(self.device)
            }

            next_states = {
                'location': torch.stack([t[3]['location'] for t in batch]).to(self.device),
                'status': torch.stack([t[3]['status'] for t in batch]).to(self.device),
                'rays': torch.stack([t[3]['rays'] for t in batch]).to(self.device),
                'relative_pos': torch.stack([t[3].get('relative_pos', torch.zeros(2)) for t in batch]).to(self.device),
                'time_features': torch.stack([t[3].get('time_features', torch.zeros(2)) for t in batch]).to(self.device)
            }

            actions = torch.tensor([t[1] for t in batch], dtype=torch.long).to(self.device)
            rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32).to(self.device)
            dones = torch.tensor([t[4] for t in batch], dtype=torch.float32).to(self.device)
            weights = weights.to(self.device)

            # Get current Q values
            q_values = self.model(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Get next Q values with Double Q-learning
            with torch.no_grad():
                # Select actions using online network
                next_actions = self.model(next_states).max(1)[1]
                # Evaluate using target network
                next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                # Calculate target
                target = rewards + (1 - dones) * self.gamma * next_q_values

            # Calculate TD errors for updating priorities
            td_errors = torch.abs(q_values - target).detach().cpu().numpy()

            # Update priorities
            self.memory.update_priorities(indices, td_errors + 1e-6)  # Small constant to avoid zero priority

            # Calculate loss
            loss = (weights * F.smooth_l1_loss(q_values, target, reduction='none')).mean()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()

        except Exception as e:
            print(f"Error in learn: {e}")

    def reset_for_new_episode(self):
        """Reset episode-specific variables"""
        self.time_alive = 0
        self.time_since_last_shot = 0

    def get_hyperparameters(self):
        """Return current hyperparameters for logging"""
        return {
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epsilon": self.epsilon,
            "steps": self.steps,
            "model_input_dim": self.input_dim,
            "action_size": self.action_size
        }

    def save_to_dict(self):
        """Return a checkpoint dictionary of the training state"""
        return {
            'online_net_state_dict': self.model.state_dict(),  # Changed from model_state_dict to match main.py
            'target_net_state_dict': self.target_model.state_dict(),  # Changed from target_model_state_dict
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'hyperparameters': self.get_hyperparameters()
        }

    def load_from_dict(self, checkpoint_dict, map_location=None):
        """Load from checkpoint dictionary"""
        if map_location is None:
            map_location = self.device

        # Load model weights
        if 'online_net_state_dict' in checkpoint_dict:
            # New format with online_net_state_dict
            self.model.load_state_dict(checkpoint_dict['online_net_state_dict'])
            self.target_model.load_state_dict(checkpoint_dict['target_net_state_dict'])
        elif 'model_state_dict' in checkpoint_dict:
            # Old format with model_state_dict
            self.model.load_state_dict(checkpoint_dict['model_state_dict'])
            self.target_model.load_state_dict(checkpoint_dict['target_model_state_dict'])
        else:
            print("Warning: Unexpected checkpoint format")

        # Try to load optimizer state
        try:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            if map_location != 'cpu':
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(map_location)
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
            print("Continuing with fresh optimizer but keeping model weights")

        # Load training progress
        self.epsilon = checkpoint_dict.get('epsilon', self.epsilon)
        self.steps = checkpoint_dict.get('steps', 0)

        # Move models to correct device
        self.device = torch.device(map_location) if isinstance(map_location, str) else map_location
        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)

        # Update aliases
        self.online_net = self.model
        self.target_net = self.target_model

        print(f"Model loaded from checkpoint at step {self.steps}")


# For compatibility with the original code
RainbowDQNAgent = SimplifiedRainbowAgent
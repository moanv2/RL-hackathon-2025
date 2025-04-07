import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math


# Rainbow DQN Network with Dueling Architecture and Noisy Layers
class RainbowDQN(nn.Module):
    def __init__(self, input_dim, output_dim, atom_size=51, support_min=-10, support_max=10):
        super(RainbowDQN, self).__init__()
        self.support = torch.linspace(support_min, support_max,
                                      atom_size).cuda() if torch.cuda.is_available() else torch.linspace(support_min,
                                                                                                         support_max,
                                                                                                         atom_size)
        self.atom_size = atom_size
        self.output_dim = output_dim

        # Feature extraction shared layers
        self.feature_layer = nn.Sequential(
            NoisyLinear(input_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, 128),
            nn.ReLU(),
        )

        # Dueling architecture - separate advantage and value streams
        self.advantage_hidden = NoisyLinear(128, 128)
        self.advantage_output = NoisyLinear(128, output_dim * atom_size)

        self.value_hidden = NoisyLinear(128, 128)
        self.value_output = NoisyLinear(128, atom_size)

        # Initialize weights
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
        advantage_hidden = F.relu(self.advantage_hidden(features))
        advantage = self.advantage_output(advantage_hidden).view(-1, self.output_dim, self.atom_size)

        value_hidden = F.relu(self.value_hidden(features))
        value = self.value_output(value_hidden).view(-1, 1, self.atom_size)

        # Combine value and advantage streams
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Apply softmax to get probability distribution
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # Avoid NaNs

        return dist

    def reset_noise(self):
        """Reset noise for all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def act(self, state_dict, epsilon=0.0):
        """Get action based on state with option for epsilon-greedy"""
        with torch.no_grad():
            # Get distribution over atoms
            dist = self(state_dict)

            # Calculate expected value
            expected_value = (dist * self.support).sum(2)

            # Epsilon-greedy action selection
            if random.random() > epsilon:
                action = expected_value.argmax(1).item()
            else:
                action = random.randrange(self.output_dim)
            return action





# Noisy Linear Layer for exploration
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.zeros((out_features, in_features)))
        self.weight_sigma = nn.Parameter(torch.zeros((out_features, in_features)))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.zeros(out_features))

        # Register buffer for noise
        self.register_buffer('weight_epsilon', torch.zeros((out_features, in_features)))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        # Initialize parameters
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        # Sample noise for weights and bias
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Outer product
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        # Sample from standard normal and transform
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


# Experience Replay with Sumtree for Prioritized Experience Replay
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        # Propagate changes up the tree
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        # Find sample based on priority
        left = 2 * idx + 1
        right = left + 1

        # Leaf node
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        # Return sum of priorities
        return self.tree[0]

    def add(self, priority, data):
        # Add new sample with given priority
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        # Update priority for index
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        # Get sample based on cumulative priority s
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]


# Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = beta_increment  # Beta annealing
        self.epsilon = epsilon  # Small constant to prevent zero priority
        self.max_priority = 1.0  # Initial max priority

    def add(self, experience):
        # Add experience with max priority
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size):
        # Sample batch based on priorities
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)  # Anneal beta

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, experience = self.tree.get(s)

            batch.append(experience)
            indices.append(idx)
            priorities.append(priority)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probabilities) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        # Update priorities after learning
        for idx, priority in zip(indices, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# N-step Return Helper
class NStepBuffer:
    def __init__(self, n_step=3, gamma=0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def get(self):
        if len(self.buffer) < self.n_step:
            return None

        obs, action, reward, next_obs, done = self.buffer[0]

        # Calculate n-step return
        for i in range(1, self.n_step):
            # If terminal state encountered before n steps
            if self.buffer[i][4]:
                reward += (self.gamma ** i) * self.buffer[i][2]
                next_obs = self.buffer[i][3]
                done = True
                break

            # Add discounted rewards
            reward += (self.gamma ** i) * self.buffer[i][2]
            next_obs = self.buffer[i][3]

        return obs, action, reward, next_obs, done


# Main Rainbow DQN Agent
class RainbowDQNAgent:
    def __init__(self, action_size=16, input_dim=38):
        self.action_size = action_size
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else
                                   "cpu")

        print(f"Using device: {self.device}")

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.min_memory_size = 1000
        self.target_update_freq = 1000
        self.train_freq = 4
        self.steps = 0

        # Distributional RL parameters
        self.atom_size = 51
        self.v_min = -10
        self.v_max = 10
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        # Prioritized Experience Replay parameters
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.epsilon_pri = 0.01

        # N-step learning
        self.n_step = 3
        self.n_step_buffer = NStepBuffer(n_step=self.n_step, gamma=self.gamma)

        # Time tracking features
        self.time_since_last_shot = 0
        self.time_alive = 0

        # Exploration parameters
        self.exploration_bonus = 0.1
        self.visited_positions = {}
        self.position_resolution = 50

        # Create networks - online and target
        self.model = RainbowDQN(self.input_dim, self.action_size, self.atom_size, self.v_min, self.v_max).to(self.device)
        self.target_model = RainbowDQN(self.input_dim, self.action_size, self.atom_size, self.v_min, self.v_max).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Target network always in eval mode

        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # PER Memory
        self.memory = PrioritizedReplayBuffer(
            capacity=100000,
            alpha=self.alpha,
            beta=self.beta,
            beta_increment=self.beta_increment,
            epsilon=self.epsilon_pri
        )

        self.last_state = None
        self.last_action = None
        self.training_started = False

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

            # Use noisy network exploration instead of epsilon-greedy
            action = self.model.act(state_tensors)

            self.last_state = state
            self.last_action = action
            action_dict = self.action_to_dict(action)

            if info["rays"][2][-1] == "player":
                action_dict['shoot'] = True
            else:
                action_dict['shoot'] = False

            return action_dict

        except Exception as e:
            print(f"Error in act: {e}")
            # Return safe default action
            return {"forward": False, "right": False, "down": False, "left": False, "rotate": 0, "shoot": False}

    def remember(self, reward, next_info, done):
        try:
            next_state = self.normalize_state(next_info)

            # Calculate exploration bonus based on position novelty
            pos_x = int(next_state['location'][0].item() * self.position_resolution)
            pos_y = int(next_state['location'][1].item() * self.position_resolution)
            grid_pos = (pos_x, pos_y)

            # Add exploration bonus for less visited areas
            exploration_bonus = 0
            if grid_pos in self.visited_positions:
                self.visited_positions[grid_pos] += 1
                visit_count = self.visited_positions[grid_pos]
                exploration_bonus = self.exploration_bonus / math.sqrt(visit_count)
            else:
                self.visited_positions[grid_pos] = 1
                exploration_bonus = self.exploration_bonus

            # Add exploration bonus to reward
            reward += exploration_bonus

            # Add to n-step buffer
            self.n_step_buffer.add(self.last_state, self.last_action, reward, next_state, done)

            # Get n-step return sample if available
            n_step_sample = self.n_step_buffer.get()
            if n_step_sample:
                self.memory.add(n_step_sample)

            # Increment step counter
            self.steps += 1

            # Check if training should start
            if len(self.memory) >= self.min_memory_size and not self.training_started:
                print(f"Starting training with {len(self.memory)} samples in memory")
                self.training_started = True

            # Perform learning step
            if self.training_started and self.steps % self.train_freq == 0:
                self.learn()

                # Reset noise in noisy layers periodically
                if self.steps % 100 == 0:
                    self.model.reset_noise()

                # Print training progress periodically
                if self.steps % 1000 == 0:
                    print(f"Step {self.steps}")

            # Update target network
            if self.steps > 0 and self.steps % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"Updated target network at step {self.steps}")

            # Reset time alive if episode done
            if done:
                self.time_alive = 0

        except Exception as e:
            print(f"Error in remember: {e}")

    def learn(self):
        """Rainbow DQN learning step with distributional RL and PER"""
        if len(self.memory) < self.batch_size:
            return

        try:
            # Sample batch from PER
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
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

            # Get current distribution
            current_dist = self.model(states)
            current_dist = current_dist[range(self.batch_size), actions]

            # Get target distribution
            with torch.no_grad():
                # Double Q-learning - select best actions using online network
                next_q_dist = self.model(next_states)
                next_q = (next_q_dist * self.support).sum(2)
                next_actions = next_q.argmax(1)

                # Get target distribution for those actions
                target_dist = self.target_model(next_states)
                target_dist = target_dist[range(self.batch_size), next_actions]

                # Apply distributional Bellman operator
                target_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (self.gamma ** self.n_step) * self.support
                target_z = target_z.clamp(min=self.v_min, max=self.v_max)

                # Project onto support
                b = (target_z - self.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()

                # Handle corner cases where b is exactly an integer
                l[(u > 0) * (l == u)] -= 1
                u[(l < (self.atom_size - 1)) * (l == u)] += 1

                # Distribute probability mass
                batch_dim = list(range(self.batch_size))
                target_b = torch.zeros(self.batch_size, self.atom_size, device=self.device)

                for i in range(self.batch_size):
                    for j in range(self.atom_size):
                        target_b[i, l[i, j]] += target_dist[i, j] * (u[i, j] - b[i, j])
                        target_b[i, u[i, j]] += target_dist[i, j] * (b[i, j] - l[i, j])

            # Calculate Cross-Entropy Loss
            kl_loss = -(target_b * current_dist.log()).sum(1)
            weighted_loss = (weights * kl_loss).mean()

            # Optimize
            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()

            # Update priorities
            priorities = kl_loss.detach().cpu().numpy()
            self.memory.update_priorities(indices, priorities)

        except Exception as e:
            print(f"Error in learn: {e}")

    def reset_for_new_episode(self):
        """Reset episode-specific variables"""
        self.time_alive = 0
        self.time_since_last_shot = 0
        self.n_step_buffer = NStepBuffer(n_step=self.n_step, gamma=self.gamma)
        self.model.reset_noise()  # Reset noise in all noisy layers

        # Curriculum learning - adapt exploration as training progresses
        if self.steps > 1000000:
            self.exploration_bonus = 0.05
        elif self.steps > 500000:
            self.exploration_bonus = 0.08

    def get_hyperparameters(self):
        """Return current hyperparameters for logging"""
        return {
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "n_step": self.n_step,
            "atom_size": self.atom_size,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "alpha": self.alpha,
            "beta": self.memory.beta,
            "steps": self.steps,
            "model_input_dim": self.input_dim,
            "action_size": self.action_size
        }

    def save_to_dict(self):
        """Return a checkpoint dictionary of the training state"""
        return {
            'online_net_state_dict': self.model.state_dict(),
            'target_net_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'hyperparameters': self.get_hyperparameters()
        }

    def load_from_dict(self, checkpoint_dict, map_location=None):
        """Load from checkpoint dictionary"""
        if map_location is None:
            map_location = self.device

        # Load model weights
        self.model.load_state_dict(checkpoint_dict['online_net_state_dict'])
        self.target_model.load_state_dict(checkpoint_dict['target_net_state_dict'])

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
        self.steps = checkpoint_dict.get('steps', 0)

        # Move models to correct device
        self.device = torch.device(map_location) if isinstance(map_location, str) else map_location
        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)

        print(f"Model loaded from checkpoint at step {self.steps}")
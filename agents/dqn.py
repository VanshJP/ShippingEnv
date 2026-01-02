"""Deep Q-Network (DQN) agent for the shipping environment."""

import random
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import BaseAgent, TrainingResult
from utils.constants import Initial, TrainingDefaults
from utils.preprocessing import (
    get_action_space_size,
    map_action_to_env_action,
    preprocess_state,
)


class DQNNetwork(nn.Module):
    """Neural network architecture for DQN."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent implementation.

    Uses experience replay and target network for stable learning.
    """

    def __init__(
        self,
        env,
        learning_rate: float = TrainingDefaults.DQN_LEARNING_RATE,
        gamma: float = TrainingDefaults.DQN_GAMMA,
        epsilon: float = TrainingDefaults.DQN_EPSILON,
        epsilon_min: float = TrainingDefaults.DQN_EPSILON_MIN,
        epsilon_decay: float = TrainingDefaults.DQN_EPSILON_DECAY,
        memory_size: int = TrainingDefaults.DQN_MEMORY_SIZE,
        batch_size: int = TrainingDefaults.DQN_BATCH_SIZE,
        target_update_freq: int = TrainingDefaults.DQN_TARGET_UPDATE_FREQ,
        hidden_size: int = 128,
    ):
        """
        Initialize the DQN agent.

        Args:
            env: The shipping environment
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            memory_size: Size of the replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency (episodes) to update target network
            hidden_size: Size of hidden layers in the network
        """
        super().__init__(env)

        # Calculate state and action sizes
        self.state_size = self._get_state_size()
        self.action_size = get_action_space_size(env)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.model = DQNNetwork(self.state_size, self.action_size, hidden_size).to(
            self.device
        )
        self.target_model = DQNNetwork(
            self.state_size, self.action_size, hidden_size
        ).to(self.device)
        self.update_target_model()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _get_state_size(self) -> int:
        """Get the state vector size from the environment."""
        state = self.env.reset()
        processed = preprocess_state(state)
        return processed.shape[1]

    def update_target_model(self) -> None:
        """Copy weights from main model to target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def is_valid_action(self, action: int) -> bool:
        """
        Check if an action is valid in the current environment state.

        Args:
            action: The action index to validate

        Returns:
            True if the action is valid, False otherwise
        """
        state = self.env._build_state()
        ship_state = state["ship"]

        # Movement actions are always valid
        if action < 4:
            return True

        num_ports = len(self.env.port_positions)

        # Port selection actions
        if action < 4 + num_ports:
            port_idx = action - 4
            origin_port_idx = ship_state["origin_port_index"]

            # Cannot select same port as origin
            if origin_port_idx is not None and port_idx == origin_port_idx:
                return False

            # Must be at the port to select it
            ship_pos = ship_state["position"]
            port_pos = self.env.port_positions[port_idx]
            if not (ship_pos[0] == port_pos[0] and ship_pos[1] == port_pos[1]):
                return False

            return True

        # Cargo actions
        if action < 4 + num_ports + Initial.MAX_CARGO_CAPACITY:
            current_port_idx = self.env._get_current_port_idx()
            if current_port_idx is not None:
                cargo_amount = action - (4 + num_ports)
                return 0 < cargo_amount <= self.env.port_cargo[current_port_idx]

        # Fuel actions
        else:
            current_port_idx = self.env._get_current_port_idx()
            if current_port_idx is not None:
                fuel_amount = action - (4 + num_ports + Initial.MAX_CARGO_CAPACITY)
                return 0 < fuel_amount <= self.env.port_fuel[current_port_idx]

        return False

    def choose_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Preprocessed state array
            deterministic: If True, always choose the best action

        Returns:
            Selected action index
        """
        # Get valid actions
        valid_actions = [a for a in range(self.action_size) if self.is_valid_action(a)]
        if not valid_actions:
            return 0  # Default to MOVE_NORTH

        # Exploration
        if not deterministic and np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)

        # Exploitation
        state_tensor = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]

        # Select best valid action
        valid_q_values = [(a, q_values[a]) for a in valid_actions]
        return max(valid_q_values, key=lambda x: x[1])[0]

    def update(self) -> Optional[float]:
        """
        Perform a training step using experience replay.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample minibatch
        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.vstack([x[0] for x in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([x[1] for x in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([x[2] for x in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.vstack([x[3] for x in minibatch])).to(
            self.device
        )
        dones = torch.FloatTensor(np.array([x[4] for x in minibatch])).to(self.device)

        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Compute loss and backpropagate
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def train(
        self,
        episodes: int = TrainingDefaults.DQN_EPISODES,
        render: bool = False,
        verbose: bool = True,
        max_steps: int = TrainingDefaults.MAX_STEPS_PER_EPISODE,
        save_freq: int = TrainingDefaults.DQN_SAVE_FREQ,
        save_path: Optional[str] = None,
        max_total_steps: Optional[int] = None,
    ) -> TrainingResult:
        """
        Train the DQN agent.

        Args:
            episodes: Number of training episodes
            render: Whether to render the environment
            verbose: Whether to print progress
            max_steps: Maximum steps per episode
            save_freq: How often to save the model (episodes)
            save_path: Base path for saving models
            max_total_steps: Maximum total steps across all episodes (stops early if reached)

        Returns:
            TrainingResult containing metrics
        """
        result = TrainingResult()
        total_steps = 0

        for episode in range(episodes):
            state = self.env.reset()
            state = preprocess_state(state)
            total_reward = 0
            step_count = 0
            done = False
            episode_loss = []

            while not done and step_count < max_steps:
                try:
                    action = self.choose_action(state)
                    env_action = map_action_to_env_action(action, self.env)
                    next_state, reward, done, _ = self.env.step(env_action)
                    next_state = preprocess_state(next_state)

                    # Store and learn
                    self.remember(state, action, reward, next_state, done)
                    loss = self.update()
                    if loss is not None:
                        episode_loss.append(loss)

                    state = next_state
                    total_reward += reward
                    step_count += 1
                    total_steps += 1

                    if render:
                        self.env.render_real_time()

                    # Stop if max_total_steps reached
                    if max_total_steps is not None and total_steps >= max_total_steps:
                        done = True
                        break

                except Exception as exc:
                    if "Destination port must be different" in str(exc):
                        continue
                    else:
                        break

            # Stop training if max_total_steps reached
            if max_total_steps is not None and total_steps >= max_total_steps:
                if verbose:
                    print(f"Reached max_total_steps ({max_total_steps}) at episode {episode + 1}")
                break

            # Update target network periodically
            if episode % self.target_update_freq == 0:
                self.update_target_model()

            # Record metrics
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            result.add_episode(
                reward=total_reward,
                length=step_count,
                loss=avg_loss,
                epsilon=self.epsilon,
            )

            # Logging
            if verbose and episode % 10 == 0:
                print(
                    f"Episode: {episode + 1}/{episodes}, "
                    f"Reward: {total_reward:.2f}, "
                    f"Epsilon: {self.epsilon:.4f}, "
                    f"Steps: {step_count}"
                )

            # Save model periodically
            if save_path and episode % save_freq == 0:
                self.save(f"{save_path}_episode_{episode}.pth")

        return result

    def save(self, path: str) -> None:
        """Save model weights to a file."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)

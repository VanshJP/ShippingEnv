"""SARSA (State-Action-Reward-State-Action) agent for the shipping environment."""

import random
from collections import defaultdict

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from agents.base import BaseAgent, TrainingResult
from shipping import environment
from utils.actions import find_current_port_index, get_possible_actions_for_state
from utils.constants import TrainingDefaults
from utils.preprocessing import discretize_state


class SARSAAgent(BaseAgent):
    """
    SARSA agent implementation using tabular Q-learning.

    SARSA is an on-policy TD control algorithm that updates Q-values
    based on the action actually taken in the next state.
    """

    def __init__(
        self,
        env,
        learning_rate: float = TrainingDefaults.SARSA_LEARNING_RATE,
        discount_factor: float = TrainingDefaults.SARSA_DISCOUNT_FACTOR,
        epsilon: float = TrainingDefaults.SARSA_EXPLORATION_RATE,
        epsilon_min: float = TrainingDefaults.SARSA_MIN_EXPLORATION_RATE,
        epsilon_decay: float = TrainingDefaults.SARSA_EXPLORATION_DECAY,
    ):
        """
        Initialize the SARSA agent.

        Args:
            env: The shipping environment
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon after each episode
        """
        super().__init__(env)

        # Q-table: maps (state, action) pairs to Q-values
        self.q_table: Dict[Tuple, float] = {}

        # Hyperparameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_q_value(self, state: Tuple, action: Tuple) -> float:
        """
        Get the Q-value for a state-action pair.

        Args:
            state: Discretized state tuple
            action: Action tuple

        Returns:
            Q-value (0.0 if not seen before)
        """
        return self.q_table.get((state, action), 0.0)

    def set_q_value(self, state: Tuple, action: Tuple, value: float) -> None:
        """
        Set the Q-value for a state-action pair.

        Args:
            state: Discretized state tuple
            action: Action tuple
            value: New Q-value
        """
        self.q_table[(state, action)] = value

    def _get_possible_actions(self, state: Dict[str, Any]) -> List[List]:
        """
        Generate all possible actions for the current state.

        Args:
            state: Environment state dictionary

        Returns:
            List of possible actions
        """
        actions = []

        # Port selection actions
        actions.extend(
            [
                [environment.ActionType.SELECT_PORT, port_idx]
                for port_idx in range(len(state["ports"]))
            ]
        )

        # Movement actions
        actions.extend(
            [
                [environment.ActionType.MOVE_SHIP, move]
                for move in [
                    (0, -1),  # NORTH
                    (0, 1),  # SOUTH
                    (-1, 0),  # EAST
                    (1, 0),  # WEST
                ]
            ]
        )

        # Cargo and fuel actions if at a port
        current_port_idx = find_current_port_index(state)
        if current_port_idx is not None:
            port = state["ports"][current_port_idx]
            # Cargo actions (up to 20 at a time)
            actions.extend(
                [
                    [environment.ActionType.TAKE_CARGO, amount]
                    for amount in range(1, min(21, port["cargo"] + 1))
                ]
            )
            # Fuel actions (up to 20 at a time)
            actions.extend(
                [
                    [environment.ActionType.TAKE_FUEL, amount]
                    for amount in range(1, min(21, port["fuel"] + 1))
                ]
            )

        return actions

    def choose_action(
        self, state: Dict[str, Any], deterministic: bool = False
    ) -> List[Any]:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Environment state dictionary
            deterministic: If True, always choose the best action

        Returns:
            Selected action as [ActionType, value]
        """
        discretized = discretize_state(state)

        # Exploration: random action
        if not deterministic and random.random() < self.epsilon:
            return self.env.sample_action()

        # Exploitation: best known action
        possible_actions = self._get_possible_actions(state)
        best_action = None
        max_q_value = float("-inf")

        for action in possible_actions:
            q_value = self.get_q_value(discretized, tuple(action))
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action

        return best_action if best_action is not None else self.env.sample_action()

    def update(
        self,
        state: Dict[str, Any],
        action: List,
        reward: float,
        next_state: Dict[str, Any],
        next_action: List,
    ) -> None:
        """
        Update Q-value using SARSA update rule.

        Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Action to be taken in next state
        """
        curr_discrete = discretize_state(state)
        next_discrete = discretize_state(next_state)

        curr_q = self.get_q_value(curr_discrete, tuple(action))
        next_q = self.get_q_value(next_discrete, tuple(next_action))

        # SARSA update rule
        new_q = curr_q + self.lr * (reward + self.gamma * next_q - curr_q)
        self.set_q_value(curr_discrete, tuple(action), new_q)

    def decay_epsilon(self) -> None:
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _try_action(self, action: List) -> Tuple[Dict, float, bool, Any]:
        """
        Attempt to execute an action, handling illegal moves.

        Args:
            action: Action to attempt

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        try:
            return self.env.step(action)
        except Exception:
            # If action is illegal, keep trying random actions
            while True:
                action = self.env.sample_action()
                try:
                    return self.env.step(action)
                except Exception:
                    continue

    def train(
        self,
        episodes: int = TrainingDefaults.SARSA_EPISODES,
        render: bool = False,
        verbose: bool = True,
        test_interval: int = 100,
        test_episodes: int = TrainingDefaults.SARSA_TEST_EPISODES,
        show_graphs: bool = False,
        max_total_steps: Optional[int] = None,
    ) -> TrainingResult:
        """
        Train the SARSA agent.

        Args:
            episodes: Number of training episodes
            render: Whether to render during training
            verbose: Whether to print progress
            test_interval: How often to test the policy (episodes)
            test_episodes: Number of episodes for policy testing
            show_graphs: Whether to display reward graphs after training
            max_total_steps: Maximum total steps across all episodes (stops early if reached)

        Returns:
            TrainingResult containing training metrics
        """
        result = TrainingResult()
        detailed_rewards: Dict[int, List[Tuple[int, float]]] = {}
        total_steps = 0

        for episode in range(episodes):
            if verbose:
                print(f"Episode {episode + 1}/{episodes}")

            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            total_reward = 0
            step_count = 0

            while not done:
                next_state, reward, done, _ = self._try_action(action)
                next_action = self.choose_action(next_state)

                # SARSA update
                self.update(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action
                total_reward += reward
                step_count += 1
                total_steps += 1

                if render:
                    self.env.render_real_time()

                # Stop if max_total_steps reached
                if max_total_steps is not None and total_steps >= max_total_steps:
                    done = True
                    break

            # Decay exploration
            self.decay_epsilon()

            # Record metrics
            result.add_episode(
                reward=total_reward, length=step_count, epsilon=self.epsilon
            )

            # Periodic testing
            if episode % test_interval == 0:
                do_render = render and (episode == episodes - test_interval)
                test_results = self.test_policy(
                    test_episodes, render=do_render, verbose=False
                )

                # Store detailed rewards by starting position
                for start_spot, avg_reward in test_results.items():
                    if start_spot not in detailed_rewards:
                        detailed_rewards[start_spot] = []
                    detailed_rewards[start_spot].append((episode, avg_reward))

            # Stop training if max_total_steps reached
            if max_total_steps is not None and total_steps >= max_total_steps:
                if verbose:
                    print(f"Reached max_total_steps ({max_total_steps}) at episode {episode + 1}")
                break

        # Show graphs if requested
        if show_graphs:
            self._plot_rewards(detailed_rewards)

        return result

    def test_policy(
        self, test_episodes: int = 10, render: bool = False, verbose: bool = True
    ) -> Dict[int, float]:
        """
        Evaluate the current policy without exploration.

        Args:
            test_episodes: Number of test episodes
            render: Whether to render during testing
            verbose: Whether to print results

        Returns:
            Dictionary mapping starting port index to average reward
        """
        rewards_by_start: Dict[int, List[float]] = defaultdict(list)

        for i in range(test_episodes):
            state = self.env.reset()
            starting_spot = state["ship"]["origin_port_index"]
            done = False
            episode_reward = 0

            while not done:
                action = self.choose_action(state, deterministic=True)
                state, reward, done, _ = self._try_action(action)
                episode_reward += reward

                if render and i == test_episodes - 1:
                    self.env.render_real_time()

            rewards_by_start[starting_spot].append(episode_reward)

        # Calculate averages
        averages = {
            start: sum(rewards) / len(rewards)
            for start, rewards in rewards_by_start.items()
        }

        if verbose:
            print(f"Test Results (avg rewards by starting port): {averages}")

        return averages

    def _plot_rewards(self, rewards: Dict[int, List[Tuple[int, float]]]) -> None:
        """
        Plot reward curves for each starting position.

        Args:
            rewards: Dictionary mapping start position to (episode, reward) tuples
        """
        for start_spot, points in rewards.items():
            episodes = [p[0] for p in points]
            reward_values = [p[1] for p in points]

            plt.figure(figsize=(10, 6))
            plt.plot(
                episodes, reward_values, marker="o", label=f"Start Port {start_spot}"
            )
            plt.title(f"Rewards Over Episodes for Start Port {start_spot}")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.legend()
            plt.grid(True)
            plt.show()

    def save(self, path: str) -> None:
        """Save the Q-table to a file."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "epsilon": self.epsilon,
                    "lr": self.lr,
                    "gamma": self.gamma,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load the Q-table from a file."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.q_table = data["q_table"]
            self.epsilon = data.get("epsilon", self.epsilon_min)
            self.lr = data.get("lr", self.lr)
            self.gamma = data.get("gamma", self.gamma)

    def reset(self) -> None:
        """Reset epsilon to initial value for retraining."""
        self.epsilon = TrainingDefaults.SARSA_EXPLORATION_RATE

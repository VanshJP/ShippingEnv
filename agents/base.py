"""Base agent interface for RL agents in the shipping environment."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseAgent(ABC):
    """
    Abstract base class for all reinforcement learning agents.

    All agents should inherit from this class and implement the required methods.
    """

    def __init__(self, env, **kwargs):
        """
        Initialize the agent.

        Args:
            env: The shipping environment instance
            **kwargs: Additional agent-specific parameters
        """
        self.env = env

    @abstractmethod
    def choose_action(self, state: Dict[str, Any], deterministic: bool = False) -> Any:
        """
        Select an action given the current state.

        Args:
            state: The current environment state
            deterministic: If True, select the best action without exploration

        Returns:
            The selected action in environment format
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Update the agent based on experience.

        Implementation varies by agent type (e.g., Q-update for SARSA,
        replay for DQN, backpropagation for MCTS).
        """
        pass

    @abstractmethod
    def train(
        self, episodes: int, render: bool = False, verbose: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """
        Train the agent for a specified number of episodes.

        Args:
            episodes: Number of training episodes
            render: Whether to render the environment during training
            verbose: Whether to print progress information
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training metrics (rewards, losses, etc.)
        """
        pass

    def save(self, path: str) -> None:
        """
        Save the agent's learned parameters to a file.

        Args:
            path: File path to save to
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support saving")

    def load(self, path: str) -> None:
        """
        Load the agent's learned parameters from a file.

        Args:
            path: File path to load from
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support loading")

    def reset(self) -> None:
        """Reset the agent's internal state (if any) for a new episode."""
        pass


class TrainingResult:
    """Container for training results and metrics."""

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: List[float] = []
        self.epsilon_values: List[float] = []
        self.additional_metrics: Dict[str, List[Any]] = {}

    def add_episode(
        self,
        reward: float,
        length: int,
        loss: Optional[float] = None,
        epsilon: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Add metrics for a completed episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if loss is not None:
            self.losses.append(loss)
        if epsilon is not None:
            self.epsilon_values.append(epsilon)
        for key, value in kwargs.items():
            if key not in self.additional_metrics:
                self.additional_metrics[key] = []
            self.additional_metrics[key].append(value)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of training results."""
        import numpy as np

        summary = {
            "total_episodes": len(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "std_reward": np.std(self.episode_rewards) if self.episode_rewards else 0,
            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
            "min_reward": min(self.episode_rewards) if self.episode_rewards else 0,
            "mean_episode_length": np.mean(self.episode_lengths)
            if self.episode_lengths
            else 0,
        }
        if self.losses:
            summary["mean_loss"] = np.mean(self.losses)
        if self.epsilon_values:
            summary["final_epsilon"] = self.epsilon_values[-1]
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary."""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "losses": self.losses,
            "epsilon_values": self.epsilon_values,
            "additional_metrics": self.additional_metrics,
            "summary": self.get_summary(),
        }

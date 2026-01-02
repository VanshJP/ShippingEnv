"""Monte Carlo Tree Search (MCTS) agent for the shipping environment."""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.base import BaseAgent, TrainingResult
from shipping import environment
from utils.constants import TrainingDefaults


class MCTSNode:
    """
    Node in the Monte Carlo search tree.

    Each node represents a state and tracks visit counts and accumulated rewards
    for UCB1-based selection.
    """

    def __init__(
        self,
        state: Dict[str, Any],
        action: Optional[List] = None,
        parent: Optional["MCTSNode"] = None,
    ):
        """
        Initialize an MCTS node.

        Args:
            state: Environment state dictionary at this node
            action: Action taken to reach this node from parent
            parent: Parent node in the tree
        """
        self.state = state
        self.action = action
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.total_reward = 0.0

    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state."""
        return self.state.get("done", False)

    def is_fully_expanded(self, env) -> bool:
        """Check if all possible actions have been tried from this node."""
        return len(self.children) == len(self.get_possible_actions(env))

    def get_possible_actions(self, env) -> List[List]:
        """
        Get all possible actions from this state.

        Args:
            env: The environment (used for action validation)

        Returns:
            List of possible actions
        """
        actions = []
        current_port_idx = self.state["ship"]["origin_port_index"]

        # If no destination selected, must select a port first
        if self.state["ship"]["destination_port_index"] is None:
            for idx in range(len(self.state["ports"])):
                if idx != current_port_idx:
                    actions.append([environment.ActionType.SELECT_PORT, idx])
        else:
            # Check if we need cargo or fuel
            if self.state["ship"]["cargo"] == 0 and current_port_idx is not None:
                port = self.state["ports"][current_port_idx]
                for amount in range(1, port["cargo"] + 1):
                    actions.append([environment.ActionType.TAKE_CARGO, amount])
            elif self.state["ship"]["fuel"] == 0 and current_port_idx is not None:
                port = self.state["ports"][current_port_idx]
                for amount in range(1, port["fuel"] + 1):
                    actions.append([environment.ActionType.TAKE_FUEL, amount])
            else:
                # Movement actions
                actions.extend(
                    [
                        [environment.ActionType.MOVE_SHIP, [0, -1]],  # NORTH
                        [environment.ActionType.MOVE_SHIP, [0, 1]],  # SOUTH
                        [environment.ActionType.MOVE_SHIP, [-1, 0]],  # EAST
                        [environment.ActionType.MOVE_SHIP, [1, 0]],  # WEST
                    ]
                )

        return actions

    def _action_to_hashable(self, action: List) -> tuple:
        """Convert an action to a hashable tuple for comparison."""
        if action is None:
            return ()
        action_type = action[0]
        action_value = action[1]
        if isinstance(action_value, list):
            return (action_type, tuple(action_value))
        else:
            return (action_type, action_value)

    def get_untried_actions(self, env) -> List[List]:
        """Get actions that haven't been expanded yet."""
        possible = self.get_possible_actions(env)
        tried = {self._action_to_hashable(c.action) for c in self.children}

        return [a for a in possible if self._action_to_hashable(a) not in tried]

    def best_child(self, c_param: float = 1.4) -> "MCTSNode":
        """
        Select the best child using UCB1 formula.

        Args:
            c_param: Exploration constant (higher = more exploration)

        Returns:
            Child node with highest UCB1 value
        """
        ucb_values = []
        for child in self.children:
            if child.visits == 0:
                ucb_values.append(float("inf"))
            else:
                exploitation = child.total_reward / child.visits
                exploration = c_param * np.sqrt(2 * np.log(self.visits) / child.visits)
                ucb_values.append(exploitation + exploration)

        return self.children[np.argmax(ucb_values)]

    def expand(
        self, action: List, next_state: Dict[str, Any], reward: float
    ) -> "MCTSNode":
        """
        Create a new child node for an action.

        Args:
            action: Action taken
            next_state: Resulting state
            reward: Immediate reward received

        Returns:
            The new child node
        """
        child = MCTSNode(next_state, action=action, parent=self)
        child.total_reward = reward
        self.children.append(child)
        return child

    def backpropagate(self, reward: float) -> None:
        """
        Propagate reward back up the tree.

        Args:
            reward: Total reward from simulation
        """
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)


class MCTSAgent(BaseAgent):
    """
    Monte Carlo Tree Search agent implementation.

    Uses tree search with UCB1 selection to find good actions
    through simulation-based planning.
    """

    def __init__(
        self,
        env,
        num_simulations: int = TrainingDefaults.MCTS_SIMULATIONS,
        exploration_param: float = TrainingDefaults.MCTS_EXPLORATION_PARAM,
        max_rollout_steps: int = 100,
    ):
        """
        Initialize the MCTS agent.

        Args:
            env: The shipping environment
            num_simulations: Number of MCTS simulations per action selection
            exploration_param: UCB1 exploration constant
            max_rollout_steps: Maximum steps in rollout simulation
        """
        super().__init__(env)
        self.num_simulations = num_simulations
        self.exploration_param = exploration_param
        self.max_rollout_steps = max_rollout_steps
        self.map_file = "mapa_mundi_binario.jpg"

    def _create_env_copy(self) -> environment.Environment:
        """
        Create a copy of the environment for simulation.

        Returns:
            A new environment instance with copied state
        """
        env_copy = environment.Environment(self.map_file)
        env_copy.port_positions = self.env.port_positions.copy()
        env_copy.port_fuel = self.env.port_fuel.copy()
        env_copy.port_cargo = self.env.port_cargo.copy()
        env_copy.ship_position = self.env.ship_position.copy()
        env_copy.cargo = self.env.cargo
        env_copy.fuel = self.env.fuel
        env_copy.origin_port_index = self.env.origin_port_index
        env_copy.destination_port_index = self.env.destination_port_index
        env_copy.np_game = self.env.np_game.copy()
        return env_copy

    def _rollout(
        self, env_copy: environment.Environment, state: Dict[str, Any]
    ) -> float:
        """
        Perform a random rollout from a state to estimate its value.

        Args:
            env_copy: Environment copy for simulation
            state: Starting state

        Returns:
            Total reward accumulated during rollout
        """
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < self.max_rollout_steps:
            action = env_copy.sample_action()
            try:
                state, reward, done, _ = env_copy.step(action)
                total_reward += reward
                steps += 1
            except Exception:
                # If action fails, try another
                continue

        return total_reward

    def choose_action(
        self, state: Dict[str, Any], deterministic: bool = False
    ) -> List[Any]:
        """
        Select an action using MCTS.

        Args:
            state: Current environment state
            deterministic: If True, use c_param=0 for final selection

        Returns:
            Best action found by MCTS
        """
        root = MCTSNode(state)

        for _ in range(self.num_simulations):
            node = root
            env_copy = self._create_env_copy()

            # Selection: traverse tree using UCB1
            while not node.is_terminal() and node.is_fully_expanded(env_copy):
                node = node.best_child(self.exploration_param)
                if node.action:
                    try:
                        _, _, done, _ = env_copy.step(node.action)
                        if done:
                            break
                    except Exception:
                        break

            # Expansion: add a new child if not terminal
            if not node.is_terminal():
                untried = node.get_untried_actions(env_copy)
                if untried:
                    action = random.choice(untried)
                    try:
                        next_state, reward, done, _ = env_copy.step(action)
                        node = node.expand(action, next_state, reward)
                    except Exception:
                        continue

            # Simulation: random rollout
            rollout_reward = self._rollout(env_copy, node.state)

            # Backpropagation: update statistics
            node.backpropagate(rollout_reward)

        # Select best action (no exploration in final choice)
        if root.children:
            c_param = 0 if deterministic else 0
            return root.best_child(c_param=c_param).action
        else:
            return self.env.sample_action()

    def update(self, *args, **kwargs) -> None:
        """
        MCTS doesn't maintain persistent learned values between episodes.
        This method is a no-op for interface compatibility.
        """
        pass

    def train(
        self,
        episodes: int = TrainingDefaults.MCTS_EPISODES,
        render: bool = False,
        verbose: bool = True,
        max_steps: int = TrainingDefaults.MCTS_MAX_STEPS,
        max_total_steps: Optional[int] = None,
    ) -> TrainingResult:
        """
        Train/evaluate the MCTS agent over multiple episodes.

        Note: MCTS doesn't learn persistent values between episodes.
        This method runs episodes to evaluate performance.

        Args:
            episodes: Number of episodes to run
            render: Whether to render the environment
            verbose: Whether to print progress
            max_steps: Maximum steps per episode
            max_total_steps: Maximum total steps across all episodes (stops early if reached)

        Returns:
            TrainingResult containing episode metrics
        """
        result = TrainingResult()
        episode_logs: List[List[Dict]] = []
        total_steps = 0

        for episode in range(episodes):
            if verbose:
                print(f"Episode {episode + 1}/{episodes}")

            state = self.env.reset()
            total_reward = 0.0
            step_count = 0
            logs = []

            while step_count < max_steps:
                current_port = state["ship"]["origin_port_index"]
                current_location = state["ship"]["position"]

                action = self.choose_action(state)

                try:
                    next_state, reward, done, _ = self.env.step(action)
                except Exception as e:
                    if verbose:
                        print(f"  Step {step_count}: Action failed - {e}")
                    # Try a random action
                    action = self.env.sample_action()
                    try:
                        next_state, reward, done, _ = self.env.step(action)
                    except Exception:
                        continue

                # Log step details
                logs.append(
                    {
                        "step": step_count + 1,
                        "current_port": current_port,
                        "current_location": current_location,
                        "action": action,
                        "next_location": next_state["ship"]["position"],
                        "reward": reward,
                    }
                )

                total_reward += reward
                state = next_state
                step_count += 1
                total_steps += 1

                if render:
                    self.env.render_real_time()

                # Stop if max_total_steps reached
                if max_total_steps is not None and total_steps >= max_total_steps:
                    done = True
                    break

                if done:
                    break

            episode_logs.append(logs)
            result.add_episode(reward=total_reward, length=step_count)

            if verbose:
                print(
                    f"  Finished: Total Reward = {total_reward:.2f}, Steps = {step_count}"
                )

            # Stop training if max_total_steps reached
            if max_total_steps is not None and total_steps >= max_total_steps:
                if verbose:
                    print(f"Reached max_total_steps ({max_total_steps}) at episode {episode + 1}")
                break

        # Store logs in additional metrics
        result.additional_metrics["episode_logs"] = episode_logs

        if verbose:
            summary = result.get_summary()
            print(f"\nTraining Summary:")
            print(f"  Average Reward: {summary['mean_reward']:.2f}")
            print(f"  Std Reward: {summary['std_reward']:.2f}")
            print(f"  Best Reward: {summary['max_reward']:.2f}")

        return result

    def print_episode_logs(self, result: TrainingResult) -> None:
        """
        Print detailed logs for each episode.

        Args:
            result: TrainingResult containing episode logs
        """
        logs = result.additional_metrics.get("episode_logs", [])
        for episode_idx, episode_log in enumerate(logs):
            print(f"\nEpisode {episode_idx + 1} Logs:")
            for step_log in episode_log:
                print(f"  Step {step_log['step']}:")
                print(f"    Current Port: {step_log['current_port']}")
                print(f"    Current Location: {step_log['current_location']}")
                print(f"    Action Taken: {step_log['action']}")
                print(f"    Next Location: {step_log['next_location']}")
                print(f"    Reward: {step_log['reward']}")

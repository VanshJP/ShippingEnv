#!/usr/bin/env python3
"""
Unified entry point for training reinforcement learning agents
on the shipping environment.

Usage:
    python main.py --agent dqn --episodes 1000
    python main.py --agent sarsa --episodes 10000 --render
    python main.py --agent mcts --episodes 10 --simulations 100
"""

import argparse
import sys
from typing import Optional

import cv2

from agents import DQNAgent, MCTSAgent, SARSAAgent
from shipping import Environment
from utils.constants import DEFAULT_MAP_FILE, DEFAULT_PORTS, TrainingDefaults


def create_environment(map_file: str = DEFAULT_MAP_FILE) -> Environment:
    """
    Create and configure the shipping environment with default ports.

    Args:
        map_file: Path to the map image file

    Returns:
        Configured Environment instance
    """
    env = Environment(map_file)
    for port_position in DEFAULT_PORTS:
        env.add_port(port_position)
    return env


def train_dqn(
    env: Environment,
    episodes: int,
    batch_size: int,
    render: bool,
    save_path: Optional[str],
    verbose: bool,
    max_total_steps: int,
) -> None:
    """Train a DQN agent."""
    print("=" * 60)
    print("Training DQN Agent")
    print("=" * 60)
    print(f"  Max Total Steps: {max_total_steps}")
    print(f"  Episodes: {episodes}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Render: {render}")
    print("=" * 60)

    agent = DQNAgent(env, batch_size=batch_size)
    result = agent.train(
        episodes=episodes,
        render=render,
        verbose=verbose,
        save_path=save_path,
        max_total_steps=max_total_steps,
    )

    summary = result.get_summary()
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Mean Reward: {summary['mean_reward']:.2f}")
    print(f"  Max Reward: {summary['max_reward']:.2f}")
    print(f"  Final Epsilon: {summary.get('final_epsilon', 'N/A')}")
    print("=" * 60)


def train_sarsa(
    env: Environment,
    episodes: int,
    render: bool,
    show_graphs: bool,
    verbose: bool,
    max_total_steps: int,
) -> None:
    """Train a SARSA agent."""
    print("=" * 60)
    print("Training SARSA Agent")
    print("=" * 60)
    print(f"  Max Total Steps: {max_total_steps}")
    print(f"  Episodes: {episodes}")
    print(f"  Render: {render}")
    print(f"  Show Graphs: {show_graphs}")
    print("=" * 60)

    agent = SARSAAgent(env)
    result = agent.train(
        episodes=episodes,
        render=render,
        verbose=verbose,
        show_graphs=show_graphs,
        max_total_steps=max_total_steps,
    )

    summary = result.get_summary()
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Mean Reward: {summary['mean_reward']:.2f}")
    print(f"  Max Reward: {summary['max_reward']:.2f}")
    print(f"  Total Episodes: {summary['total_episodes']}")
    print("=" * 60)


def train_mcts(
    env: Environment,
    episodes: int,
    simulations: int,
    render: bool,
    verbose: bool,
    show_logs: bool,
    max_total_steps: int,
) -> None:
    """Train/evaluate an MCTS agent."""
    print("=" * 60)
    print("Running MCTS Agent")
    print("=" * 60)
    print(f"  Max Total Steps: {max_total_steps}")
    print(f"  Episodes: {episodes}")
    print(f"  Simulations per action: {simulations}")
    print(f"  Render: {render}")
    print("=" * 60)

    agent = MCTSAgent(env, num_simulations=simulations)
    result = agent.train(
        episodes=episodes,
        render=render,
        verbose=verbose,
        max_total_steps=max_total_steps,
    )

    if show_logs:
        agent.print_episode_logs(result)

    summary = result.get_summary()
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"  Mean Reward: {summary['mean_reward']:.2f}")
    print(f"  Std Reward: {summary['std_reward']:.2f}")
    print(f"  Best Reward: {summary['max_reward']:.2f}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train reinforcement learning agents on the shipping environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --agent dqn --episodes 1000
  python main.py --agent sarsa --episodes 5000 --render
  python main.py --agent mcts --episodes 10 --simulations 100

Available agents:
  dqn    - Deep Q-Network (neural network-based)
  sarsa  - SARSA (tabular Q-learning)
  mcts   - Monte Carlo Tree Search (planning-based)
        """,
    )

    # Required arguments
    parser.add_argument(
        "--agent",
        "-a",
        type=str,
        required=True,
        choices=["dqn", "sarsa", "mcts"],
        help="RL agent to use (dqn, sarsa, or mcts)",
    )

    # Common arguments
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=None,
        help="Number of training episodes (default varies by agent)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=TrainingDefaults.DEFAULT_MAX_TOTAL_STEPS,
        help=f"Maximum total steps across all episodes (default: {TrainingDefaults.DEFAULT_MAX_TOTAL_STEPS})",
    )
    parser.add_argument(
        "--no-render",
        action="store_false",
        dest="render",
        help="Disable rendering during training",
    )
    parser.set_defaults(render=True)
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--map",
        "-m",
        type=str,
        default=DEFAULT_MAP_FILE,
        help=f"Path to map image file (default: {DEFAULT_MAP_FILE})",
    )

    # DQN-specific arguments
    dqn_group = parser.add_argument_group("DQN options")
    dqn_group.add_argument(
        "--batch-size",
        type=int,
        default=TrainingDefaults.DQN_BATCH_SIZE,
        help=f"Batch size for DQN training (default: {TrainingDefaults.DQN_BATCH_SIZE})",
    )
    dqn_group.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Base path for saving DQN model checkpoints",
    )

    # SARSA-specific arguments
    sarsa_group = parser.add_argument_group("SARSA options")
    sarsa_group.add_argument(
        "--no-graphs",
        action="store_false",
        dest="show_graphs",
        help="Disable showing graphs after SARSA training",
    )
    parser.set_defaults(show_graphs=True)

    # MCTS-specific arguments
    mcts_group = parser.add_argument_group("MCTS options")
    mcts_group.add_argument(
        "--simulations",
        "-s",
        type=int,
        default=TrainingDefaults.MCTS_SIMULATIONS,
        help=f"Number of MCTS simulations per action (default: {TrainingDefaults.MCTS_SIMULATIONS})",
    )
    mcts_group.add_argument(
        "--show-logs",
        action="store_true",
        help="Show detailed step logs for MCTS episodes",
    )

    return parser.parse_args()


def get_default_episodes(agent: str) -> int:
    """Get the default number of episodes for an agent type."""
    defaults = {
        "dqn": TrainingDefaults.DQN_EPISODES,
        "sarsa": TrainingDefaults.SARSA_EPISODES,
        "mcts": TrainingDefaults.MCTS_EPISODES,
    }
    return defaults.get(agent, 1000)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set default episodes if not specified
    episodes = args.episodes if args.episodes else get_default_episodes(args.agent)
    verbose = not args.quiet

    try:
        # Create environment
        env = create_environment(args.map)
        print(f"Environment created with {len(DEFAULT_PORTS)} ports")

        # Run selected agent
        if args.agent == "dqn":
            train_dqn(
                env=env,
                episodes=episodes,
                batch_size=args.batch_size,
                render=args.render,
                save_path=args.save_path,
                verbose=verbose,
                max_total_steps=args.max_steps,
            )

        elif args.agent == "sarsa":
            train_sarsa(
                env=env,
                episodes=episodes,
                render=args.render,
                show_graphs=args.show_graphs,
                verbose=verbose,
                max_total_steps=args.max_steps,
            )

        elif args.agent == "mcts":
            train_mcts(
                env=env,
                episodes=episodes,
                simulations=args.simulations,
                render=args.render,
                verbose=verbose,
                show_logs=args.show_logs,
                max_total_steps=args.max_steps,
            )

        return 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return 1

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        raise

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())

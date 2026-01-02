# Shipping Environment

A reinforcement learning environment for training agents to navigate ships between ports, managing cargo and fuel resources.

## Project Structure

```
ShippingEnv/
├── main.py              # Unified CLI entry point
├── agents/              # Reinforcement learning agents
│   ├── __init__.py
│   ├── base.py          # BaseAgent interface & TrainingResult
│   ├── dqn.py           # Deep Q-Network agent
│   ├── sarsa.py         # SARSA agent
│   └── mcts.py          # Monte Carlo Tree Search agent
├── utils/               # Shared utilities
│   ├── __init__.py
│   ├── constants.py     # Training defaults & constants
│   ├── preprocessing.py # State preprocessing functions
│   └── actions.py       # Action mapping utilities
├── shipping/            # Environment implementation
│   ├── __init__.py
│   ├── environment.py   # Main Environment class
│   ├── type.py          # Type definitions
│   └── util.py          # Utility functions
└── mapa_mundi_binario.jpg  # Default map file
```

## Setup

### Using uv (Recommended)

```bash
# Create virtual environment and install dependencies
uv venv --python 3.12
uv sync
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the unified entry point with your choice of RL agent:

```bash
uv run python main.py --agent <agent_name> [options]
```

Or if using pip:

```bash
python main.py --agent <agent_name> [options]
```

### Available Agents

| Agent | Description |
|-------|-------------|
| `dqn` | Deep Q-Network - Neural network-based Q-learning with experience replay |
| `sarsa` | SARSA - Tabular on-policy TD control algorithm |
| `mcts` | Monte Carlo Tree Search - Planning-based approach using simulation |

### Command Line Options

#### Common Options

| Option | Short | Description |
|--------|-------|-------------|
| `--agent` | `-a` | **Required.** Agent to use: `dqn`, `sarsa`, or `mcts` |
| `--episodes` | `-e` | Number of training episodes (default varies by agent) |
| `--render` | `-r` | Render the environment during training |
| `--quiet` | `-q` | Suppress verbose output |
| `--map` | `-m` | Path to map image file (default: `mapa_mundi_binario.jpg`) |

#### DQN Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size` | Batch size for training | 32 |
| `--save-path` | Base path for saving model checkpoints | None |

#### SARSA Options

| Option | Description | Default |
|--------|-------------|---------|
| `--show-graphs` | Display reward graphs after training | False |

#### MCTS Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--simulations` | `-s` | Number of MCTS simulations per action | 50 |
| `--show-logs` | | Show detailed step logs for episodes | False |

## Examples

### Train a DQN agent for 1000 episodes

```bash
uv run python main.py --agent dqn --episodes 1000
```

### Train a DQN agent with model saving

```bash
uv run python main.py --agent dqn --episodes 5000 --save-path ./models/dqn
```

### Train a SARSA agent with visualization

```bash
uv run python main.py --agent sarsa --episodes 10000 --render --show-graphs
```

### Run MCTS with custom simulation count

```bash
uv run python main.py --agent mcts --episodes 20 --simulations 100 --show-logs
```

### Quick test run (quiet mode)

```bash
uv run python main.py --agent dqn --episodes 10 --quiet
```

## Default Training Parameters

### DQN
- Episodes: 1000
- Batch Size: 32
- Learning Rate: 0.001
- Discount Factor (γ): 0.95
- Epsilon: 1.0 → 0.01 (decay: 0.995)
- Memory Size: 2000
- Target Network Update: Every 10 episodes

### SARSA
- Episodes: 10000
- Learning Rate: 0.1
- Discount Factor (γ): 0.9
- Epsilon: 1.0 → 0.0001 (decay: 0.999)

### MCTS
- Episodes: 10
- Simulations per Action: 50
- Exploration Parameter: 1.4
- Max Rollout Steps: 100

## Environment Details

The shipping environment simulates a ship navigating between ports on a map. The agent must:

- **Navigate**: Move the ship in four directions (North, South, East, West)
- **Manage Fuel**: Collect fuel at ports to continue moving
- **Transport Cargo**: Pick up cargo at origin ports and deliver to destination ports
- **Select Routes**: Choose destination ports strategically

### Default Ports

The environment is initialized with 5 ports at the following positions:
- Port 0: [41, 40]
- Port 1: [60, 22]
- Port 2: [78, 29]
- Port 3: [49, 72]
- Port 4: [62, 72]

## Programmatic Usage

You can also use the agents directly in your own scripts:

```python
from shipping import Environment
from agents import DQNAgent, SARSAAgent, MCTSAgent

# Create environment
env = Environment("mapa_mundi_binario.jpg")
env.add_port([41, 40])
env.add_port([60, 22])
# ... add more ports

# Create and train an agent
agent = DQNAgent(env, learning_rate=0.001, gamma=0.95)
result = agent.train(episodes=1000, render=False, verbose=True)

# Get training summary
print(result.get_summary())

# Save the trained model
agent.save("my_model.pth")
```

## License

MIT License

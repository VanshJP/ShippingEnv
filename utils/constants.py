"""Shared constants for the shipping environment RL agents."""


class Initial:
    """Initial values for ship state."""

    CARGO = 0
    FUEL = 200
    MAX_CARGO_CAPACITY = 50


class ActionType:
    """Action type identifiers for the environment."""

    MOVE_SHIP = 1
    SELECT_PORT = 2
    TAKE_FUEL = 3
    TAKE_CARGO = 4


class TrainingDefaults:
    """Default training hyperparameters."""

    # Common
    MAX_STEPS_PER_EPISODE = 1000
    DEFAULT_MAX_TOTAL_STEPS = 1000

    # DQN defaults
    DQN_EPISODES = 1000
    DQN_BATCH_SIZE = 32
    DQN_MEMORY_SIZE = 2000
    DQN_GAMMA = 0.95
    DQN_EPSILON = 1.0
    DQN_EPSILON_MIN = 0.01
    DQN_EPSILON_DECAY = 0.995
    DQN_LEARNING_RATE = 0.001
    DQN_TARGET_UPDATE_FREQ = 10
    DQN_SAVE_FREQ = 100

    # SARSA defaults
    SARSA_EPISODES = 10000
    SARSA_LEARNING_RATE = 0.1
    SARSA_DISCOUNT_FACTOR = 0.9
    SARSA_EXPLORATION_RATE = 1.0
    SARSA_MIN_EXPLORATION_RATE = 0.0001
    SARSA_EXPLORATION_DECAY = 0.999
    SARSA_TEST_EPISODES = 10

    # MCTS defaults
    MCTS_EPISODES = 1000
    MCTS_SIMULATIONS = 50
    MCTS_MAX_STEPS = 100
    MCTS_EXPLORATION_PARAM = 1.4


# Default port positions for the environment
DEFAULT_PORTS = [
    [41, 40],
    [60, 22],
    [78, 29],
    [49, 72],
    [62, 72],
]

# Default map file
DEFAULT_MAP_FILE = "mapa_mundi_binario.jpg"

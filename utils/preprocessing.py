"""Shared preprocessing utilities for RL agents."""

import numpy as np

from shipping import ShipMove


class Constants:
    """Shared constants for the shipping environment."""

    CARGO = 0
    FUEL = 200
    MAX_CARGO_CAPACITY = 50


class ActionType:
    """Action type identifiers for the environment."""

    MOVE_SHIP = 1
    SELECT_PORT = 2
    TAKE_FUEL = 3
    TAKE_CARGO = 4


def preprocess_state(env_state):
    """
    Convert environment state dict to a numpy array for neural network input.

    Args:
        env_state: Dictionary containing ship and port information

    Returns:
        numpy array of shape (1, state_size)
    """
    ship_pos = env_state["ship"]["position"]
    ship_fuel = env_state["ship"]["fuel"]
    ship_cargo = env_state["ship"]["cargo"]
    origin_port_index = env_state["ship"]["origin_port_index"]
    destination_port_index = env_state["ship"]["destination_port_index"]

    # Handle None values
    origin_port_index = origin_port_index if origin_port_index is not None else -1
    destination_port_index = (
        destination_port_index if destination_port_index is not None else -1
    )

    port_data = []
    for port in env_state["ports"]:
        port_data.extend(
            [port["position"][0], port["position"][1], port["fuel"], port["cargo"]]
        )

    state = [
        ship_pos[0],
        ship_pos[1],
        ship_fuel,
        ship_cargo,
        origin_port_index,
        destination_port_index,
    ] + port_data

    return np.reshape(state, [1, len(state)])


def discretize_state(state):
    """
    Discretize state for tabular methods (SARSA, Q-learning).

    Args:
        state: Environment state dictionary

    Returns:
        Tuple that can be used as a dictionary key
    """
    ship_pos = tuple(state["ship"]["position"])
    fuel_bucket = min(int(state["ship"]["fuel"] / 20), 4)
    cargo_bucket = min(int(state["ship"]["cargo"] / 10), 4)

    dest_port = (
        state["ship"]["destination_port_index"]
        if state["ship"]["destination_port_index"] is not None
        else -1
    )
    prev_port = (
        state["ship"]["origin_port_index"]
        if state["ship"]["origin_port_index"] is not None
        else -1
    )

    return (ship_pos, cargo_bucket, fuel_bucket, prev_port, dest_port)


def get_action_space_size(env):
    """
    Calculate the total action space size for the environment.

    Args:
        env: The shipping environment

    Returns:
        Total number of possible actions
    """
    num_ports = len(env.port_positions)
    max_cargo = Constants.MAX_CARGO_CAPACITY
    max_fuel = Constants.FUEL
    move_actions = 4  # NORTH, EAST, SOUTH, WEST

    return move_actions + num_ports + max_cargo + max_fuel


def map_action_to_env_action(agent_action, env):
    """
    Convert a discrete action index to environment action format.

    Args:
        agent_action: Integer action index
        env: The shipping environment

    Returns:
        List [ActionType, action_parameter]
    """
    num_ports = len(env.port_positions)
    move_actions = 4

    if agent_action < move_actions:
        moves = [ShipMove.NORTH, ShipMove.EAST, ShipMove.SOUTH, ShipMove.WEST]
        return [ActionType.MOVE_SHIP, moves[agent_action]]
    elif agent_action < move_actions + num_ports:
        return [ActionType.SELECT_PORT, agent_action - move_actions]
    elif agent_action < move_actions + num_ports + Constants.MAX_CARGO_CAPACITY:
        cargo_amount = agent_action - (move_actions + num_ports)
        return [ActionType.TAKE_CARGO, cargo_amount]
    else:
        fuel_amount = agent_action - (
            move_actions + num_ports + Constants.MAX_CARGO_CAPACITY
        )
        return [ActionType.TAKE_FUEL, fuel_amount]


def find_current_port_index(state):
    """
    Find the index of the port at the ship's current position.

    Args:
        state: Environment state dictionary

    Returns:
        Port index or None if not at a port
    """
    ship_pos = tuple(state["ship"]["position"])
    for idx, port in enumerate(state["ports"]):
        if tuple(port["position"]) == ship_pos:
            return idx
    return None


def get_state_size(env):
    """
    Get the state size for neural network input.

    Args:
        env: The shipping environment

    Returns:
        Tuple (1, state_dimension)
    """
    state = env.reset()
    processed = preprocess_state(state)
    return (1, processed.shape[1])

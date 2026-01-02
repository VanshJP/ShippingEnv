"""Shared action utilities for RL agents."""

from shipping import ShipMove


class ActionType:
    """Action type constants for the shipping environment."""

    MOVE_SHIP = 1
    SELECT_PORT = 2
    TAKE_FUEL = 3
    TAKE_CARGO = 4


# Movement directions mapping
MOVE_DIRECTIONS = [
    ShipMove.NORTH,
    ShipMove.EAST,
    ShipMove.SOUTH,
    ShipMove.WEST,
]

MOVE_TUPLES = [
    (0, -1),  # NORTH
    (0, 1),  # SOUTH
    (-1, 0),  # EAST
    (1, 0),  # WEST
]

NUM_MOVE_ACTIONS = 4


def get_action_space_size(env, max_cargo_capacity, max_fuel):
    """
    Calculate the total action space size for the environment.

    Args:
        env: The shipping environment
        max_cargo_capacity: Maximum cargo the ship can carry
        max_fuel: Maximum fuel the ship can carry

    Returns:
        Total number of possible actions
    """
    num_ports = len(env.port_positions)
    return NUM_MOVE_ACTIONS + num_ports + max_cargo_capacity + max_fuel


def map_action_to_env_action(agent_action, env, max_cargo_capacity):
    """
    Convert an agent's discrete action index to an environment action.

    Args:
        agent_action: Integer action index from the agent
        env: The shipping environment
        max_cargo_capacity: Maximum cargo capacity

    Returns:
        List of [ActionType, action_value] for the environment
    """
    num_ports = len(env.port_positions)

    if agent_action < NUM_MOVE_ACTIONS:
        return [ActionType.MOVE_SHIP, MOVE_DIRECTIONS[agent_action]]
    elif agent_action < NUM_MOVE_ACTIONS + num_ports:
        return [ActionType.SELECT_PORT, agent_action - NUM_MOVE_ACTIONS]
    elif agent_action < NUM_MOVE_ACTIONS + num_ports + max_cargo_capacity:
        cargo_amount = agent_action - (NUM_MOVE_ACTIONS + num_ports)
        return [ActionType.TAKE_CARGO, cargo_amount]
    else:
        fuel_amount = agent_action - (NUM_MOVE_ACTIONS + num_ports + max_cargo_capacity)
        return [ActionType.TAKE_FUEL, fuel_amount]


def get_possible_actions_for_state(state, env, max_cargo_amount=20, max_fuel_amount=20):
    """
    Get all possible actions for the current state.

    Args:
        state: Current environment state dictionary
        env: The shipping environment
        max_cargo_amount: Maximum cargo to consider taking at once
        max_fuel_amount: Maximum fuel to consider taking at once

    Returns:
        List of possible [ActionType, value] actions
    """
    actions = []

    # Port selection actions
    actions.extend(
        [[ActionType.SELECT_PORT, port_idx] for port_idx in range(len(state["ports"]))]
    )

    # Movement actions
    actions.extend([[ActionType.MOVE_SHIP, move] for move in MOVE_TUPLES])

    # Cargo and fuel actions if at a port
    current_port_idx = find_current_port_index(state)
    if current_port_idx is not None:
        port = state["ports"][current_port_idx]
        actions.extend(
            [
                [ActionType.TAKE_CARGO, amount]
                for amount in range(1, min(max_cargo_amount + 1, port["cargo"] + 1))
            ]
        )
        actions.extend(
            [
                [ActionType.TAKE_FUEL, amount]
                for amount in range(1, min(max_fuel_amount + 1, port["fuel"] + 1))
            ]
        )

    return actions


def find_current_port_index(state):
    """
    Find the index of the port the ship is currently at.

    Args:
        state: Current environment state dictionary

    Returns:
        Port index if ship is at a port, None otherwise
    """
    ship_pos = tuple(state["ship"]["position"])
    for idx, port in enumerate(state["ports"]):
        if tuple(port["position"]) == ship_pos:
            return idx
    return None

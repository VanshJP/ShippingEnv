import numpy as np
import cv2
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Import your environment modules
from shipping import Environment, ShipMove

class Initial:
    CARGO = 0
    FUEL = 200
    MAX_CARGO_CAPACITY = 50

class ActionType:
    MOVE_SHIP = 1
    SELECT_PORT = 2
    TAKE_FUEL = 3
    TAKE_CARGO = 4

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        class DQN(nn.Module):
            def __init__(self, input_size, num_actions):
                super(DQN, self).__init__()
                self.fc1 = nn.Linear(input_size, 128)
                self.fc2 = nn.Linear(128, 128)
                self.fc3 = nn.Linear(128, num_actions)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        return DQN(self.state_size[1], self.action_size)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def is_valid_action(self, action, env):
        """
        Validates the action based on the current environment state.
        """
        state = env._build_state()
        ship_state = state['ship']

        if action < 4:  # Movement actions
            return True

        # Port selection actions
        elif action < 4 + len(env.port_positions):
            port_idx = action - 4
            origin_port_idx = ship_state['origin_port_index']

            # Prevent selecting the same port as origin
            if origin_port_idx is not None and port_idx == origin_port_idx:
                return False

            # Check if ship is at the port position
            ship_pos = ship_state['position']
            port_pos = env.port_positions[port_idx]
            if not (ship_pos[0] == port_pos[0] and ship_pos[1] == port_pos[1]):
                return False

            return True

        # Cargo actions
        elif action < 4 + len(env.port_positions) + Initial.MAX_CARGO_CAPACITY:
            current_port_idx = env._get_current_port_idx()
            if current_port_idx is not None:
                cargo_amount = action - (4 + len(env.port_positions))
                return 0 < cargo_amount <= env.port_cargo[current_port_idx]

        # Fuel actions
        else:
            current_port_idx = env._get_current_port_idx()
            if current_port_idx is not None:
                fuel_amount = action - (4 + len(env.port_positions) + Initial.MAX_CARGO_CAPACITY)
                return 0 < fuel_amount <= env.port_fuel[current_port_idx]

        return False

    def act(self, state, env, deterministic=False):
        """
        Select an action based on the current state.

        Args:
            state: The current state
            env: The environment
            deterministic: If True, always choose best action, if False use epsilon-greedy
        """
        # Get list of valid actions first
        valid_actions = [a for a in range(self.action_size) if self.is_valid_action(a, env)]
        if not valid_actions:
            return 0  # Default to MOVE_NORTH if no valid actions

        if not deterministic and np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)

        # Convert state to tensor for model
        state_tensor = torch.from_numpy(state).float().to(self.device)
        act_values = self.model(state_tensor).cpu().data.numpy()[0]

        # Select highest-valued valid action
        valid_act_values = [(a, act_values[a]) for a in valid_actions]
        return max(valid_act_values, key=lambda x: x[1])[0]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.vstack([x[0] for x in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([x[1] for x in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([x[2] for x in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.vstack([x[3] for x in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([x[4] for x in minibatch])).to(self.device)

        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Next Q values
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def train(self, env, episodes, batch_size):
        try:
            for e in range(episodes):
                state = env.reset()
                state = preprocess_state(state)
                total_reward = 0
                done = False
                step_count = 0
                max_steps = 1000  # Prevent infinite loops

                while not done and step_count < max_steps:
                    try:
                        action = self.act(state, env)
                        env_action = map_action_to_env_action(action, env)
                        next_state, reward, done, _ = env.step(env_action)
                        next_state = preprocess_state(next_state)

                        # Store experience and update agent
                        self.remember(state, action, reward, next_state, done)
                        if len(self.memory) >= batch_size:
                            self.replay(batch_size)

                        # Update for next step
                        state = next_state
                        total_reward += reward
                        step_count += 1

                        # Render environment if available
                        env.render_real_time()

                    except Exception as exc:
                        print(f"Error during step {step_count}: {exc}")
                        if "Destination port must be different" in str(exc):
                            reward = -10
                            continue  # Try another action
                        else:
                            break  # Exit episode for other errors

                # Episode completion logging
                if e % 10 == 0:
                    self.update_target_model()
                    print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}, Steps: {step_count}")

                if e % 100 == 0:
                    self.save(f"./dqn_shipping_agent_{e}.pth")

        except Exception as e:
            print(f"Training error: {e}")
            raise

def preprocess_state(env_state):
    ship_pos = env_state['ship']['position']
    ship_fuel = env_state['ship']['fuel']
    ship_cargo = env_state['ship']['cargo']
    origin_port_index = env_state['ship']['origin_port_index']
    destination_port_index = env_state['ship']['destination_port_index']

    port_data = []
    for port in env_state['ports']:
        port_data.extend([port['position'][0], port['position'][1], port['fuel'], port['cargo']])

    state = [ship_pos[0], ship_pos[1], ship_fuel, ship_cargo, origin_port_index, destination_port_index] + port_data
    return np.reshape(state, [1, len(state)])

def get_action_space_size(env):
    num_ports = len(env.port_positions)
    max_cargo = Initial.MAX_CARGO_CAPACITY
    max_fuel = Initial.FUEL
    move_actions = len([ShipMove.NORTH, ShipMove.EAST, ShipMove.SOUTH, ShipMove.WEST])
    total_actions = move_actions + num_ports + max_cargo + max_fuel
    return total_actions

def map_action_to_env_action(agent_action, env):
    num_ports = len(env.port_positions)
    move_actions = 4

    if agent_action < move_actions:
        moves = [ShipMove.NORTH, ShipMove.EAST, ShipMove.SOUTH, ShipMove.WEST]
        return [ActionType.MOVE_SHIP, moves[agent_action]]
    elif agent_action < move_actions + num_ports:
        return [ActionType.SELECT_PORT, agent_action - move_actions]
    elif agent_action < move_actions + num_ports + Initial.MAX_CARGO_CAPACITY:
        cargo_amount = agent_action - (move_actions + num_ports)
        return [ActionType.TAKE_CARGO, cargo_amount]
    else:
        fuel_amount = agent_action - (move_actions + num_ports + Initial.MAX_CARGO_CAPACITY)
        return [ActionType.TAKE_FUEL, fuel_amount]

if __name__ == "__main__":
    # Create environment and add ports
    env = Environment("mapa_mundi_binario.jpg")
    env.add_port([41, 40])
    env.add_port([60, 22])
    env.add_port([78, 29])
    env.add_port([49, 72])
    env.add_port([62, 72])

    # Initialize agent
    state_size = (1, len(preprocess_state(env.reset())[0]))
    action_size = get_action_space_size(env)
    agent = DQNAgent(state_size, action_size)

    # Training parameters
    batch_size = 32
    episodes = 1000

    # Start training
    agent.train(env, episodes, batch_size)

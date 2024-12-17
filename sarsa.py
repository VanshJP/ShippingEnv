'''
import numpy as np
import random

class SARSAAgent:
    def __init__(self, environment, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.env = environment

        # Initialize Q-table
        self.q_table = {}

    def get_q_value(self, state, action):
        state = self.convert_state_to_array(state)
        if type(action) == list:
            actionKey = action[0]
        else:
            actionKey = action

        if action[0] == 1:
            actionKey = tuple([1, action[1][0], action[1][1]])
        else:
            actionKey = tuple([action[0]])

        return self.q_table.get((state, actionKey), 0.0)

    def update_q_value(self, state, action, value):
        state = self.convert_state_to_array(state)
        if type(action) == list:
            actionKey = action[0]
        else:
            actionKey = action

            if action[0] == 1:
                actionKey = tuple([1, action[1][0], action[1][1]])
            else:
                 actionKey = tuple([action[0]])

        self.q_table[(state, actionKey)] = value

    def choose_action(self, state):
        """
        Choose an action based on the epsilon-greedy policy.
        """
        r = random.random()
        state = (self.convert_state_to_array(state))
        #print(state)
        if r < self.epsilon:
            return self.env.sample_action()  # Explore
        else:
            q_values = [self.get_q_value(state, action) for action in range(self.action_size)]
            return int(np.argmax(q_values))  # Exploit

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Update the Q-value using the SARSA update rule.
        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * Q(s', a') - Q(s, a)]
        """
        state = tuple(self.convert_state_to_array(state))
        next_state = tuple(self.convert_state_to_array(next_state))
        current_q = self.get_q_value(state, action)
        next_q = self.get_q_value(next_state, next_action) if not done else 0.0

        # SARSA update rule
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q

        # Update Q-value
        self.update_q_value(state, action, current_q + self.alpha * td_error)

    def decay_epsilon(self):
        """
        Decay the exploration rate.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def convert_state_to_array(self, state):
        if type(state) == tuple:
            return state
        # Extract ship features
        ship_features = (state['ship']['position'] + [
            state['ship']['fuel'],
            state['ship']['cargo'],
        ])

        if not state['ship']['destination_port_index']:
            ship_features += [-1]
        else:
            ship_features += [state['ship']['destination_port_index']]

        return np.array(ship_features, dtype = 'd')


def train_sarsa_agent(agent, episodes, max_steps_per_episode):
    for episode in range(episodes):
        print(episode)
        state = agent.env.reset()
        action = agent.choose_action(state)

        for step in range(max_steps_per_episode):
            #print(step)
            next_state, reward, done, _ = agent.env.step(action)
            next_action = agent.choose_action(next_state)

            agent.update(state, action, reward, next_state, next_action, done)

            state, action = next_state, next_action
            if done:
                break

        if episode % 100 == 0 :
            testingReward = test_agent(agent, episode, max_steps_per_episode)
            print(episode, testingReward)


def test_agent(agent, episode, max_steps_per_episode):
        episode_rewards = []

        for episode in range(episode):
            state = agent.env.reset()
            total_reward = 0

            for step in range(max_steps_per_episode):
                state = tuple(agent.convert_state_to_array(state))
                q_values = [agent.get_q_value(state, action) for action in range(agent.action_size)]
                action = int(np.argmax(q_values))
                # Disable exploration during testing
                next_state, reward, done, _ = agent.env.step(action)

                state = next_state
                total_reward += reward

                if done:
                    break

            episode_rewards.append(total_reward)
            print(f"Test Episode {episode + 1}/{episode} - Total Reward: {total_reward}")

        return episode_rewards
'''
import numpy as np
import random
from shipping import environment

class SARSAAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.01,
                 exploration_decay_rate=0.995):
        self.env = env

        # Q-table to store state-action values
        self.q_table = {}

        # Learning parameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay_rate

    def _discretize_state(self, state):
        # Discretize ship position
        ship_pos = tuple(state['ship']['position'])

        # Discretize fuel into buckets
        fuel_bucket = min(int(state['ship']['fuel'] / 20), 4)

        # Discretize cargo into buckets
        cargo_bucket = min(int(state['ship']['cargo'] / 10), 4)

        # Convert destination port to an index or -1 if None
        dest_port = state['ship']['destination_port_index'] if state['ship'][
                                                                   'destination_port_index'] is not None else -1

        # Return a tuple that can be used as a dictionary key
        return (ship_pos, fuel_bucket, cargo_bucket, dest_port)

    def get_q_value(self, state, action):
        print(type(state), type(action))
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        discretized_state = self._discretize_state(state)

        # Exploration
        if random.random() < self.epsilon:
            return self.env.sample_action()

        # Exploitation
        else:
            # Find the best action for the current state
            best_action = None
            max_q_value = float('-inf')

            # Generate possible actions similar to environment's sample_action method
            possible_actions = [
                [environment.ActionType.SELECT_PORT, port_idx] for port_idx in range(len(state['ports']))
            ]
            possible_actions.extend([
                [environment.ActionType.MOVE_SHIP, move] for move in [
                    [0, -1],  # NORTH
                    [0, 1],  # SOUTH
                    [-1, 0],  # EAST
                    [1, 0]  # WEST
                ]
            ])

            # Add cargo and fuel actions if at a port
            current_port_idx = self._find_current_port_index(state)
            if current_port_idx is not None:
                possible_actions.extend([
                    [environment.ActionType.TAKE_CARGO, cargo_amount]
                    for cargo_amount in range(1, min(21, state['ports'][current_port_idx]['cargo'] + 1))
                ])
                possible_actions.extend([
                    [environment.ActionType.TAKE_FUEL, fuel_amount]
                    for fuel_amount in range(1, min(21, state['ports'][current_port_idx]['fuel'] + 1))
                ])

            # Find the action with the highest Q-value
            for action in possible_actions:
                q_value = self.get_q_value(discretized_state, tuple(action))
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action

            return best_action if best_action is not None else self.env.sample_action()

    def _find_current_port_index(self, state):
        ship_pos = tuple(state['ship']['position'])
        for idx, port in enumerate(state['ports']):
            if tuple(port['position']) == ship_pos:
                return idx
        return None

    def update(self, state, action, reward, next_state, next_action):
        # Discretize states
        curr_state_discrete = self._discretize_state(state)
        next_state_discrete = self._discretize_state(next_state)

        # Get current and next Q-values
        curr_q_value = self.get_q_value(curr_state_discrete, tuple(action))
        next_q_value = self.get_q_value(next_state_discrete, tuple(next_action))

        # SARSA update rule
        new_q_value = curr_q_value + self.lr * (
                reward + self.gamma * next_q_value - curr_q_value
        )

        # Update Q-table
        self.q_table[(curr_state_discrete, tuple(action))] = new_q_value

    def decay_exploration(self):
        """
        Decay the exploration rate.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def train_agent(env, num_episodes=100000):
    # Initialize the agent with the environment
    agent = SARSAAgent(env)

    total_rewards = []

    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()

        # Choose initial action
        action = agent.choose_action(state)

        episode_reward = 0
        done = False

        while not done:
            # Take the action and observe next state and reward
            next_state, reward, done, _ = env.step(action)

            # Choose next action
            next_action = agent.choose_action(next_state)

            # Update the agent
            agent.update(state, action, reward, next_state, next_action)

            # Move to the next state and action
            state = next_state
            action = next_action

            episode_reward += reward

        # Decay exploration
        agent.decay_exploration()

        total_rewards.append(episode_reward)

        # Optional: Print progress
        if episode % 100 == 0:
            print(
                f"Episode {episode}, Average Reward: {np.mean(total_rewards[-100:])}, Exploration Rate: {agent.epsilon:.4f}")

    return agent





from collections import defaultdict

import random
import matplotlib.pyplot as plt
from shipping import environment

class SARSAAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.0001,
                 exploration_decay_rate=0.999):
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
        dest_port = state['ship']['destination_port_index'] if state['ship']['destination_port_index'] is not None else -1

        prev_port = state['ship']['origin_port_index'] if state['ship']['origin_port_index'] is not None else -1

        # Return a tuple that can be used as a dictionary key
        return (ship_pos, cargo_bucket, fuel_bucket, prev_port, dest_port)

    def get_q_value(self, state, action):
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
                    (0, -1),  # NORTH
                    (0, 1),  # SOUTH
                    (-1, 0),  # EAST
                    (1, 0)  # WEST
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

    def choose_action2(self, state):
        discretized_state = self._discretize_state(state)

        # Should never run
        if random.random() < -1:
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
                    (0, -1),  # NORTH
                    (0, 1),  # SOUTH
                    (-1, 0),  # EAST
                    (1, 0)  # WEST
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


def train_agent(env, num_episodes= 10000):
    # Initialize the agent with the environment
    agent = SARSAAgent(env)
    detailed_rewards = dict()

    for episode in range(num_episodes):
        print(f'Episode {episode}')
        # Reset the environment
        state = env.reset()

        # Choose initial action
        action = agent.choose_action(state)
        done = False


        while not done:
            # Take the action and observe next state and reward
            try:
                # Attempt to take the action
                next_state, reward, done, _ = env.step(action)
            except:
                # Handle illegal moves
                #print(f"Illegal move attempted with action: {action}. Now sampling a random action.")
                illegalMove = True
                while illegalMove:
                    # Choose a new action and attempt it
                    action = env.sample_action()
                    try:
                        next_state, reward, done, _ = env.step(action)
                        illegalMove = False  # Exit loop if the move is successful
                    except:
                        continue


            # Choose next action
            next_action = agent.choose_action(next_state)

            # Update the agent
            agent.update(state, action, reward, next_state, next_action)

            # Move to the next state and action
            state = next_state
            action = next_action

        # Decay exploration
        agent.decay_exploration()

        if episode % 100 == 0:
            if episode == num_episodes - 100:
                avg_rewards = test_policy(env, agent, True)
            else:
                avg_rewards = test_policy(env, agent, False)
            for start_spot, reward in avg_rewards:
                if start_spot not in detailed_rewards:
                    detailed_rewards[start_spot] = []
                detailed_rewards[start_spot].append((episode, reward))

    graphRewards(detailed_rewards, num_episodes)
    return agent

def test_policy(env, agent, doRender = False, test_episodes = 10):
    """Evaluate the current policy by running a few episodes without exploration."""
    totalReward_per_StartSpot = defaultdict(list)
    for i in range(test_episodes):
        state = env.reset()
        startingSpot = state['ship']['origin_port_index']
        done = False
        episode_reward = 0

        while not done:
            # Choose action greedily based on the current policy
            action = agent.choose_action(state)
            try:
                # Attempt to take the action
                next_state, reward, done, _ = env.step(action)
            except:
                # Handle illegal moves
                #print(f"Illegal move attempted with action: {action}. Now sampling a random action.")
                illegalMove = True
                while illegalMove:
                    # Choose a new action and attempt it
                    action = env.sample_action()
                    try:
                        next_state, reward, done, _ = env.step(action)
                        illegalMove = False  # Exit loop if the move is successful
                    except:
                        continue

            if doRender and i == 9:
                env.render_real_time()

            state = next_state
            episode_reward += reward

        totalReward_per_StartSpot[startingSpot].append(episode_reward)

    averages = dict()
    for key, values in totalReward_per_StartSpot.items():
        averages[key] = sum(values) / len(values)
    return averages.items()


def graphRewards(rewards, num_episodes):
    for start_spot, points in rewards.items():
        # Extract x and y coordinates from the tuples
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]

        # Create a new figure for each start spot
        plt.figure(figsize=(10, 6))

        # Plot the points
        plt.plot(x_coords, y_coords, marker='o', label=f"Start Spot {start_spot}", color='b')

        # Add titles and labels
        plt.title(f"Rewards Over Episodes for Start Spot {start_spot}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        # Add a legend
        plt.legend()

        # Display the plot
        plt.grid(True)
        plt.show()





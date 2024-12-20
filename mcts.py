import numpy as np
import random
from shipping import environment  

class Node:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0

    def is_terminal(self):
        return self.state.get("done", False)

    def is_fully_expanded(self, env):
        return len(self.children) == len(self.get_possible_actions(env))

    def get_possible_actions(self, env):
        actions = []
        current_port_idx = self.state["ship"]["origin_port_index"]

        if self.state["ship"]["destination_port_index"] is None:
            for idx in range(len(self.state["ports"])):
                if idx != current_port_idx:
                    actions.append([environment.ActionType.SELECT_PORT, idx])
        else:
            if self.state["ship"]["cargo"] == 0 and current_port_idx is not None:
                for amount in range(1, self.state["ports"][current_port_idx]["cargo"] + 1):
                    actions.append([environment.ActionType.TAKE_CARGO, amount])
            elif self.state["ship"]["fuel"] == 0 and current_port_idx is not None:
                for amount in range(1, self.state["ports"][current_port_idx]["fuel"] + 1):
                    actions.append([environment.ActionType.TAKE_FUEL, amount])
            else:
                actions.extend([
                    [environment.ActionType.MOVE_SHIP, move] for move in [
                        [0, -1],  # NORTH
                        [0, 1],   # SOUTH
                        [-1, 0],  # EAST
                        [1, 0]    # WEST
                    ]
                ])
        return actions

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.total_reward / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, action, next_state, reward):
        child = Node(next_state, action=action, parent=self)
        child.total_reward = reward
        self.children.append(child)
        return child

    def backpropagate(self, reward):
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

class MCTSAgent:
    def __init__(self, environment, num_simulations=100):
        self.env = environment
        self.num_simulations = num_simulations

    def choose_action(self, state):
        root = Node(state)  
        for _ in range(self.num_simulations):
            node = root
            path = [node]
            env_copy = self.create_env_copy(self.env)

            # Selection
            while not node.is_terminal() and node.is_fully_expanded(env_copy):
                node = node.best_child()
                action = node.action
                _, _, done, _ = env_copy.step(action)
                if done:
                    break
                path.append(node)

            # Expansion
            if not node.is_terminal():
                possible_actions = node.get_possible_actions(env_copy)
                if possible_actions:
                    action = random.choice(possible_actions)
                    next_state, reward, done, _ = env_copy.step(action)
                    new_node = node.expand(action, next_state, reward)
                    path.append(new_node)

            # Simulation
            total_reward = self.rollout(env_copy, path[-1].state)

            # Backpropagation
            for node in reversed(path):
                node.backpropagate(total_reward)

        # Choose the best action
        return root.best_child(c_param=0).action

    def rollout(self, env_copy, state):
        total_reward = 0
        done = False
        while not done:
            action = env_copy.sample_action()
            state, reward, done, _ = env_copy.step(action)
            total_reward += reward
        return total_reward

    def create_env_copy(self, env):
        env_copy = environment.Environment("mapa_mundi_binario.jpg")
        env_copy.port_positions = env.port_positions.copy()
        env_copy.port_fuel = env.port_fuel.copy()
        env_copy.port_cargo = env.port_cargo.copy()
        env_copy.ship_position = env.ship_position.copy()
        env_copy.cargo = env.cargo
        env_copy.fuel = env.fuel
        env_copy.origin_port_index = env.origin_port_index
        env_copy.destination_port_index = env.destination_port_index
        env_copy.np_game = env.np_game.copy()
        return env_copy

def train_mcts_agent(env, episodes, simulations_per_episode):
    agent = MCTSAgent(env, num_simulations=simulations_per_episode)
    episode_rewards = []  
    episode_logs = []  

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        state = env.reset()
        total_reward = 0
        logs = []  # Log details for this episode

        for step in range(100):  
            current_port = state['ship']['origin_port_index']
            current_location = state['ship']['position']

            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            env.render_real_time()

            logs.append({
                "step": step + 1,
                "current_port": current_port,
                "current_location": current_location,
                "action": action,
                "next_location": next_state['ship']['position'],
                "reward": reward
            })

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        episode_logs.append(logs)
        print(f"Episode {episode + 1} finished. Total Reward: {total_reward}")

    print("\nTraining Summary:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Rewards per Episode: {episode_rewards}")

    return agent, episode_rewards, episode_logs

if __name__ == "__main__":
    env = environment.Environment("mapa_mundi_binario.jpg")
    print("Adding ports to the environment...")
    env.add_port([41, 40])
    env.add_port([60, 22])
    env.add_port([78, 29])
    env.add_port([49, 72])
    env.add_port([62, 72])

    print("Ports added. Training MCTS agent...")
    trained_agent, rewards, logs = train_mcts_agent(env, episodes=10, simulations_per_episode=50)

    print("\nTraining Complete!")
    print(f"Average Reward over 10 Episodes: {np.mean(rewards):.2f}")

    # Display port logs for each episode
    for episode_idx, episode_log in enumerate(logs):
        print(f"\nEpisode {episode_idx + 1} Logs:")
        for step_log in episode_log:
            print(f"  Step {step_log['step']}:")
            print(f"    Current Port: {step_log['current_port']}")
            print(f"    Current Location: {step_log['current_location']}")
            print(f"    Action Taken: {step_log['action']}")
            print(f"    Next Location: {step_log['next_location']}")
            print(f"    Reward: {step_log['reward']}")
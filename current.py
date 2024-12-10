import numpy as np
import cv2
import random
from heapq import heappop, heappush

# Constants
COLOR_WATER = [255, 0, 0]
COLOR_GROUND = [37, 73, 141]
COLOR_TRAVEL = [0, 0, 255]
COLOR_PORT = [154, 152, 150]
COLOR_DESTINY_PORT = [0, 255, 0]
COLOR_STORM = [128, 128, 128]
COLOR_BOAT = [65, 138, 222]

ID_GROUND = 0
ID_WATER = 1
ID_PORT = 2
ID_DESTINY_PORT = 3
ID_BOAT = 5
ID_STORM = 6
ID_TRAVEL = 7

PORT_POSITION = [[41, 40], [60, 22], [78, 29], [49, 72], [62, 72]]
STORM_POSITION = [
    [[20, 30], [30, 40]],
    [[55, 65], [10, 20]],
    [[80, 90], [60, 70]],
    [[10, 20], [70, 80]],
    [[40, 50], [85, 95]],
]

SIZE_RENDER = (355, 533)
SIZE_GAME = (100, 100)
X_MIN = 100
X_MAX = 300
Y_MIN = 50
Y_MAX = 200

WORLD_MAP_PATH = "mapa_mundi_binario.jpg"  # Update with your path

class Environment:
    def __init__(self):
        self.np_game = cv2.imread(WORLD_MAP_PATH, cv2.IMREAD_GRAYSCALE)
        if self.np_game is None:
            raise FileNotFoundError(f"Cannot read the image at {WORLD_MAP_PATH}")

        self.np_game = self.np_game[Y_MIN:Y_MAX, X_MIN:X_MAX]
        self.np_game = cv2.resize(self.np_game, SIZE_GAME, interpolation=cv2.INTER_AREA)
        _, self.np_game = cv2.threshold(self.np_game, 128, 1, cv2.THRESH_BINARY)
        self.np_game = self.np_game.astype(int)

        for x, y in PORT_POSITION:
            if 0 <= x < SIZE_GAME[0] and 0 <= y < SIZE_GAME[1]:
                self.np_game[x, y] = ID_PORT

        self.state = []
        self.done = False
        self.reward = 0
        self.cargo = 0
        self.fuel = 100
        self.port_cargo = {}
        self.storm_active = {}
        self.storm_size = {}
        self.original_storm_positions = STORM_POSITION.copy()
        self.storm_movement_range = {}
        self.current_port_index = 0
        
        self.initialize_ports()
        self.initialize_storms()

    def initialize_ports(self):
        for i in range(len(PORT_POSITION)):
            self.port_cargo[i] = [random.randint(5, 20), random.randint(5, 20)]

    def initialize_storms(self):
        for i in range(len(STORM_POSITION)):
            # Only activate the storm if it's on a water tile
            if self.is_storm_on_water(STORM_POSITION[i]):
                self.storm_active[i] = random.random() < 0.3
            else:
                self.storm_active[i] = False  # Deactivate if not on water
            self.storm_size[i] = random.randint(5, 10)
            self.storm_movement_range[i] = (
                (-2, 2),  # x movement range
                (-2, 2)   # y movement range
            )

    def is_storm_on_water(self, storm_pos):
        """Checks if a storm's initial position is entirely on water tiles."""
        for x in range(storm_pos[0][0], storm_pos[0][1]):
            for y in range(storm_pos[1][0], storm_pos[1][1]):
                if 0 <= x < SIZE_GAME[0] and 0 <= y < SIZE_GAME[1]:
                    if self.np_game[x, y] != ID_WATER:
                        return False
                else:
                    return False  # Out of bounds is considered not on water
        return True

    def is_valid_storm_move(self, new_pos):
        """Checks if a new storm position is entirely on water tiles."""
        for x in range(new_pos[0][0], new_pos[0][1]):
            for y in range(new_pos[1][0], new_pos[1][1]):
                if 0 <= x < SIZE_GAME[0] and 0 <= y < SIZE_GAME[1]:
                    if self.np_game[x, y] != ID_WATER:
                        return False
                else:
                    return False  # Out of bounds is considered invalid
        return True

    def update_storms(self):
        temp_storm_cells = {}

        for storm_index, active in self.storm_active.items():
            if active:
                x_range, y_range = self.storm_movement_range[storm_index]
                dx = random.randint(x_range[0], x_range[1])
                dy = random.randint(y_range[0], y_range[1])

                new_pos = [
                    [
                        self.original_storm_positions[storm_index][0][0] + dx,
                        self.original_storm_positions[storm_index][0][1] + dx,
                    ],
                    [
                        self.original_storm_positions[storm_index][1][0] + dy,
                        self.original_storm_positions[storm_index][1][1] + dy,
                    ],
                ]

                # Keep within bounds and ensure new position is on water
                new_pos[0][0] = max(0, min(new_pos[0][0], SIZE_GAME[0] - self.storm_size[storm_index]))
                new_pos[0][1] = max(0, min(new_pos[0][1], SIZE_GAME[0] - self.storm_size[storm_index]))
                new_pos[1][0] = max(0, min(new_pos[1][0], SIZE_GAME[1] - self.storm_size[storm_index]))
                new_pos[1][1] = max(0, min(new_pos[1][1], SIZE_GAME[1] - self.storm_size[storm_index]))

                # If the new position is valid (entirely on water), update the storm's position
                if self.is_valid_storm_move(new_pos):
                    STORM_POSITION[storm_index] = new_pos

                    for x in range(new_pos[0][0], new_pos[0][1]):
                        for y in range(new_pos[1][0], new_pos[1][1]):
                            if 0 <= x < SIZE_GAME[0] and 0 <= y < SIZE_GAME[1]:
                                temp_storm_cells[(x, y)] = True
                else:
                    # If the move is invalid, deactivate the storm
                    self.storm_active[storm_index] = False

        # Reset all ID_STORM cells to ID_WATER
        for x in range(SIZE_GAME[0]):
            for y in range(SIZE_GAME[1]):
                if self.np_game[x, y] == ID_STORM:
                    self.np_game[x, y] = ID_WATER

        # Update cells that are going to be covered by storms to ID_STORM
        for (x, y) in temp_storm_cells:
            self.np_game[x, y] = ID_STORM

    def calculate_fuel_cost(self, start, end):
        distance = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return distance * (1 + random.uniform(-0.1, 0.1))

    def render_real_time(self):
        np_render = np.zeros((self.np_game.shape[0], self.np_game.shape[1], 3), dtype="uint8")
        
        np_render[self.np_game == ID_WATER] = COLOR_WATER
        np_render[self.np_game == ID_GROUND] = COLOR_GROUND

        for storm_index, active in self.storm_active.items():
            if active:
                for x in range(STORM_POSITION[storm_index][0][0], STORM_POSITION[storm_index][0][1]):
                    for y in range(STORM_POSITION[storm_index][1][0], STORM_POSITION[storm_index][1][1]):
                        if 0 <= x < SIZE_GAME[0] and 0 <= y < SIZE_GAME[1]:
                            np_render[x, y] = COLOR_STORM

        np_render[self.np_game == ID_PORT] = COLOR_PORT
        np_render[self.np_game == ID_DESTINY_PORT] = COLOR_DESTINY_PORT
        np_render[self.np_game == ID_TRAVEL] = COLOR_TRAVEL

        x, y = np.where(self.np_game == ID_BOAT)
        if len(x) > 0 and len(y) > 0:
            x, y = x[0], y[0]
            np_render[x, y - 3:y + 3, :] = COLOR_BOAT
            np_render[x + 1, y - 2:y + 2, :] = COLOR_BOAT
            np_render[x - 4:x, y, :] = [0, 0, 0]
            np_render[x - 4:x - 1, y + 1, :] = [255, 255, 255]
            np_render[x - 3, y + 2, :] = [255, 255, 255]

        frame_resized = cv2.resize(np_render, SIZE_RENDER, interpolation=cv2.INTER_AREA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Display stats
        cv2.putText(frame_resized, f"Fuel: {self.fuel:.2f}", (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_resized, f"Cargo: {self.cargo}", (10, 40), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_resized, f"Reward: {self.reward}", (10, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display port info
        port_info_y = 80
        for port_index, cargo in self.port_cargo.items():
            cv2.putText(frame_resized, f"P{port_index + 1} - Drop: {cargo[0]}, Pick: {cargo[1]}", 
                       (10, port_info_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            port_info_y += 20

        cv2.imshow("Environment", frame_resized)
        key = cv2.waitKey(100)
        if key == ord("q"):
            cv2.destroyAllWindows()
            exit()

    def reset(self):
        self.np_game[((self.np_game == ID_BOAT) + (self.np_game == ID_TRAVEL))] = ID_WATER
        self.done = False
        self.reward = 0
        self.cargo = 0
        self.fuel = 100
        self.current_port_index = 0
        self.initialize_ports()
        self.initialize_storms()

        destination_port = PORT_POSITION[self.current_port_index]
        self.np_game[destination_port[0], destination_port[1]] = ID_BOAT
        self.state = destination_port

        return self.state

    def update_cargo_at_port(self):
        self.port_cargo[self.current_port_index] = [random.randint(5, 20), random.randint(5, 20)]

    def is_in_storm(self, x, y):
        for storm_index, active in self.storm_active.items():
            if active:
                if (STORM_POSITION[storm_index][0][0] <= x < STORM_POSITION[storm_index][0][1] and 
                    STORM_POSITION[storm_index][1][0] <= y < STORM_POSITION[storm_index][1][1]):
                    return True
        return False

    def step(self):
        ship_x, ship_y = self.state
        self.current_port_index = (self.current_port_index + 1) % len(PORT_POSITION)
        target_port = PORT_POSITION[self.current_port_index]

        dx, dy = target_port[0] - ship_x, target_port[1] - ship_y
        move_x, move_y = np.sign(dx), np.sign(dy)

        if 0 <= ship_x + move_x < SIZE_GAME[0] and 0 <= ship_y + move_y < SIZE_GAME[1]:
            new_x, new_y = ship_x + move_x, ship_y + move_y
            fuel_cost = self.calculate_fuel_cost([ship_x, ship_y], [new_x, new_y])

            if self.fuel >= fuel_cost:
                if self.is_in_storm(new_x, new_y):
                    self.reward -= 5
                    if random.random() < 0.03:
                        self.cargo = 0
                        self.reward -= 50
                        print("Lost all cargo in the storm!")

                    if random.random() < 0.05:
                        move_x, move_y = 0, 0

                if self.np_game[new_x, new_y] == ID_GROUND:
                    self.reward -= 5
                else:
                    self.np_game[ship_x, ship_y] = ID_TRAVEL
                    self.np_game[new_x, new_y] = ID_BOAT
                    self.state = [new_x, new_y]
                    self.fuel -= fuel_cost
                    self.reward -= 1

                    if self.state == target_port:
                        drop_off = self.port_cargo[self.current_port_index][0]
                        pick_up = self.port_cargo[self.current_port_index][1]

                        self.cargo -= drop_off
                        self.cargo = max(0, self.cargo)
                        self.cargo += pick_up

                        self.update_cargo_at_port()
                        self.done = True
                        self.reward += 100
            else:
                self.reward -= 10
                self.done = True
        else:
            self.reward -= 10

        return self.state, self.reward, self.done

if __name__ == "__main__":
    env = Environment()
    state = env.reset()

    try:
        for _ in range(1000):
            state, reward, done = env.step()
            env.update_storms()
            env.render_real_time()
            if done:
                print(f"Completed the journey with reward: {reward}")
                state = env.reset()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
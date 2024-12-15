import numpy as np
import cv2
import random

from .type import Color, Entity
from .util import calculate_euclidean_distance

class Environment:
    def __init__(self, map_path, game_size=(100,100)):
        # Configurations
        self.size_game = game_size

        self.port_positions = []
        self.port_cargo = []

        self.state = []
        self.done = False
        self.reward = 0
        self.cargo = 0
        self.fuel = 100
        self.current_port_index = 0
        
        self.storm_positions = []
        self.storm_active = {}
        self.storm_size = {}
        self.storm_movement_range = {}
        self.original_storm_positions = self.storm_positions.copy()

        self.initialize_map(map_path)

    def initialize_map(self, map_path):
        self.np_game = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.np_game is None:
            raise FileNotFoundError(f"Cannot read the image at {map_path}")
        
        x_min, x_max = 100, 300
        y_min, y_max = 50, 200

        self.np_game = self.np_game[y_min:y_max, x_min:x_max]
        self.np_game = cv2.resize(self.np_game, self.size_game, interpolation=cv2.INTER_AREA)
        _, self.np_game = cv2.threshold(self.np_game, 128, 1, cv2.THRESH_BINARY)
        self.np_game = self.np_game.astype(int)
    
    def add_port(self, pos):
        x, y = pos[0], pos[1]
        if not self.is_within_map(x, y):
            raise ValueError("Coordinates not within map")

        self.port_positions.append(pos)
        self.port_cargo.append([random.randint(5, 20), random.randint(5, 20)])
        self.np_game[x, y] = Entity.PORT
    
    def remove_port(self, idx):
        if 0 <= idx < self.port_positions and 0 <= idx < self.port_cargo:
            x, y = self.port_positions[idx][0], self.port_positions[idx][1]
            self.np_game[x, y] = Entity.GROUND
            self.port_cargo.pop(idx)
            return self.port_positions.pop(idx)
        else:
            raise IndexError("Invalid index")
    
    def add_storm(self, pos):
        i = len(self.storm_positions)
        self.storm_positions.append(pos)
        self.original_storm_positions.append(pos)

        if self.is_storm_on_water(pos):
            self.storm_active[i] = random.random() < 0.3
        else:
            self.storm_active[i] = False  # Deactivate if not on water

        self.storm_size[i] = random.randint(5, 10)
        self.storm_movement_range[i] = (
            (-2, 2),  # x movement range
            (-2, 2)   # y movement range
        )

    def remove_storm(self, idx):
        if 0 <= idx < self.storm_positions and 0 <= idx < self.original_storm_positions:
            del self.storm_active[idx]
            del self.storm_size[idx]
            del self.storm_movement_range[idx]

            self.storm_positions.pop(idx)

            return self.original_storm_positions(idx)
        else:
            raise IndexError("Invalid index")

    def is_within_map(self, x, y):
        """
        Checks if the given coordinates are within the game map boundaries.

        This method determines whether the specified x and y coordinates 
        fall inside the game's map dimensions. It verifies that the coordinates 
        are non-negative and less than the map's width and height.

        Args:
            x (int): The x-coordinate to check.
            y (int): The y-coordinate to check.

        Returns:
            bool: True if the coordinates are within the map boundaries, 
                False otherwise.

        Example:
            # Assuming self.size_game is (10, 10)
            self.is_within_map(5, 5)   # Returns True
            self.is_within_map(-1, 3)  # Returns False
            self.is_within_map(10, 7)  # Returns False
        """
        within_width = 0 <= x < self.size_game[0]
        within_height = 0 <= y < self.size_game[1]
        return within_width and within_height

    def is_storm_on_water(self, storm_pos):
        """Checks if a storm's initial position is entirely on water tiles."""
        for x in range(storm_pos[0][0], storm_pos[0][1]):
            for y in range(storm_pos[1][0], storm_pos[1][1]):
                if self.is_within_map(x, y):
                    if self.np_game[x, y] != Entity.WATER:
                        return False
                else:
                    return False  # Out of bounds is considered not on water
        return True

    def is_valid_storm_move(self, new_pos):
        """Checks if a new storm position is entirely on water tiles."""
        for x in range(new_pos[0][0], new_pos[0][1]):
            for y in range(new_pos[1][0], new_pos[1][1]):
                if self.is_within_map(x, y):
                    if self.np_game[x, y] != Entity.WATER:
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
                new_pos[0][0] = max(0, min(new_pos[0][0], self.size_game[0] - self.storm_size[storm_index]))
                new_pos[0][1] = max(0, min(new_pos[0][1], self.size_game[0] - self.storm_size[storm_index]))
                new_pos[1][0] = max(0, min(new_pos[1][0], self.size_game[1] - self.storm_size[storm_index]))
                new_pos[1][1] = max(0, min(new_pos[1][1], self.size_game[1] - self.storm_size[storm_index]))

                # If the new position is valid (entirely on water), update the storm's position
                if self.is_valid_storm_move(new_pos):
                    self.storm_positions[storm_index] = new_pos

                    for x in range(new_pos[0][0], new_pos[0][1]):
                        for y in range(new_pos[1][0], new_pos[1][1]):
                            # TODO: Possible unecessary check
                            if self.is_within_map(x, y):
                                temp_storm_cells[(x, y)] = True
                else:
                    # If the move is invalid, deactivate the storm
                    self.storm_active[storm_index] = False

        # Reset all Entity.STORM cells to Entity.WATER
        for x in range(self.size_game[0]):
            for y in range(self.size_game[1]):
                if self.np_game[x, y] == Entity.STORM:
                    self.np_game[x, y] = Entity.WATER

        # Update cells that are going to be covered by storms to Entity.STORM
        for (x, y) in temp_storm_cells:
            self.np_game[x, y] = Entity.STORM

    def calculate_fuel_cost(self, start, end):
        return calculate_euclidean_distance(start, end) * (1 + random.uniform(-0.1, 0.1))

    def render_real_time(self, render_size=(355, 533)):
        np_render = np.zeros((self.np_game.shape[0], self.np_game.shape[1], 3), dtype="uint8")
        
        np_render[self.np_game == Entity.WATER] = Color.WATER
        np_render[self.np_game == Entity.GROUND] = Color.GROUND

        for storm_index, active in self.storm_active.items():
            if active:
                for x in range(self.storm_positions[storm_index][0][0], self.storm_positions[storm_index][0][1]):
                    for y in range(self.storm_positions[storm_index][1][0], self.storm_positions[storm_index][1][1]):
                        if self.is_within_map(x, y):
                            np_render[x, y] = Color.STORM

        np_render[self.np_game == Entity.PORT] = Color.PORT
        np_render[self.np_game == Entity.DESTINY_PORT] = Color.DESTINY_PORT
        np_render[self.np_game == Entity.TRAVEL] = Color.TRAVEL

        x, y = np.where(self.np_game == Entity.BOAT)
        if len(x) > 0 and len(y) > 0:
            x, y = x[0], y[0]
            np_render[x, y - 3:y + 3, :] = Color.BOAT
            np_render[x + 1, y - 2:y + 2, :] = Color.BOAT
            np_render[x - 4:x, y, :] = Color.BLACK
            np_render[x - 4:x - 1, y + 1, :] = Color.WHITE
            np_render[x - 3, y + 2, :] = Color.WHITE

        frame_resized = cv2.resize(np_render, render_size, interpolation=cv2.INTER_AREA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Display stats
        cv2.putText(frame_resized, f"Fuel: {self.fuel:.2f}", (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_resized, f"Cargo: {self.cargo}", (10, 40), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_resized, f"Reward: {self.reward}", (10, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display port info
        port_info_y = 80
        for port_index, cargo in enumerate(self.port_cargo):
            cv2.putText(frame_resized, f"P{port_index + 1} - Drop: {cargo[0]}, Pick: {cargo[1]}", 
                       (10, port_info_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            port_info_y += 20

        cv2.imshow("Environment", frame_resized)
        key = cv2.waitKey(100)
        if key == ord("q"):
            cv2.destroyAllWindows()
            exit()

    def reset(self):
        self.np_game[((self.np_game == Entity.BOAT) + (self.np_game == Entity.TRAVEL))] = Entity.WATER
        self.done = False
        self.reward = 0
        self.cargo = 0
        self.fuel = 100
        self.current_port_index = 0

        destination_port = self.port_positions[self.current_port_index]
        self.np_game[destination_port[0], destination_port[1]] = Entity.BOAT
        self.state = destination_port

        return self.state

    def update_cargo_at_port(self):
        self.port_cargo[self.current_port_index] = [random.randint(5, 20), random.randint(5, 20)]

    def is_in_storm(self, x, y):
        for storm_index, active in self.storm_active.items():
            if active:
                if (self.storm_positions[storm_index][0][0] <= x < self.storm_positions[storm_index][0][1] and 
                    self.storm_positions[storm_index][1][0] <= y < self.storm_positions[storm_index][1][1]):
                    return True
        return False

    def step(self):
        ship_x, ship_y = self.state
        self.current_port_index = (self.current_port_index + 1) % len(self.port_positions)
        target_port = self.port_positions[self.current_port_index]

        dx, dy = target_port[0] - ship_x, target_port[1] - ship_y
        move_x, move_y = np.sign(dx), np.sign(dy)
        new_x, new_y = ship_x + move_x, ship_y + move_y

        if self.is_within_map(new_x, new_y):
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

                if self.np_game[new_x, new_y] == Entity.GROUND:
                    self.reward -= 5
                else:
                    self.np_game[ship_x, ship_y] = Entity.TRAVEL
                    self.np_game[new_x, new_y] = Entity.BOAT
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
            raise ValueError("Move is out of bounds")

        return self.state, self.reward, self.done
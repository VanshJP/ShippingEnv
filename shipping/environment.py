import numpy as np
import cv2
import random

from .type import Color, Entity, ActionType, ShipMove
from .util import calculate_euclidean_distance, normalize

class Environment:
    def __init__(self, map_path, game_size=(100,100)):
        # ** Configurations
        self.size_game = game_size

        self.port_positions = []
        self.port_fuel = []
        self.port_cargo = []

        self.ship_position = []
        self.cargo = 0
        self.fuel = 200
        self.destination_port_index = None
        
        self.storm_positions = []
        self.storm_active = {}
        self.storm_size = {}
        self.storm_movement_range = {}
        self.original_storm_positions = self.storm_positions.copy()

        self._initialize_map(map_path)

    def _initialize_map(self, map_path):
        self.np_game = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.np_game is None: raise FileNotFoundError(f"Cannot read the image at {map_path}")
        
        x_min, x_max = 100, 300
        y_min, y_max = 50, 200

        self.np_game = self.np_game[y_min:y_max, x_min:x_max]
        self.np_game = cv2.resize(self.np_game, self.size_game, interpolation=cv2.INTER_AREA)
        _, self.np_game = cv2.threshold(self.np_game, 128, 1, cv2.THRESH_BINARY)
        self.np_game = self.np_game.astype(int)
    
    def add_port(self, pos):
        x, y = pos[0], pos[1]
        if not self._is_within_map(x, y):
            raise ValueError("Coordinates not within map")

        self.port_positions.append(pos)
        self.port_fuel.append(random.randint(5, 20))
        self.port_cargo.append(random.randint(5, 20))
        self.np_game[x, y] = Entity.PORT
    
    def remove_port(self, idx):
        if 0 <= idx < self.port_positions and 0 <= idx < self.port_cargo:
            x, y = self.port_positions[idx]
            self.np_game[x, y] = Entity.GROUND
            self.port_fuel.pop(idx)
            self.port_cargo.pop(idx)
            return self.port_positions.pop(idx)
        else:
            raise IndexError("Invalid index")
    
    def add_storm(self, pos):
        i = len(self.storm_positions)
        self.storm_positions.append(pos)
        self.original_storm_positions.append(pos)

        if self._is_storm_on_water(pos):
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

    def _is_within_map(self, x, y):
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
            self._is_within_map(5, 5)   # Returns True
            self._is_within_map(-1, 3)  # Returns False
            self._is_within_map(10, 7)  # Returns False
        """
        within_width = 0 <= x < self.size_game[0]
        within_height = 0 <= y < self.size_game[1]
        return within_width and within_height

    def _is_storm_on_water(self, storm_pos):
        """Checks if a storm's initial position is entirely on water tiles."""
        for x in range(storm_pos[0][0], storm_pos[0][1]):
            for y in range(storm_pos[1][0], storm_pos[1][1]):
                if self._is_within_map(x, y):
                    if self.np_game[x, y] != Entity.WATER:
                        return False
                else:
                    return False  # Out of bounds is considered not on water
        return True

    def _is_valid_storm_move(self, new_pos):
        """Checks if a new storm position is entirely on water tiles."""
        for x in range(new_pos[0][0], new_pos[0][1]):
            for y in range(new_pos[1][0], new_pos[1][1]):
                if self._is_within_map(x, y):
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
                if self._is_valid_storm_move(new_pos):
                    self.storm_positions[storm_index] = new_pos

                    for x in range(new_pos[0][0], new_pos[0][1]):
                        for y in range(new_pos[1][0], new_pos[1][1]):
                            # TODO: Possible unecessary check
                            if self._is_within_map(x, y):
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

    def _calculate_fuel_cost(self, start, end):
        return calculate_euclidean_distance(start, end) * (1 + random.uniform(-0.1, 0.1))

    def render_real_time(self, render_size=(355, 533)):
        np_render = np.zeros((self.np_game.shape[0], self.np_game.shape[1], 3), dtype="uint8")
        
        np_render[self.np_game == Entity.WATER] = Color.WATER
        np_render[self.np_game == Entity.GROUND] = Color.GROUND

        for storm_index, active in self.storm_active.items():
            if active:
                for x in range(self.storm_positions[storm_index][0][0], self.storm_positions[storm_index][0][1]):
                    for y in range(self.storm_positions[storm_index][1][0], self.storm_positions[storm_index][1][1]):
                        if self._is_within_map(x, y):
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

        # Display port info
        port_info_y = 80
        for idx in range(len(self.port_positions)):
            cv2.putText(frame_resized, f"P{idx + 1} - Cargo: {self.port_cargo[idx]} - Fuel: {self.port_fuel[idx]}", 
                       (10, port_info_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            port_info_y += 20

        cv2.imshow("Environment", frame_resized)
        key = cv2.waitKey(100)
        if key == ord("q"):
            cv2.destroyAllWindows()
            exit()


    def _is_in_storm(self, x, y):
        for storm_index, active in self.storm_active.items():
            if active:
                if (self.storm_positions[storm_index][0][0] <= x < self.storm_positions[storm_index][0][1] and 
                    self.storm_positions[storm_index][1][0] <= y < self.storm_positions[storm_index][1][1]):
                    return True
        return False
    
    def _is_ship_at_port(self):
        for idx, port_position in enumerate(self.port_positions):
            if port_position == self.port_positions:
                return idx
        return None
    
    def _sample_random_port(self):
        if len(self.port_positions) == 0: raise Exception("No ports available")
        random_destination_index = random.randint(0, len(self.port_positions) - 1)

        # ** Choose a new port rather than a port ship is currently on
        while self.port_positions[random_destination_index] == self.ship_position or random_destination_index == self.destination_port_index:
            random_destination_index = random.randint(0, len(self.port_positions) - 1)

        return random_destination_index
    
    def _sample_random_cargo(self, port_idx):
        return random.randint(1, self.port_cargo[port_idx])

    def _sample_random_fuel(self, port_idx):
        return random.randint(1, self.fuel[port_idx])
    
    def _sample_random_move(self):
        moves_list = [ShipMove.NORTH, ShipMove.EAST, ShipMove.SOUTH, ShipMove.WEST]
        return random.choice(moves_list)
    
    def _calculate_cargo_loss(self):
        """
        Generate cargo loss using a custom probabilistic approach with random.
        
        Args:
            total_cargo (int): Total amount of cargo
        
        Returns:
            int: Amount of cargo lost
        """
        # Generate a random number to determine loss type
        loss_type = random.random()

        # Low probability of losing nothing (10% chance)
        if loss_type < 0.1:
            return 0
        
        # Low probability of losing everything (10% chance)
        elif loss_type > 0.9:
            return self.cargo
        
        # Medium probability of losing some cargo (80% of cases)
        else:
            # Use beta distribution to create a more nuanced loss
            # beta generates a value between 0 and 1
            beta_loss = random.betavariate(2, 2)
            
            # Scale the beta distribution to cargo amount
            loss = int(beta_loss * self.cargo)
            
            return loss
    
    def _build_state(self):
        ship = {
            "position": self.ship_position,
            "fuel": self.fuel,
            "cargo": self.fuel,
            "destination_port_index": self.destination_port_index
        }

        ports = []

        for idx, port_position in enumerate(self.port_positions):
            current_port = {
                "index": idx,
                "position": port_position,
                "fuel": self.port_fuel[idx],
                "cargo": self.port_cargo[idx]
            }
            ports.append(current_port)

        return {
            "ship": ship,
            "ports": ports
        }

    def reset(self):
        self.np_game[((self.np_game == Entity.BOAT) + (self.np_game == Entity.TRAVEL))] = Entity.WATER

        self.cargo = 0
        self.fuel = 100
        self.destination_port_index = self._sample_random_port()

        destination_port = self.port_positions[self.destination_port_index]
        self.np_game[destination_port[0], destination_port[1]] = Entity.BOAT
        self.ship_position = destination_port

        return self._build_state()

    def sample_action(self):
        current_port_idx = self._is_ship_at_port()
        if self.destination_port_index == None:
            # If no destination port selected, randomly return an index
            return [ActionType.SELECT_PORT, self._sample_random_port()]
        elif self.cargo == 0 and current_port_idx is not None:
            # If destination port selected, but no cargo
            return [ActionType.TAKE_CARGO, self._sample_random_cargo(current_port_idx)]
        elif self.fuel == 0 and current_port_idx is not None:
            # If destination port selected, but no fuel
            return [ActionType.TAKE_FUEL, self._sample_random_fuel(current_port_idx)]
        else:
            # If destination port is selected, move a random location
            return [ActionType.MOVE_SHIP, self._sample_random_move()]
    
    def _select_port(self, value):
        if 0 <= value < len(self.port_positions):
            if self.destination_port_index == value: raise Exception("Destination port must be different from current one")
            self.destination_port_index = value
        else: raise IndexError("Port index is out of range")

        return 0, False
        
    def _move_ship(self, move):
        if len(move) != 2: raise ValueError("Move needs to be of format (x,y) or [x, y]")
        # Select port first before moving ship
        if self.destination_port_index == None: raise Exception("Cannot move without destination port")

        reward, done = 0, False

        move_x, move_y = move
        ship_x, ship_y = self.ship_position
        new_x, new_y = ship_x + move_x, ship_y + move_y

        if not self._is_within_map(new_x, new_y): raise ValueError("Move is out of range")

        # ** Check if ship has enough fuel to move to desired space
        fuel_cost = self._calculate_fuel_cost([ship_x, ship_y], [new_x, new_y])
        if self.fuel < fuel_cost:
            reward -= 10
            done = True

        # ** Penalize ship if within storm [remove this if storm is no longer considered]
        if self._is_in_storm(new_x, new_y):
            reward -= 5
            if random.random() < 0.03:
                self.cargo = 0
                reward -= 50
                print("Lost all cargo in the storm!")
            if random.random() < 0.05:
                move_x, move_y = 0, 0

        # ** Manuvering process
        if self.np_game[new_x, new_y] == Entity.GROUND:
            reward -= 5
        else: 
            # ** Move ship
            self.ship_position = [new_x, new_y]
            self.fuel -= fuel_cost
            reward -= 1

            # ** Update graphics
            self.np_game[ship_x, ship_y] = Entity.TRAVEL
            self.np_game[new_x, new_y] = Entity.BOAT

        # ** Losing cargo process
        min_capacity, max_capacity = 0, 50
        likelihood_of_losing_cargo = normalize(self.cargo, max_capacity, min_capacity)
        if random.random() <= likelihood_of_losing_cargo:
            cargo_loss = self._calculate_cargo_loss()
            self.cargo -= cargo_loss
            reward -= cargo_loss * -3

        if self.ship_position == self.port_positions[self.destination_port_index]:
            drop_off, pick_up = self.cargo, self.port_cargo[self.destination_port_index]

            # ** Process cargo
            self.cargo = 0
            reward += drop_off * 2
            self.cargo += pick_up

            self.destination_port_index = None
            reward += 100

        return reward, done
    
    def _take_cargo(self, value):
        current_port_idx = self._is_ship_at_port()
        if current_port_idx is None: raise Exception("Not currently at port")

        if 0 < value <= self.port_cargo[current_port_idx]: self.cargo = value
        else: raise ValueError("Invalid fuel amount")

        return 0, False

    def _take_fuel(self, value):
        current_port_idx = self._is_ship_at_port()
        if current_port_idx is None: raise Exception("Not currently at port")

        if 0 < value <= self.port_fuel[current_port_idx]: self.fuel = value
        else: raise ValueError("Invalid fuel amount")

        return 0, False

    def step(self, action):
        if len(self.port_positions) == 0: raise Exception("No ports available")

        reward, done, meta = 0, False, {}
        action_category, action_value = action
        match action_category:
            case ActionType.SELECT_PORT:
                reward, done = self._select_port(action_value)
            case ActionType.TAKE_CARGO:
                reward, done = self._take_cargo(action_value)
            case ActionType.TAKE_FUEL:
                reward, done = self._take_fuel(action_value)
            case ActionType.MOVE_SHIP:
                reward, done = self._move_ship(action_value)
            case _:
                raise ValueError("Action category unknown")
        
        return self._build_state(), reward, done, meta
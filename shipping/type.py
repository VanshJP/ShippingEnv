class Action:
    """
    Enum for action space where values are (x,y) tuple 
    """
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (-1, 0)
    WEST = (1, 0)

class Entity:
    """
    Enum for entity IDs where values are integers
    """

    GROUND = 0
    WATER = 1
    PORT = 2
    DESTINY_PORT = 3
    BOAT = 5
    STORM = 6
    TRAVEL = 7

    def __str__(self):
        return str(self.value)

class Color:
    """
    Enum for colors where values are 1D array with BGR values
    """

    WATER = [255, 0, 0]
    GROUND = [37, 73, 141]
    TRAVEL = [0, 0, 255]
    PORT = [154, 152, 150]
    DESTINY_PORT = [0, 255, 0]
    STORM = [128, 128, 128]
    BOAT = [65, 138, 222]
    WHITE = [255, 255, 255]
    BLACK = [0, 0, 0]
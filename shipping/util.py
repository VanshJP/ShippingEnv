import numpy as np

def calculate_euclidean_distance(start, end):
    return np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
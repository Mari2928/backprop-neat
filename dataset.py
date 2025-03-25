import random
import numpy as np
import math

# settings
n_size = 200 # training size
n_testSize = 200 # testing size
noise_level = 0.5

class Dataset:
    def __init__(self, task):
        if task == "xor":
            random.seed(42)
            self.train_data_points = generate_xor_data(num_points=n_size, noise=noise_level)
            self.test_data_points = generate_xor_data(num_points=n_testSize, noise=noise_level)
        elif task == "spiral":
            random.seed(42)
            self.train_data_points = generate_spiral_data(num_points=n_size, noise=noise_level)
            self.test_data_points = generate_spiral_data(num_points=n_testSize, noise=noise_level)
        elif task == "circle":
            random.seed(42)
            self.train_data_points = generate_circle_data(num_points=n_size, noise=noise_level)
            self.test_data_points = generate_circle_data(num_points=n_testSize, noise=noise_level)

class DataPoint:
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label

def randf(low, high):
    return random.uniform(low, high)

def randn(mean, std_dev):
    return random.normalvariate(mean, std_dev)

def generate_spiral_data(num_points=None, noise=0.5):
    particle_list = []
    N = num_points if num_points is not None else 100  # Default nSize of 100

    def gen_spiral(delta_t, label):
        n = N // 2
        for i in range(n):
            r = i / n * 6.0
            t = 1.75 * i / n * 2 * np.pi + delta_t
            x = r * np.sin(t) + randf(-1, 1) * noise
            y = r * np.cos(t) + randf(-1, 1) * noise
            particle_list.append(DataPoint(x, y, label))

    flip = 0  # or np.random.randint(0, 2)
    backside = 1 - flip
    gen_spiral(0, flip)  # Positive examples
    gen_spiral(np.pi, backside)  # Negative examples

    return particle_list

def generate_xor_data(num_points=None, noise=0.5):
    particle_list = []
    N = num_points if num_points is not None else 100  # Assuming default nSize of 100
    noise_value = noise

    for _ in range(N):
        x = randf(-5.0, 5.0) + randn(0, noise_value)
        y = randf(-5.0, 5.0) + randn(0, noise_value)
        label = 0
        if (x > 0 and y > 0) or (x < 0 and y < 0):
            label = 1
        particle_list.append(DataPoint(x, y, label))

    return particle_list

def generate_circle_data(num_points=None, noise=0.5):
    particle_list = []
    radius = 5.0

    # Default value if num_points is None
    N = num_points if num_points is not None else 100
    n = N // 2

    def get_circle_label(x, y):
        return 1 if (x * x + y * y < (radius * 0.5) * (radius * 0.5)) else 0

    # Generate positive points inside the circle
    for _ in range(n):
        r = random.uniform(0, radius * 0.5)
        angle = random.uniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        noise_x = random.uniform(-radius, radius) * noise / 3
        noise_y = random.uniform(-radius, radius) * noise / 3
        label = get_circle_label(x, y)
        particle_list.append(DataPoint(x + noise_x, y + noise_y, label))

    # Generate negative points outside the circle
    for _ in range(n):
        r = random.uniform(radius * 0.75, radius)
        angle = random.uniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        noise_x = random.uniform(-radius, radius) * noise / 3
        noise_y = random.uniform(-radius, radius) * noise / 3
        label = get_circle_label(x, y)
        particle_list.append(DataPoint(x + noise_x, y + noise_y, label))

    return particle_list


import json
import numpy as np


class OUNoise:
    """Implementação do ruído Ornstein-Uhlenbeck para exploração."""
    
    def __init__(self, action_dimension, category):
        self.action_dimension = action_dimension
        self.read_configs(category)
        self.set_configs()

    def read_configs(self, category):
        with open('configs/noise.json', 'r') as file:
            self.configs = json.load(file)
        self.configs = self.configs[category]

    def set_configs(self):
        self.multiplier = 1.0
        self.dt = self.configs['dt']
        self.mu = self.configs['mu']
        self.theta = self.configs['theta']
        self.sigma = self.configs['sigma']
        self.min_multiplier = self.configs['min_multiplier']
        self.iterations_to_minimum_multiplier = self.configs['iterations_to_minimum_multiplier']
        self.decay_rate = (self.multiplier - self.min_multiplier) / self.iterations_to_minimum_multiplier
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def decay(self):
        if self.multiplier > self.min_multiplier:
            self.multiplier -= self.decay_rate
        else:
            self.multiplier = self.min_multiplier
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state * self.multiplier

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
        self.mu = self.configs['mu']
        self.theta = self.configs['theta']
        self.sigma = self.configs['sigma']
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

import json
import numpy as np


class OUNoise:
    """Implementação do ruído Ornstein-Uhlenbeck para exploração."""
    
    def __init__(self, action_dimension):
        self.read_configs()
        self.mu = self.configs['ou_noise']['mu']
        self.theta = self.configs['ou_noise']['theta']
        self.sigma = self.configs['ou_noise']['sigma']
        self.decay_factor = self.configs['ou_noise']['inicial_decay_factor']
        self.min_decay_factor = self.configs['ou_noise']['min_decay_factor']
        self.iterations_to_minimum = self.configs['ou_noise']['iterations_to_minimum']
        self.decay_rate = self.get_decay_rate()
        
        self.action_dimension = action_dimension
        self.reset()
    
    def read_configs(self):
        with open('configs/noise.json', 'r') as file:
            self.configs = json.load(file)

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
    
    def get_decay_rate(self):
        decay_rate = self.decay_factor - self.min_decay_factor
        decay_rate /= self.iterations_to_minimum
        return decay_rate

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state* self.decay_factor

    def decay(self):
        if self.decay_factor == 0.1:
            return
        
        self.decay_factor -= self.decay_rate
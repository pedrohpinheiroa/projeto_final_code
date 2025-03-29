import json
import numpy as np
from typing import Dict

class Reward:
    
    def __init__(self):
        self.read_configs()
        self.set_configs()

    def read_configs(self):
        with open('configs/reward.json', 'r') as file:
            self.configs = json.load(file)

    def set_configs(self):
        self.position_weight = self.configs['position_weight']
        self.velocity_weight = self.configs['velocity_weight']
        self.knock_penalty = self.configs['knock_penalty']
        self.over_pwm_penalty = self.configs['over_pwm_penalty']
        self.stability_reward = self.configs['stability_reward']
        self.stability_threshold = self.configs['stability_threshold']

    def position_reward(self, state: Dict) -> float:
        position_error = -np.abs(state['position'])
        position_reward = np.exp(5.0 * position_error)
        reward = self.position_weight * position_reward
        return reward
    
    def velocity_reward(self, state: Dict) -> float:
        velocity_penalty = -np.abs(state['velocity'])
        reward = self.velocity_weight * velocity_penalty
        return reward
    
    def check_knock(self, state: Dict) -> float:
        if state['knock']:
            return self.knock_penalty
        return 0.0

    def check_over_pwm(self, state: Dict) -> float:
        if state['over_pwm']:
            return self.over_pwm_penalty
        return 0.0

    def check_stability(self, state: Dict) -> float:
        if (abs(state['position']) < self.stability_threshold and 
            abs(state['velocity']) < self.stability_threshold):
            return self.stability_reward
        return 0.0

    def get(self, state: Dict) -> float:
        reward = 0.0
        reward += self.position_reward(state)
        reward += self.velocity_reward(state)
        reward += self.check_knock(state)
        reward += self.check_over_pwm(state)
        reward += self.check_stability(state)            
        return reward
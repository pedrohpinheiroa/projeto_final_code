import json
import numpy as np
from typing import Dict

class Reward:
    
    def __init__(self, max_angle):
        self.max_angle = max_angle
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
        self.under_pwm_penalty = self.configs['under_pwm_penalty']
        self.stability_reward = self.configs['stability_reward']
        self.stability_threshold = self.configs['stability_threshold']

    def position_reward(self, state: Dict) -> float:
        position_reward = np.abs(state['position']) / self.max_angle
        position_reward = np.round(position_reward, 3)
        position_reward = np.clip(position_reward, 0.0, 1.0)
        position_reward = -position_reward ** 2
        reward = self.position_weight * position_reward
        reward = np.round(reward, 3)
        return reward
    
    def velocity_reward(self, state: Dict) -> float:
        velocity_penalty = -abs(state['velocity']) ** 2
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

    def check_under_pwm(self, state: Dict) -> float:
        if state['under_pwm']:
            return self.under_pwm_penalty
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
        reward += self.check_stability(state)
        # reward += self.check_over_pwm(state)
        # reward += self.check_under_pwm(state)
        return reward
import json
from typing import Dict, Tuple

import numpy as np


class Physics:

    def __init__(self, min_pwm, max_pwm):
        self.read_configs()
        self.set_configs(min_pwm, max_pwm)

    def read_configs(self):
        with open('configs/physics.json', 'r') as file:
            self.configs = json.load(file)

    def set_configs(self, min_pwm, max_pwm):
        self.gravity = self.configs['gravity']
        self.time_step = self.configs['time_step']

        self.bar_mass = self.configs['bar_mass']
        self.bar_length = self.configs['bar_length']
        self.motor_mass = self.configs['motor_mass']

        self.inertia = (self.bar_mass * (self.bar_length ** 2) / 3)
        self.inertia += (2 * self.motor_mass * (self.bar_length ** 2))

        self.min_pwm = min_pwm
        self.max_pwm = max_pwm
        self.pwm_range = self.max_pwm - self.min_pwm

    def _get_pwms(self, action: float) -> Tuple[float, float]:
        action = action[0]
        if action < 0:
            right_pwm = self.min_pwm
            left_pwm = self.min_pwm + abs(action) * self.pwm_range
        elif action > 0:
            left_pwm = self.min_pwm
            right_pwm = self.min_pwm + abs(action) * self.pwm_range
        else:
            left_pwm = self.min_pwm
            right_pwm = self.min_pwm
        
        left_pwm = np.clip(left_pwm, self.min_pwm, self.max_pwm)
        right_pwm = np.clip(right_pwm, self.min_pwm, self.max_pwm)
        return np.float16(left_pwm), np.float16(right_pwm)

    def _get_forces(self, left_pwm: float, right_pwm: float)->Tuple[float, float]:
        left_force = self.gravity*(70*left_pwm - 1)/100
        right_force = self.gravity*(70*right_pwm - 1)/100
        return left_force, right_force
    
    def _get_torque(self, left_force: float, right_force: float)->Tuple[float, float]:
        left_torque = left_force * self.bar_length
        right_torque = right_force * self.bar_length
        torque = right_torque - left_torque
        return torque
    
    def _get_acceleration(self, torque: float)->float:
        return torque / self.inertia
    
    def _get_velocity(self, acceleration: float, velocity: float)->float:
        time_step = self.time_step/1000
        return velocity + acceleration * time_step
    
    def _get_position(self, velocity: float, position: float)->float:
        time_step = self.time_step/1000
        return position + velocity * time_step
    
    def apply_action(self, state:Dict, action: float):
        left_pwm, right_pwm = self._get_pwms(action)
        left_force, right_force = self._get_forces(left_pwm, right_pwm)
        torque = self._get_torque(left_force, right_force)
        acceleration = self._get_acceleration(torque)
        velocity = self._get_velocity(acceleration, state['velocity'])
        position = self._get_position(velocity, state['position'])
        state['time'] = state['time'] + self.time_step
        state['position'] = position
        state['velocity'] = velocity
        state['acceleration'] = acceleration
        state['torque'] = torque
        state['left_pwm'] = np.float16(left_pwm)
        state['right_pwm'] = np.float16(right_pwm)
        return state

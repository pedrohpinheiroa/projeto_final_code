import json
from typing import Dict, Tuple

import numpy as np


class Physics:

    def __init__(self):
        self.read_configs()
        self.set_configs()

    def read_configs(self):
        with open('configs/physics.json', 'r') as file:
            self.configs = json.load(file)

    def set_configs(self):
        self.gravity = self.configs['gravity']
        self.time_step = self.configs['time_step']

        self.bar_mass = self.configs['bar_mass']
        self.bar_length = self.configs['bar_length']
        self.motor_mass = self.configs['motor_mass']

        self.inertia = (self.bar_mass * (self.bar_length ** 2) / 3)
        self.inertia += (2 * self.motor_mass * (self.bar_length ** 2))

    def _get_forces(self, left_pwm: float, right_pwm: float)->Tuple[float, float]:
        left_force = self.gravity*(70*left_pwm - 1)/100
        right_force = self.gravity*(70*right_pwm - 1)/100
        return left_force, right_force
    
    def _get_torques(self, left_force: float, right_force: float)->Tuple[float, float]:
        left_torque = left_force * self.bar_length
        right_torque = right_force * self.bar_length
        return left_torque, right_torque
    
    def _get_acceleration(self, left_torque: float, right_torque: float)->float:
        torque = right_torque - left_torque
        return np.round(torque / self.inertia,1)
    
    def _get_velocity(self, acceleration: float, velocity: float)->float:
        time_step = self.time_step/1000
        return np.round(velocity + acceleration * time_step, 3)
    
    def _get_position(self, velocity: float, position: float)->float:
        time_step = self.time_step/1000
        return np.round(position + velocity * time_step, 4)
    
    def apply_action(self, state:Dict, action: Tuple[float, float]):
        left_pwm, right_pwm = action
        left_force, right_force = self._get_forces(left_pwm, right_pwm)
        left_torque, right_torque = self._get_torques(left_force, right_force)
        acceleration = self._get_acceleration(left_torque, right_torque)
        velocity = self._get_velocity(acceleration, state['velocity'])
        position = self._get_position(velocity, state['position'])
        state['time'] = state['time'] + self.time_step
        state['position'] = position
        state['velocity'] = velocity
        state['acceleration'] = acceleration
        state['left_pwm'] = np.float16(left_pwm)
        state['right_pwm'] = np.float16(right_pwm)
        return state

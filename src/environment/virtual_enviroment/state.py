import json
import numpy as np
from typing import Dict


class State:

    def __init__(self):
        self.read_configs()
        self.set_configs()
        self.reset()

    def read_configs(self):
        with open('configs/state.json', 'r') as file:
            self.configs = json.load(file)

    def set_configs(self):
        self.max_angle = np.round(np.radians(self.configs['max_angle']),2)
        self.pwm_min = self.configs['pwm_min']
        self.pwm_max = self.configs['pwm_max']
        self.max_episode_time = self.configs['max_episode_time']

    def reset(self):
        self.time = np.int32(0)
        self.position = np.float32(np.random.uniform(-self.max_angle, self.max_angle))
        self.velocity = np.float32(np.random.uniform(-1, 1))
        self.acceleration = np.float32(0.0)
        self.left_pwm = self.pwm_min
        self.right_pwm = self.pwm_min

        self.knock = False
        self.over_pwm = False
        self.under_pwm = False
        self.done = False

    def get(self)->Dict:
        return {
            "time":self.time, 
            "position":self.position, 
            "velocity":self.velocity, 
            "acceleration":self.acceleration, 
            "left_pwm":self.left_pwm, 
            "right_pwm":self.right_pwm, 
            "knock":self.knock, 
            "over_pwm":self.over_pwm, 
            "under_pwm":self.under_pwm,
            "done":self.done
        }

    def _handle_knock(self, state: Dict) -> Dict:
        if abs(state['position']) > self.max_angle:
            state['position'] = np.sign(state['position']) * self.max_angle
            state['velocity'] = np.float16(0.0)
            state['acceleration'] = np.float16(0.0)
            state['knock'] = True
            state['done'] = True

        return state

    def _handle_over_pwm(self, state: Dict) -> Dict:
        if state['left_pwm'] > self.pwm_max or state['right_pwm'] > self.pwm_max:
            state['over_pwm'] = True
        else:
            state['over_pwm'] = False
        return state
    
    def _handle_under_pwm(self, state: Dict) -> Dict:
        if state['left_pwm'] < self.pwm_min or state['right_pwm'] < self.pwm_min:
            state['under_pwm'] = True
        else:
            state['under_pwm'] = False
        return state

    def _handle_episode_time(self, state: Dict) -> Dict:
        if state['time'] > self.max_episode_time:
            state['done'] = True
        return state


    def set(self, state: Dict):
        state = self._handle_knock(state)
        state = self._handle_over_pwm(state)
        state = self._handle_under_pwm(state)
        state = self._handle_episode_time(state)
        for key in state.keys():
            setattr(self, key, state[key])

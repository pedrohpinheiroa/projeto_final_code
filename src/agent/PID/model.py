import numpy as np
import json

class Controller:
    def __init__(self):
        self.read_configs()
        self.set_configs()
        self.reset()

    def read_configs(self):
        with open('configs/model.json', 'r') as file:
            self.configs = json.load(file)
        self.configs = self.configs['PID']
    
    def set_configs(self):
        self.Kp = self.configs.get('Kp')
        self.Ki = self.configs.get('Ki')
        self.Kd = self.configs.get('Kd')
        self.setpoint = self.configs['setpoint']
        self.output_min = self.configs['output_min']
        self.output_max = self.configs['output_max']
        self.integral_decay = self.configs.get('integral_decay')
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_output = 0.0

    def act(self, state):
        position, velocity = state

        error = self.setpoint - position
        self.integral += error

        proporcional_value = self.Kp * error
        integral_value = self.Ki * self.integral
        derivative_value = self.Kd * (-velocity)

        output =  proporcional_value + integral_value + derivative_value
        output = np.clip(output, self.output_min, self.output_max)
        return np.array([output], dtype=np.float32)
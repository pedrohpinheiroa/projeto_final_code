from typing import Tuple
from src.visualization import SeesawVisualizer

from .state import State
from .physics import Physics
from .history import History
from .reward import Reward

class Seesaw():

    def __init__(self):
        self.state = State()
        self.physics = Physics(self.state.pwm_min, self.state.pwm_max)
        self.history = History()
        self.reward = Reward(max_angle=self.state.max_angle)
        self.visualizer = SeesawVisualizer()

    def reset(self):
        self.state.reset()
        self.history.clear()

    def get_state(self):
        return self.state.get()

    def step(self, action: Tuple[float, float]):
        state = self.get_state()
        new_state = self.physics.apply_action(state, action)
        self.state.set(new_state)

    def is_done(self):
        return self.state.done

    def get_reward(self):
        state = self.get_state()
        return self.reward.get(state)

    def save_in_history(self, state):
        self.history.add(state)

    def get_episode_history(self):
        return self.history.get()

    def render(self):
        if self.visualizer:
            state = self.get_state()
            angle_rad = state['position']
            self.visualizer.render(angle_rad=angle_rad, metrics_dict=state)

    def close(self):
        if self.visualizer:
            self.visualizer.close()



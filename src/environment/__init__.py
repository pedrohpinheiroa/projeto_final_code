from typing import Tuple

from .state import State
from .physics import Physics


class Seesaw():

    def __init__(self, state=None, physics=None):
        self.state = state or State()
        self.physics = physics or Physics()

    def reset(self):
        self.state.reset()

    def get_state(self):
        return self.state.get()

    def step(self, action: Tuple[float, float]):
        state = self.get_state()
        new_state = self.physics.apply_action(state, action)
        self.state.set(new_state)

    def render(self):
        pass

    def close(self):
        pass

    def get_episode_history(self):
        pass

    def get_reward(self):
        pass

    def get_done(self):
        pass



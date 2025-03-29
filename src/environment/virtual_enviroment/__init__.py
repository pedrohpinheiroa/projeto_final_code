from typing import Tuple

from .state import State
from .physics import Physics
from .history import History
from .reward import Reward

class Seesaw():

    def __init__(self, state=None, physics=None, history=None, reward=None):
        self.state = state or State()
        self.physics = physics or Physics()
        self.history = history or History()
        self.reward = reward or Reward()

    def reset(self):
        self.state.reset()

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
        pass

    def close(self):
        pass



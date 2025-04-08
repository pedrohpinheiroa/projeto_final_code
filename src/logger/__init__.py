from .metrics import Metrics
from .writer import Writer


class Logger:

    def __init__(self):
        self.writer = Writer()
        self.metrics = Metrics()
        self.reset()

    def reset(self, **kwargs):
        self.iteration_data = []

    def add_information(self):...

    def save_log(self, episode, reward, loss):
        self.writer.write_scalar('reward', reward, episode)
        self.writer.write_scalar('loss', loss, episode)
        self.metrics.episode_reward(reward)
        self.metrics.episode_loss(loss)

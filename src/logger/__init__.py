from .metrics import Metrics
from .writer import Writer


class Logger:

    def __init__(self):
        self.writer = Writer()
        self.metrics = Metrics()
        self.reset()

    def reset(self):
        self.episode_data = {
            'reward':[],
            'critic_loss':[],
            'actor_loss':[],
            'noise':[],
            'position':[],
            'velocity':[],
        }

    def add_information(self, reward, critic_loss, actor_loss, noise, position, velocity):
        self.episode_data['reward'].append(reward)
        self.episode_data['critic_loss'].append(critic_loss)
        self.episode_data['actor_loss'].append(actor_loss)
        self.episode_data['noise'].append(sum(noise) / len(noise))
        self.episode_data['position'].append(position)
        self.episode_data['velocity'].append(velocity)
    
    def save(self, episode, real_episode_duration, virtual_episode_duration):
        self.writer.write_scalar('Episode Duration (Virtual)', virtual_episode_duration, episode)
        self.writer.write_scalar('Episode Duration (Real)', real_episode_duration, episode)

        total_reward = self.metrics.total_reward(self.episode_data)
        self.writer.write_scalar('Total Reward', total_reward, episode)

        best_reward = self.metrics.best_reward(total_reward)
        self.writer.write_scalar('Best Reward', best_reward, episode)

        critic_loss, actor_loss = self.metrics.loss(self.episode_data)
        self.writer.write_scalar('Total Critic Loss', critic_loss, episode)
        self.writer.write_scalar('Total Actor Loss', actor_loss, episode)
        self.writer.write_scalar('Total Loss', critic_loss + actor_loss, episode)

        noise = self.metrics.noise(self.episode_data)
        self.writer.write_scalar('Mean Noise', noise, episode)

        mean_position = self.metrics.mean_position(self.episode_data)
        self.writer.write_scalar('Mean Position', mean_position, episode)
        mode_position = self.metrics.mode_position(self.episode_data)
        self.writer.write_scalar('Mode Position', mode_position, episode)

        mean_velocity = self.metrics.mean_velocity(self.episode_data)
        self.writer.write_scalar('Mean Velocity', mean_velocity, episode)
        mode_velocity = self.metrics.mode_velocity(self.episode_data)
        self.writer.write_scalar('Mode Velocity', mode_velocity, episode)

import numpy as np
from .metrics import Metrics
from .writer import Writer
from .hyperparameters import Hyperparameters

class Logger:

    def __init__(self, env, agent):
        self.writer = Writer()
        self.metrics = Metrics()
        self.hyperparameters = Hyperparameters(env, agent)
        self.reset()
        self.save_hyperparameters()
    
    def save_hyperparameters(self):
        self.writer.write_hyperparameters(
            self.hyperparameters.get_all()
        )

    def reset(self):
        self.episode_data = {
            'reward':np.array([]),
            'critic_loss':np.array([]),
            'critic_gradient':np.array([]),
            'predict_q':np.array([]),
            'target_q':np.array([]),
            'actor_loss':np.array([]),
            'actor_gradient':np.array([]),
            'noise':np.array([]),
            'position':np.array([]),
            'velocity':np.array([]),
        }
        for key in self.episode_data:
            self.episode_data[key] = np.array(self.episode_data[key], dtype=float)

    def add_information(self, reward, critic_loss, critic_gradient, predict_q, target_q, actor_loss, actor_gradient, noise, position, velocity):
        self.episode_data['reward'] = np.append(self.episode_data['reward'], reward)
        self.episode_data['critic_loss'] = np.append(self.episode_data['critic_loss'], critic_loss)
        self.episode_data['critic_gradient'] = np.append(self.episode_data['critic_gradient'], critic_gradient)
        self.episode_data['predict_q'] = np.append(self.episode_data['predict_q'], predict_q)
        self.episode_data['target_q'] = np.append(self.episode_data['target_q'], target_q)
        self.episode_data['actor_loss'] = np.append(self.episode_data['actor_loss'], actor_loss)
        self.episode_data['actor_gradient'] = np.append(self.episode_data['actor_gradient'], actor_gradient)
        self.episode_data['noise'] = np.append(self.episode_data['noise'], noise)
        self.episode_data['position'] = np.append(self.episode_data['position'], position)
        self.episode_data['velocity'] = np.append(self.episode_data['velocity'], velocity)
    
    def save(self, episode, real_episode_duration, virtual_episode_duration):
        self.writer.write_scalar('Episode/Duration (Virtual)', virtual_episode_duration, episode)
        self.writer.write_scalar('Episode/Duration (Real)', real_episode_duration, episode)

        mean_reward = self.metrics.mean_reward(self.episode_data)
        best_mean_reward = self.metrics.best_mean_reward(mean_reward)
        self.writer.write_scalar('Reward/Mean', mean_reward, episode)
        self.writer.write_scalar('Reward/Best Mean', best_mean_reward, episode)

        mean_critic_loss = self.metrics.mean_critic_loss(self.episode_data)
        mean_critic_gradient = self.metrics.mean_critic_gradient(self.episode_data)
        mean_predict_q = self.metrics.mean_predict_q(self.episode_data)
        mean_target_q = self.metrics.mean_target_q(self.episode_data)
        self.writer.write_scalar('Critic/Mean Loss', mean_critic_loss, episode)
        self.writer.write_scalar('Critic/Mean Gradient', mean_critic_gradient, episode)
        self.writer.write_scalar('Critic/Mean Predict Q-value', mean_predict_q, episode)
        self.writer.write_scalar('Critic/Mean Target Q-value', mean_target_q, episode)

        mean_actor_loss = self.metrics.mean_actor_loss(self.episode_data)
        mean_actor_gradient = self.metrics.mean_actor_gradient(self.episode_data)
        self.writer.write_scalar('Actor/Mean Loss', mean_actor_loss, episode)
        self.writer.write_scalar('Actor/Mean Gradient', mean_actor_gradient, episode)

        noise = self.metrics.noise(self.episode_data)
        self.writer.write_scalar('Noise/Mean', noise, episode)

        mean_position = self.metrics.mean_position(self.episode_data)
        mode_position = self.metrics.mode_position(self.episode_data)
        self.writer.write_scalar('Position/Mean', mean_position, episode)
        self.writer.write_scalar('Position/Mode', mode_position, episode)

        mean_velocity = self.metrics.mean_velocity(self.episode_data)
        mode_velocity = self.metrics.mode_velocity(self.episode_data)
        self.writer.write_scalar('Velocity/Mean', mean_velocity, episode)
        self.writer.write_scalar('Velocity/Mode', mode_velocity, episode)

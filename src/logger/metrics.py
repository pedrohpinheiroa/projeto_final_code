import numpy as np

class Metrics:

    def __init__(self):
        self.training_best_reward = -float('inf')

    def mean_reward(self, episode_data):
        episode_rewards = episode_data['reward']
        return np.mean(episode_rewards)
    
    def noise(self, episode_data):
        noise = episode_data['noise']
        return  np.mean(noise)

    def best_mean_reward(self, mean_reward):
        if mean_reward > self.training_best_reward:
            self.training_best_reward = mean_reward
        return self.training_best_reward
    
    def mean_position(self, episode_data):
        position = episode_data['position']
        return  np.mean(position)
    
    def mode_position(self, episode_data):
        position = episode_data['position']
        position = position.tolist()
        return max(set(position), key=position.count)
    
    def mean_velocity(self, episode_data):
        velocity = episode_data['velocity']
        return np.mean(velocity)
    
    def mode_velocity(self, episode_data):
        velocity = episode_data['velocity']
        velocity = velocity.tolist()
        return max(set(velocity), key=velocity.count)

    def mean_critic_loss(self, episode_data):
        critic_loss = episode_data['critic_loss']
        critic_loss = np.array(critic_loss)
        return np.mean(critic_loss.flatten())
    
    def mean_critic_gradient(self, episode_data):
        critic_gradient = episode_data['critic_gradient']
        return np.mean(critic_gradient)
    
    def mean_predict_q(self, episode_data):
        q_values = episode_data['predict_q']
        return np.mean(q_values)
    
    def mean_target_q(self, episode_data):
        target_q = episode_data['target_q']
        target_q = np.array(target_q)
        return np.mean(target_q.flatten())

    def mean_actor_loss(self, episode_data):
        actor_loss = episode_data['actor_loss']
        return np.mean(actor_loss.flatten())

    def mean_actor_gradient(self, episode_data):
        actor_gradient = episode_data['actor_gradient']
        return np.mean(actor_gradient)
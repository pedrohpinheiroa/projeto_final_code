class Metrics:

    def __init__(self):
        self.training_best_reward = -float('inf')

    def mean_reward(self, episode_data):
        episode_rewards = episode_data['reward']
        return sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0

    def loss(self, episode_data):
        critic_loss = episode_data['critic_loss']
        actor_loss = episode_data['actor_loss']
        return sum(critic_loss), sum(actor_loss)
    
    def noise(self, episode_data):
        noise = episode_data['noise']
        return sum(noise) / len(noise) if noise else 0

    def best_mean_reward(self, mean_reward):
        if mean_reward > self.training_best_reward:
            self.training_best_reward = mean_reward
        return self.training_best_reward
    
    def mean_position(self, episode_data):
        position = episode_data['position']
        return sum(position) / len(position)
    
    def mode_position(self, episode_data):
        position = episode_data['position']
        return max(set(position), key=position.count)
    
    def mean_velocity(self, episode_data):
        velocity = episode_data['velocity']
        return sum(velocity) / len(velocity)
    
    def mode_velocity(self, episode_data):
        velocity = episode_data['velocity']
        return max(set(velocity), key=velocity.count)
    

from src.agent.DDPG import Agent
from src.environment.virtual_enviroment import Seesaw

class Hyperparameters:

    def __init__(self, env:Seesaw, agent:Agent):
        self.env = env
        self.agent = agent
    
    def get_hyperparameters(self):
        buffer_hyperparameters = {
            'buffer_size': self.agent.buffer.buffer_size,
            'buffer_batch_size': self.agent.buffer.batch_size,
        }
        return buffer_hyperparameters
    
    def get_model_hyperparameters(self):
        model_hyperparameters = {
            'model_actor_learning_rate': self.agent.actor.learning_rate,
            'model_critic_learning_rate': self.agent.critic.learning_rate,
            'model_tau': self.agent.actor.tau,
            'model_gamma': self.agent.gamma,
        }
        return model_hyperparameters
    
    def get_noise_hyperparameters(self):
        noise_hyperparameters = {
            'noise_mu': self.agent.noise.mu,
            'noise_theta': self.agent.noise.theta,
            'noise_sigma': self.agent.noise.sigma,
        }
        return noise_hyperparameters
    
    def get_physics_hyperparameters(self):
        physics_hyperparameters = {
            'physics_gravity': self.env.physics.gravity,
            'physics_time_step': self.env.physics.time_step,
            'physics_bar_mass': self.env.physics.bar_mass,
            'physics_bar_length': self.env.physics.bar_length,
            'physics_motor_mass': self.env.physics.motor_mass,
        }
        return physics_hyperparameters
    
    def get_reward_hyperparameters(self):
        reward_hyperparameters = {
            'reward_position_weight': self.env.reward.position_weight,
            'reward_velocity_weight': self.env.reward.velocity_weight,
            'reward_torque_weight': self.env.reward.torque_weight,
            'reward_knock_penality': self.env.reward.knock_penalty,
            'reward_stability_reward': self.env.reward.stability_reward,
            'reward_stability_threshold': self.env.reward.stability_threshold,
        }
        return reward_hyperparameters
    
    def get_state_hyperparameters(self):
        state_hyperparameters = {
            'state_max_angle': self.env.state.max_angle,
            'state_pwm_max': self.env.state.pwm_max,
            'state_pwm_min': self.env.state.pwm_min,
            'state_max_episode_time': self.env.state.max_episode_time,
        }
        return state_hyperparameters
    
    def get_all(self):
        hyperparameters = {}
        hyperparameters.update(self.get_hyperparameters())
        hyperparameters.update(self.get_model_hyperparameters())
        hyperparameters.update(self.get_noise_hyperparameters())
        hyperparameters.update(self.get_physics_hyperparameters())
        hyperparameters.update(self.get_reward_hyperparameters())
        hyperparameters.update(self.get_state_hyperparameters())
        return hyperparameters


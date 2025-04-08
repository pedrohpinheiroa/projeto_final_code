import numpy as np
from .buffer import ReplayBuffer
from .noise import OUNoise
from .model import Actor, Critic

class Agent:

    def __init__(self):
        self.buffer = ReplayBuffer()
        self.actor = Actor()
        self.critic = Critic()
        self.noise = OUNoise(action_dimension=self.critic.action_dimension)
        self.gamma = 0.99 

    def reset(self):
        self.noise.reset()

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def sample_experiences(self):
        return self.buffer.sample()
    
    def get_all_experience(self):
        return self.buffer.get_all()

    def act(self, state, add_noise=True):
        state = (state.get('position'), state.get('velocity'))
        action = self.actor.predict(state)
        if add_noise:
            noise = self.noise.sample()
            action = np.clip(action + noise, -self.actor.action_bound, self.actor.action_bound)
        return action

    def learn(self):
        if len(self.buffer) < self.buffer.batch_size:
            return
        
        experiences = self.sample_experiences()
        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])

        target_actions = np.array([self.actor.target_predict(next_state) for next_state in next_states])
        target_q_values = np.array([self.critic.target_predict(next_states[i], target_actions[i])[0] for i in range(len(experiences))])
        target_q = rewards + (1 - dones) * self.gamma * target_q_values

        critic_loss = self.critic.train(states, actions, target_q)
        actor_loss = self.actor.train(states, self.critic.model)

        self.actor.update_target()
        self.critic.update_target()
        return critic_loss, actor_loss

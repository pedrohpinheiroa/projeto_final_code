import time
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
        noise = (0,0)
        if add_noise:
            noise = self.noise.sample()
            action = np.clip(action + noise, -self.actor.action_bound, self.actor.action_bound)
        return action, noise

    def learn(self):
        batch_size = self.buffer.batch_size
        if len(self.buffer) < batch_size:
            return (0, 0)
        
        experiences = self.sample_experiences()
        states = np.zeros((batch_size, self.actor.input_dimension), dtype=np.float32)
        actions = np.zeros((batch_size, self.actor.action_dim), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        next_states = np.zeros((batch_size, self.actor.input_dimension), dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)
        
        # Preencher arrays sem loops de compreensão
        for i, exp in enumerate(experiences):
            states[i] = exp[0]
            actions[i] = exp[1]
            rewards[i] = exp[2]
            next_states[i] = exp[3]
            dones[i] = exp[4]

        target_actions = self.actor.target_model.predict(next_states, batch_size=batch_size, verbose=0)
        target_q_values = self.critic.target_model.predict([next_states, target_actions], batch_size=batch_size, verbose=0).flatten()
        target_q = rewards + (1 - dones) * self.gamma * target_q_values

        critic_loss = self.critic.train(states, actions, target_q)
        actor_loss = self.actor.train(states, self.critic.model)

        self.actor.update_target()
        self.critic.update_target()
        return critic_loss, actor_loss

    def save(self):
        base_filename = f"models/{int(time.time())}"
        self.actor.save(base_filename)
        self.critic.save(base_filename)
    
    def load(self, filename):
        self.actor.load(filename)
        self.critic.load(filename)
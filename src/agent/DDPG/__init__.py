import time
import keras
import numpy as np
import tensorflow as tf
from .buffer import ReplayBuffer
from .noise import OUNoise
from .model import Actor, Critic


class Agent:

    def __init__(self):
        self.buffer = ReplayBuffer()
        self.actor = Actor()
        self.critic = Critic()
        self.noise = OUNoise(action_dimension=self.critic.action_dimension)
        self.gamma = 0.95

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

    @tf.function
    def update_models(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            target_actions = self.actor.target_model(next_states, training=True)
            y = rewards + self.gamma * self.critic.target_model(
                [next_states, target_actions], training=True
            )
            critic_value = self.critic.model([states, actions], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(critic_grad, self.critic.model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor.model(states, training=True)
            critic_value = self.critic.model([states, actions], training=True)
            actor_loss = -keras.ops.mean(critic_value)
        
        actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_grad, self.actor.model.trainable_variables)
        )
        
        return critic_loss, critic_grad, critic_value, y, actor_loss, actor_grad

    def learn(self):
        batch_size = self.buffer.batch_size
        if len(self.buffer) < batch_size:
            return (0, 0, 0, 0, 0, 0)
        
        experiences = self.sample_experiences()
        states = np.zeros((batch_size, self.actor.input_dimension), dtype=np.float32)
        actions = np.zeros((batch_size, self.actor.action_dim), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        next_states = np.zeros((batch_size, self.actor.input_dimension), dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)
        
        for i, exp in enumerate(experiences):
            states[i] = exp[0]
            actions[i] = exp[1]
            rewards[i] = exp[2]
            next_states[i] = exp[3]
            dones[i] = exp[4]

        states = keras.ops.convert_to_tensor(states)
        actions = keras.ops.convert_to_tensor(actions)
        rewards = keras.ops.convert_to_tensor(rewards)
        next_states = keras.ops.convert_to_tensor(next_states)
        dones = keras.ops.convert_to_tensor(dones)

        critic_loss, critic_gradient, predict_q, target_q_values, actor_loss, actor_gradient = self.update_models(states, actions, rewards, next_states, dones)

        critic_gradient = tf.linalg.global_norm(critic_gradient).numpy()
        actor_gradient = tf.linalg.global_norm(actor_gradient).numpy()
        predict_q = tf.reduce_mean(predict_q).numpy()
        target_q_values = tf.reduce_mean(target_q_values).numpy()

        self.actor.update_target()
        self.critic.update_target()
        return critic_loss, critic_gradient, predict_q, target_q_values, actor_loss, actor_gradient

    def save(self):
        base_filename = f"models/{int(time.time())}"
        self.actor.save(base_filename)
        # self.critic.save(base_filename)
    
    def load(self, filename):
        self.actor.load(filename)
        self.critic.load(filename)
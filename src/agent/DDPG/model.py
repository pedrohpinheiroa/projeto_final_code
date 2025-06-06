import json
import random

import keras
import tensorflow as tf
import numpy as np
from keras import layers, optimizers

class Actor:
    """Rede neural do ator que mapeia estados para ações."""
    
    def __init__(self):
        self.read_configs()
        self.input_dimension = self.configs['input_dimension']
        self.action_dim = self.configs['action_dimension']
        self.action_bound = self.configs['action_bound']
        self.hidden_layers = self.configs['actor_hidden_layers']
        self.tau = self.configs['polyak_averaging_tau']

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = optimizers.Adam(self.configs['actor_learning_rate'])

    def read_configs(self):
        with open('configs/model.json', 'r') as file:
            self.configs = json.load(file)

    def _build_model(self):
        inputs = layers.Input(shape=(self.input_dimension,))
        x = inputs
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)

        # Using sigmoid activation to get outputs between 0 and 1, then scaling
        outputs = layers.Dense(self.action_dim, activation='sigmoid')(x)
        outputs = outputs * self.action_bound + 0.1
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def predict(self, state):
        """Prediz a ação para um estado."""
        if not isinstance(state, np.ndarray) or state.ndim == 1:
            state_array = np.array([state], dtype=np.float32)
        else:
            state_array = state.astype(np.float32)
        return self.model.predict(state_array, verbose=0)[0]

    def target_predict(self, state):
        """Prediz a ação usando o modelo alvo."""
        if not isinstance(state, np.ndarray) or state.ndim == 1:
            state_array = np.array([state], dtype=np.float32)
        else:
            state_array = state.astype(np.float32)
        return self.target_model.predict(state_array, verbose=0)[0]
    
    def update_target(self):
        """Atualiza os pesos do modelo alvo usando a Média de Polyak."""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        
        self.target_model.set_weights(target_weights)
    
    def save(self, base_filename):
        """Salva os pesos do modelo."""
        filename = f"{base_filename}_actor.weights.h5"
        self.model.save_weights(filename)
    
    def load(self, filename):
        """Carrega os pesos do modelo."""
        self.model.load_weights(filename)
        


class Critic:
    """Rede neural do crítico que estima o Q-valor dado um estado e uma ação."""
    
    def __init__(self):
        self.read_configs()
        self.input_dimension = self.configs['input_dimension']
        self.action_dimension = self.configs['action_dimension']
        self.state_hidden_layers = self.configs['critic_state_hidden_layers']
        self.action_hidden_layers = self.configs['critic_action_hidden_layers']
        self.hidden_layers = self.configs['critic_hidden_layers']
        self.tau = self.configs['polyak_averaging_tau']
        self.optimizer = optimizers.Adam(self.configs['critic_learning_rate'])

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def read_configs(self):
        with open('configs/model.json', 'r') as file:
            self.configs = json.load(file)

    def _build_model(self):
        # Entrada de estado
        state_input = layers.Input(shape=(self.input_dimension,))  # position, velocity
        state_out = state_input
        for units in self.state_hidden_layers:
            state_out = layers.Dense(units, activation='relu')(state_out)

        action_input = layers.Input(shape=(self.action_dimension,))  # left_pwm, right_pwm
        action_out = action_input
        for units in self.action_hidden_layers:
            action_out = layers.Dense(units, activation='relu')(action_out)
        
        # Combina as duas entradas
        concat = layers.Concatenate()([state_out, action_out])
        x = concat
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
        
        outputs = layers.Dense(1)(x)  # Q-valor
        model = keras.Model([state_input, action_input], outputs)
        return model
    
    def predict(self, state, action):
        """Prediz o Q-valor para um par estado-ação."""
        if not isinstance(state, np.ndarray) or state.ndim == 1:
            state_array = np.array([state], dtype=np.float32)
        else:
            state_array = state.astype(np.float32)
            
        if not isinstance(action, np.ndarray) or action.ndim == 1:
            action_array = np.array([action], dtype=np.float32)
        else:
            action_array = action.astype(np.float32)
            
        return self.model.predict([state_array, action_array], verbose=0)[0]

    def target_predict(self, state, action):
        """Prediz o Q-valor usando o modelo alvo."""
        if not isinstance(state, np.ndarray) or state.ndim == 1:
            state_array = np.array([state], dtype=np.float32)
        else:
            state_array = state.astype(np.float32)
            
        if not isinstance(action, np.ndarray) or action.ndim == 1:
            action_array = np.array([action], dtype=np.float32)
        else:
            action_array = action.astype(np.float32)
            
        return self.target_model.predict([state_array, action_array], verbose=0)[0]
    
    def update_target(self):
        """Atualiza os pesos do modelo alvo usando soft update."""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        
        self.target_model.set_weights(target_weights)
    
    @tf.function
    def train(self, states, actions, target_q_values):
        """Treina o crítico usando mini-batch."""
        with tf.GradientTape() as tape:
            q_values = self.model([states, actions], training=True)  # Q-values atuais
            loss = tf.keras.losses.MSE(target_q_values, q_values)    # Cálculo da perda
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=0.5)  # Norma máxima = 1.0
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        gradient_norm = tf.linalg.global_norm(gradients).numpy()
        return loss.numpy(), np.mean(q_values), gradient_norm

    def save(self, base_filename):
        """Salva os pesos do modelo."""
        filename = f"{base_filename}_critic.weights.h5"
        self.model.save_weights(filename)
    
    def load(self, filename):
        """Carrega os pesos do modelo."""
        self.model.load_weights(filename)

import tensorflow as tf
import numpy as np
from keras import layers, models

class Actor:
    """Rede neural do ator que mapeia estados para ações."""
    
    def __init__(self, action_dim=2, action_bound=0.35):
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.model = self._build_model()
        self.target_model = self._build_model()
        # Garantir que os pesos do target model são iguais inicialmente
        self.target_model.set_weights(self.model.get_weights())
        
    def _build_model(self):
        inputs = layers.Input(shape=(2,))  # position, velocity
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='tanh')(x)
        # Saídas escalonadas pelo limite de ação [-action_bound, action_bound]
        outputs = outputs * self.action_bound
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def predict(self, state):
        """Prediz a ação para um estado."""
        state_array = np.array([state])
        return self.model.predict(state_array)[0]
    
    def target_predict(self, state):
        """Prediz a ação usando o modelo alvo."""
        state_array = np.array([state])
        return self.target_model.predict(state_array)[0]
    
    def update_target(self, tau=0.001):
        """Atualiza os pesos do modelo alvo usando soft update."""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        for i in range(len(weights)):
            target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
        
        self.target_model.set_weights(target_weights)


class Critic:
    """Rede neural do crítico que estima o Q-valor dado um estado e uma ação."""
    
    def __init__(self):
        self.model = self._build_model()
        self.target_model = self._build_model()
        # Garantir que os pesos do target model são iguais inicialmente
        self.target_model.set_weights(self.model.get_weights())
        
    def _build_model(self):
        # Entrada de estado
        state_input = layers.Input(shape=(2,))  # position, velocity
        state_out = layers.Dense(16, activation='relu')(state_input)
        state_out = layers.Dense(32, activation='relu')(state_out)
        
        # Entrada de ação
        action_input = layers.Input(shape=(2,))  # left_pwm, right_pwm
        action_out = layers.Dense(32, activation='relu')(action_input)
        
        # Combina as duas entradas
        concat = layers.Concatenate()([state_out, action_out])
        
        x = layers.Dense(256, activation='relu')(concat)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(1)(x)  # Q-valor
        
        model = models.Model([state_input, action_input], outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        return model
    
    def predict(self, state, action):
        """Prediz o Q-valor para um par estado-ação."""
        state_array = np.array([state])
        action_array = np.array([action])
        return self.model.predict([state_array, action_array])[0]
    
    def target_predict(self, state, action):
        """Prediz o Q-valor usando o modelo alvo."""
        state_array = np.array([state])
        action_array = np.array([action])
        return self.target_model.predict([state_array, action_array])[0]
    
    def update_target(self, tau=0.001):
        """Atualiza os pesos do modelo alvo usando soft update."""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        for i in range(len(weights)):
            target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
        
        self.target_model.set_weights(target_weights)
    
    def train(self, states, actions, target_q_values):
        """Treina o crítico usando mini-batch."""
        return self.model.train_on_batch([states, actions], target_q_values)


class OUNoise:
    """Implementação do ruído Ornstein-Uhlenbeck para exploração."""
    
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class DDPGModel:
    """Modelo DDPG completo."""
    
    def __init__(self, action_dim=2, action_bound=0.35):
        self.actor = Actor(action_dim, action_bound)
        self.critic = Critic()
        self.noise = OUNoise(action_dim)
        self.gamma = 0.99  # Fator de desconto
        self.action_bound = action_bound
        
    def act(self, state, add_noise=True):
        """Escolhe uma ação com base no estado atual."""
        action = self.actor.predict(state)
        if add_noise:
            noise = self.noise.sample()
            action = np.clip(action + noise, -self.action_bound, self.action_bound)
        return action
    
    def learn(self, experiences):
        """Aprende a partir de um lote de experiências."""
        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])
        
        # Treina o crítico
        target_actions = np.array([self.actor.target_predict(next_state) for next_state in next_states])
        target_q_values = np.array([self.critic.target_predict(next_states[i], target_actions[i])[0] 
                                   for i in range(len(experiences))])
        
        target_q = rewards + (1 - dones) * self.gamma * target_q_values
        
        # Atualiza o crítico
        self.critic.train(states, actions, target_q)
        
        # Atualiza o ator usando o gradiente do crítico
        with tf.GradientTape() as tape:
            predicted_actions = self.actor.model(states)
            critic_values = self.critic.model([states, predicted_actions])
            actor_loss = -tf.math.reduce_mean(critic_values)
        
        actor_gradients = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        tf.keras.optimizers.Adam(0.0001).apply_gradients(
            zip(actor_gradients, self.actor.model.trainable_variables)
        )
        
        # Soft update dos modelos alvo
        self.actor.update_target()
        self.critic.update_target()
import time
import keras
import numpy as np
import tensorflow as tf
from .buffer import ReplayBuffer
from .noise import OUNoise
from .model import Actor, Critic


class Agent:
    """
    Implementa o agente DDPG (Deep Deterministic Policy Gradient).

    Responsabilidades:
        - Gerenciar o buffer de experiências.
        - Gerenciar as redes do ator e do crítico.
        - Gerar ações (com ou sem ruído de exploração).
        - Aprender a partir de experiências (atualizar as redes neurais).
        - Salvar e carregar modelos treinados.
    """

    def __init__(self):
        """
        Inicializa o agente DDPG, criando buffer, ator, crítico e ruído.
        """
        self.buffer = ReplayBuffer()
        self.actor = Actor()
        self.critic = Critic()
        self.noise = OUNoise(action_dimension=self.critic.action_dimension)
        self.gamma = self.actor.gamma
        
        

    def reset(self):
        """
        Reseta o ruído do agente.
        """
        self.noise.reset()

    def add_experience(self, state: dict, action: np.ndarray, reward: float, next_state: dict, done: bool) -> None:
        """
        Adiciona uma experiência ao buffer de replay.

        Parâmetros
        ----------
        state : dict
            Estado atual do ambiente.
        action : array_like
            Ação tomada pelo agente.
        reward : float
            Recompensa recebida.
        next_state : dict
            Próximo estado do ambiente.
        done : bool
            Indica se o episódio terminou.
        """
        self.buffer.add(state, action, reward, next_state, done)

    def sample_experiences(self) -> list:
        """
        Amostra um mini-lote de experiências do buffer de replay.

        Retorna
        -------
        list
            Lista de experiências amostradas.
        """
        return self.buffer.sample()
    
    def get_all_experience(self) -> list:
        """
        Retorna todas as experiências do buffer de replay.

        Retorna
        -------
        list
            Lista de todas as experiências armazenadas.
        """
        return self.buffer.get_all()
    
    def decay_noise(self) -> None:
        """
        Decai o fator de ruído do agente.
        """
        self.noise.decay()

    def act(self, state: dict, add_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Gera uma ação para o estado atual, com ou sem ruído de exploração.

        Parâmetros
        ----------
        state : dict
            Estado atual do ambiente.
        add_noise : bool, opcional
            Se True, adiciona ruído de exploração à ação (padrão: True).

        Retorna
        -------
        action : np.ndarray
            Ação sugerida pelo ator (com ou sem ruído).
        noise : np.ndarray
            Ruído aplicado à ação.
        """
        state = (state.get('position'), state.get('velocity'))
        action = self.actor.predict(state)
        noise = np.zeros(self.critic.action_dimension, dtype=np.float32)
        if add_noise:
            noise = self.noise.sample()
            action = action + noise
        action = np.clip(action, -1.0, 1.0)
        action = np.round(action, 4)
        return action, noise

    @tf.function
    def update_models(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor
    ) -> tuple[tf.Tensor, list, tf.Tensor, tf.Tensor, tf.Tensor, list]:
        """
        Atualiza as redes do crítico e do ator usando um mini-lote de experiências.

        Parâmetros
        ----------
        states : tf.Tensor
            Estados do mini-lote.
        actions : tf.Tensor
            Ações do mini-lote.
        rewards : tf.Tensor
            Recompensas do mini-lote.
        next_states : tf.Tensor
            Próximos estados do mini-lote.
        dones : tf.Tensor
            Flags de término de episódio do mini-lote.

        Retorna
        -------
        critic_loss : tf.Tensor
            Perda do crítico.
        critic_grad : list
            Gradientes do crítico.
        critic_value : tf.Tensor
            Q-values atuais do crítico.
        y : tf.Tensor
            Q-targets calculados.
        actor_loss : tf.Tensor
            Perda do ator.
        actor_grad : list
            Gradientes do ator.
        """
        with tf.GradientTape() as tape:
            target_actions = self.actor.target_model(next_states, training=True)
            y = rewards + self.gamma * self.critic.target_model(
                [next_states, target_actions], training=True
            )
            critic_value = self.critic.model([states, actions], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        if self.critic.clip_gradients:
            critic_grad, _ = tf.clip_by_global_norm(critic_grad, clip_norm=0.5)  # TESTE Norma máxima = 1.0

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

    def learn(self) -> tuple[float, float, float, float, float, float]:
        """
        Executa um passo de aprendizado, atualizando as redes se houver experiências suficientes.

        Retorna
        -------
        critic_loss : float
            Perda média do crítico.
        critic_gradient : float
            Norma global dos gradientes do crítico.
        predict_q : float
            Q-value médio predito pelo crítico.
        target_q_values : float
            Q-value médio alvo.
        actor_loss : float
            Perda média do ator.
        actor_gradient : float
            Norma global dos gradientes do ator.
        """
        batch_size = self.buffer.batch_size
        if len(self.buffer) < batch_size:
            return (0, 0, 0, 0, 0, 0)
        
        experiences = self.sample_experiences()
        states = np.zeros((batch_size, self.actor.input_dimension), dtype=np.float32)
        actions = np.zeros((batch_size, self.actor.action_dim), dtype=np.float32)
        rewards = np.zeros((batch_size, 1), dtype=np.float32)
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

    def save(self, filename: str = None) -> None:
        """
        Salva os pesos do ator (e opcionalmente do crítico) em arquivos na pasta models/.

        Parâmetros
        ----------
        filename : str, opcional
            Nome base do arquivo para salvar os pesos. Se não fornecido, usa timestamp.
        """
        if filename:
            base_filename = f"models/{filename}"
        else:
            base_filename = f"models/{int(time.time())}"
        self.actor.save(base_filename)
        self.critic.save(base_filename)
    
    def load(self, filename: str) -> None:
        """
        Carrega os pesos do ator e do crítico a partir de arquivos.

        Parâmetros
        ----------
        filename : str
            Nome base do arquivo dos pesos a serem carregados.
        """
        self.actor.load(filename)
        self.critic.load(filename)
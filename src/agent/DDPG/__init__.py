import numpy as np
from .buffer import ReplayBuffer

class Agent:

    def __init__(self):
        self.buffer = ReplayBuffer()

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def sample_experience(self):
        return self.buffer.sample()
    
    def get_all_experience(self):
        return self.buffer.get_all()

    def act(self, state):
        return np.random.uniform(-0.35, 0.35, size=(2,))

    def learn(self):
        expperience_batch = self.sample_experience()
        # Implementar a lógica de aprendizado aqui
        pass

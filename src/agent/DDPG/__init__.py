from .buffer import ReplayBuffer

class Agent:

    def __init__(self):
        self.buffer = ReplayBuffer()

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def sample_experience(self):
        return self.buffer.sample()
    
    def act(self, state):
        # Implementar a lógica de seleção de ação aqui
        pass

    def learn(self):
        expperience_batch = self.sample_experience()
        # Implementar a lógica de aprendizado aqui
        pass

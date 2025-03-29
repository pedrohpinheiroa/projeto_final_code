import random

class ReplayBuffer:
    """Buffer de experiência para armazenar transições e amostrar mini-lotes."""
    
    def __init__(self, buffer_size=100000, batch_size=64):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0
    
    def __len__(self):
        """Retorna o número atual de experiências armazenadas."""
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size
        
    def sample(self):
        return random.sample(self.buffer, self.batch_size)
    
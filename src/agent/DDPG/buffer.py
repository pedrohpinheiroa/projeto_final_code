import json
import random
from typing import Dict, List, Tuple, Any

class ReplayBuffer:
    """Buffer de experiência para armazenar transições e amostrar mini-lotes."""
    
    def __init__(self):
        self.buffer = []
        self.position = 0
        self.read_configs()
        self.set_configs()
    
    def __len__(self):
        return len(self.buffer)
    
    def read_configs(self):
        with open('configs/buffer.json', 'r') as file:
            self.configs = json.load(file)

    def set_configs(self):
        self.buffer_size = self.configs['buffer_size']
        self.batch_size = self.configs['batch_size']
        
    def create_model_state(self, state:Dict) -> Tuple[float, float]:
        return (state.get('position'), state.get('velocity'))

    def add(self, state:Dict, action:Tuple[float, float], reward:float, next_state:Dict, done:bool)-> None:
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        
        state = self.create_model_state(state)
        next_state = self.create_model_state(next_state)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size
        
    def sample(self):
        return random.sample(self.buffer, self.batch_size)
    
    def get_all(self) -> List[Tuple[Dict, Tuple[float, float], float, Dict, bool]]:
        return self.buffer
import time
import numpy as np
from src.environment.virtual_enviroment import Seesaw
from src.agent.DDPG import Agent

def format_time(seconds):
    """Formata segundos em hh:mm:ss"""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

# Configuração inicial
env = Seesaw()
agent = Agent()
EPISODES = 1000  # Comece com um número menor para testes

# Métricas de acompanhamento
metrics = {
    'rewards': [],
    'durations': [],
    'best_reward': -np.inf,
    'avg_reward': 0,
    'avg_steps': 0,
    'start_time': time.time()
}

for episode in range(EPISODES):
    episode_start = time.time()
    total_reward = 0
    steps = 0
    
    # Reset environment
    env.reset()
    
    while not env.is_done():
        # Coleta de experiência
        state = env.get_state()
        action = agent.act(state)
        env.step(action)
        reward = env.get_reward()
        next_state = env.get_state()
        done = env.is_done()
        
        # Armazenamento e aprendizado
        agent.add_experience(state, action, reward, next_state, done)
        agent.learn()
        
        total_reward += reward
        steps += 1

    # Atualização de métricas
    metrics['rewards'].append(total_reward)
    metrics['durations'].append(time.time() - episode_start)
    
    # Atualiza melhores resultados
    if total_reward > metrics['best_reward']:
        metrics['best_reward'] = total_reward

import time
import numpy as np
from src.environment.virtual_enviroment import Seesaw
from src.agent.DDPG import Agent
from src.logger import Logger

env = Seesaw()
agent = Agent()
logger = Logger()

EPISODES = 1000
for episode in range(EPISODES):
    episode_start = time.time()
    env.reset()
    
    while not env.is_done():
        state = env.get_state()
        action = agent.act(state)
        env.step(action)
        reward = env.get_reward()
        next_state = env.get_state()
        done = env.is_done()
        critic_loss, actor_loss = agent.learn()
        
        env.save_in_history(state)
        agent.add_experience(state, action, reward, next_state, done)
        logger.add_information()

    state = env.get_state()
    env.save_in_history(state)

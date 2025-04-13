import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
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
        if episode == 1:
            print(state)
        action, noise = agent.act(state)
        env.step(action)
        reward = env.get_reward()
        next_state = env.get_state()
        done = env.is_done()
        critic_loss, actor_loss = agent.learn()
        
        env.save_in_history(state)
        agent.add_experience(state, action, reward, next_state, done)
        logger.add_information(
            reward=reward,
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            noise=noise,
            position = state['position'],
            velocity = state['velocity'],
        )

    state = env.get_state()
    env.save_in_history(state)
    logger.save(
        episode=episode, 
        real_episode_duration = time.time() - episode_start,
        virtual_episode_duration = next_state['time'],
    )
    logger.reset()
    if episode % 10 == 0:
        agent.save()
    print(f"Episode {episode} finished. Time: {time.time() - episode_start:.2f}s")

import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import tensorflow as tf

from src.environment.virtual_enviroment import Seesaw
from src.agent.DDPG import Agent
from src.logger import Logger


def main():
    RENDER_SIMULATION = True
    EPISODES = 50

    env = Seesaw(randomize_initial_state=False)
    agent = Agent()
    logger = Logger(env, agent)

    for episode in range(EPISODES):
        if episode % 2 == 0:
            env.state.randomize_initial_state = True
        else:
            env.state.randomize_initial_state = False
        
        episode_reward = 0.0
        episode_start = time.time()
        env.reset()
        while not env.is_done():
            state = env.get_state()
            action, noise = agent.act(state)
            env.step(action)
            reward = env.get_reward()
            episode_reward += reward
            next_state = env.get_state()
            done = env.is_done()
            critic_loss, critic_gradient, predict_q, target_q, actor_loss, actor_gradient = agent.learn()
            if RENDER_SIMULATION:
                env.render()
            
            env.save_in_history(state)
            agent.add_experience(state, action, reward, next_state, done)
            logger.add_information(
                reward=reward,
                critic_loss=critic_loss,
                critic_gradient=critic_gradient,
                predict_q=predict_q,
                target_q=target_q,
                actor_loss=actor_loss,
                actor_gradient=actor_gradient,
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

        if episode == 0:
            os.system("cls" if os.name == "nt" else "clear")

        print(f"Episode {episode} finished. Time: {time.time() - episode_start:.2f}s. Episode Reward: {episode_reward}", end=("\n"))
        agent.save(f"{logger.get_name()}/")

if __name__ == "__main__":
    with tf.device('/CPU:0'):
        main()
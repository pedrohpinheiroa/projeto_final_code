import time

import numpy as np
import tensorflow as tf

from src.environment.virtual_enviroment import Seesaw
from src.agent.DDPG import Agent
from src.logger import Logger

def _render_current_model(agent):
    env = Seesaw()
    env.reset()
    env.render()
    while not env.is_done():
        action, _ = agent.act(env.get_state(), add_noise=False)
        env.step(action)
        env.render()
    print(f"Final Position: {env.get_state()['position']:.4f}, Final Velocity: {env.get_state()['velocity']:.4f}, Final Time: {env.get_state()['time']:.4f}")
    env.close()

def main():
    RENDER_SIMULATION = False
    RENDER_EVERY = 100
    EPISODES = 50_000

    env = Seesaw()
    agent = Agent()
    logger = Logger()
    for episode in range(EPISODES):
        episode_start = time.time()
        env.reset()
        
        while not env.is_done():
            state = env.get_state()
            action, noise = agent.act(state)
            env.step(action)
            reward = env.get_reward()
            next_state = env.get_state()
            done = env.is_done()
            critic_loss, critic_gradient, predict_q, target_q, actor_loss, actor_gradient = agent.learn()
            
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
        agent.decay_noise()
        
        print(f"Episode {episode} finished. Time: {time.time() - episode_start:.2f}s. Total Virtual Time: {state['time']/1000:.4f}s Final Position: {state['position']:.4f}", end=("\n"))
        if RENDER_SIMULATION and episode % RENDER_EVERY == 0:
            print("Rendering current model...")
            _render_current_model(agent)
            print("Rendering completed.")

        agent.save("current_model_training")

if __name__ == "__main__":
    with tf.device('/CPU:0'):
        main()
    # main()
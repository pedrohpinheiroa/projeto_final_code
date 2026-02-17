import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf

from src.environment.virtual_enviroment import Seesaw
from src.agent.DDPG import Agent
from src.agent.PID import Agent as PIDAgent
from src.logger import Logger

def run_test(agent:Agent):
    ddpg_env = Seesaw(randomize_initial_state=False)
    total_ddpg_reward = 0
    ddpg_env.reset()
    ddpg_env.state.position = 0.35

    while not ddpg_env.is_done():
        ddpg_state = ddpg_env.get_state()
        ddpg_action, _ = agent.act(ddpg_state, add_noise=False)
        ddpg_env.step(ddpg_action)
        total_ddpg_reward += ddpg_env.get_reward()
    
    ddpg_env.reset()
    ddpg_env.state.position = -0.35
    while not ddpg_env.is_done():
        ddpg_state = ddpg_env.get_state()
        ddpg_action, _ = agent.act(ddpg_state, add_noise=False)
        ddpg_env.step(ddpg_action)
        total_ddpg_reward += ddpg_env.get_reward()

    return total_ddpg_reward/2

def run_episode(env:Seesaw, agent:Agent, logger:Logger):
    episode_reward = 0.0
    episode_start = time.time()
    env.reset()
    while not env.is_done():
        state = env.get_state()
        action, noise = agent.act(state, add_noise=ADD_NOISE)
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
    real_episode_duration = time.time() - episode_start
    virtual_episode_duration = next_state['time']
    return env, agent, logger, real_episode_duration, virtual_episode_duration, episode_reward

def main():
    better_result = -np.inf
    env = Seesaw(randomize_initial_state=True)
    agent = Agent()
    logger = Logger(env, agent)
    for episode in range(EPISODES):        
        env, agent, logger, real_episode_duration, virtual_episode_duration, episode_reward = run_episode(env, agent, logger)
        logger.save(
            episode=episode, 
            real_episode_duration=real_episode_duration,
            virtual_episode_duration=virtual_episode_duration,
        )
        logger.reset()

        if episode == 0:
            os.system("cls" if os.name == "nt" else "clear")

        result = run_test(agent)
        print(f"Episode {episode} finished. Time: {real_episode_duration:.2f}s. Episode Reward: {episode_reward:.2f}. Model Reward {result:.2f}. Better Reward {better_result:.2f}", end=("\r"))
        if result > better_result and episode > 1:
            agent.save(f"{logger.get_name()}/")
            percentual_improvement = (result - better_result) / abs(better_result) * 100 if better_result != 0 else 0
            print(f"Episode {episode} finished. Time: {real_episode_duration:.2f}s. Episode Reward: {episode_reward:.2f}. Model Reward {result:.2f}. Better Reward {better_result:.2f}", end=("\n"))
            better_result = result
            print(f"New Better Reward: {better_result:.2f}, Percentual Improvement: {percentual_improvement:.2f}%", end=("\n"))
            print("="*100)
        

if __name__ == "__main__":
    RENDER_SIMULATION = False
    EPISODES = 10_000
    ADD_NOISE = True

    with tf.device('/CPU:0'):
        main()
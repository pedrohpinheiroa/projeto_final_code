# Preciso tratar o tempo em Miliseconds. Isso vai reduzir o erro matemático dos binários.
# Talvez isso corrija o bug da duração máxima do episódio ser 8 segundos, com tempos de execução tão divergentes.
import os
import json
import time
from src.environment.virtual_enviroment import Seesaw
from src.agent.DDPG import Agent
from src.logger import Logger

def create_checkpoint(agent:Agent):
    DIR = "./checkpoint"
    agent.actor.model.save_weights(f"{DIR}/actor_model.weights.h5")
    agent.actor.target_model.save_weights(f"{DIR}/actor_target.weights.h5")
    agent.critic.model.save_weights(f"{DIR}/critic_model.weights.h5")
    agent.critic.target_model.save_weights(f"{DIR}/critic_target.weights.h5")

def load_checkpoint(agent:Agent):
    DIR = "./checkpoint"
    try:
        agent.actor.model.load_weights(f"{DIR}/actor_model.weights.h5")
        agent.actor.target_model.load_weights(f"{DIR}/actor_target.weights.h5")
        agent.critic.model.load_weights(f"{DIR}/critic_model.weights.h5")
        agent.critic.target_model.load_weights(f"{DIR}/critic_target.weights.h5")
    except:
        pass
    finally:
        return agent


env = Seesaw()
agent = Agent()
logger = Logger()
agent = load_checkpoint(agent)

EPISODES = 100_000
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
    if episode % 50 == 0 and episode > 0:
        agent.save()
    
    create_checkpoint(agent)
    
    print(f"Episode {episode} finished. Time: {time.time() - episode_start:.2f}s. Total Virtual Time: {state['time']/1000:.4f}s Final Position: {state['position']:.4f}", end=("\n"))

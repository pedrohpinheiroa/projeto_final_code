import time
from src.environment.virtual_enviroment import Seesaw
from src.agent.DDPG import Agent
from src.logger import Logger


env = Seesaw()
agent = Agent()
logger = Logger()

EPISODES = 10_000
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
    if episode % 50 == 0 and episode > 0:
        agent.save()
        
    print(f"Episode {episode} finished. Time: {time.time() - episode_start:.2f}s. Total Virtual Time: {state['time']/1000:.4f}s Final Position: {state['position']:.4f}", end=("\n"))

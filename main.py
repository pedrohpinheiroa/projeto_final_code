from src.environment import Seesaw
from src.agent.DDPG import Agent

env = Seesaw()
agent = Agent()


while not env.is_done():
    state = env.get_state()
    env.save_in_history(state)

    action = (0.3, 0.0)
    env.step(action)
    
    reward = env.get_reward()
    new_state = env.get_state()
    done = env.is_done()
    agent.add_experience(state, action, reward, new_state, done)

state = new_state
env.save_in_history(state)
print(env.get_episode_history())

    # print(env.get_state())
    # print(env.get_reward())
    # print(env.get_done())
    # print(env.get_episode_history())
    # env.render()
    # env.close()

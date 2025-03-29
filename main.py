from src.environment import Seesaw

env = Seesaw()
while not env.is_done():
    action = (0.1, 0.1)
    env.step(action)
    print(env.get_state())
    print(env.get_reward())
    print(env.get_done())
    print(env.get_episode_history())
    env.render()
    env.close()

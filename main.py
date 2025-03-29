from src.environment.virtual_enviroment import Seesaw
from src.agent.DDPG import Agent

env = Seesaw()
agent = Agent()


while not env.is_done():
    state = env.get_state()
    env.save_in_history(state)

    action = agent.act(state)
    env.step(action)
    
    reward = env.get_reward()
    new_state = env.get_state()
    done = env.is_done()
    agent.add_experience(state, action, reward, new_state, done)
    print(new_state, reward, action)

state = new_state
env.save_in_history(state)


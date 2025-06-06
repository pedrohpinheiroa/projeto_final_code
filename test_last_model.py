import tensorflow as tf
import numpy as np
from src.environment.virtual_enviroment import Seesaw
from src.agent.DDPG import Agent

def main():
    env = Seesaw()
    agent = Agent()
    try:
        agent.actor.load("models/current_model_training_actor.weights.h5")
    except FileNotFoundError:
        print('Model file not found. Please train the model first.')

    env.reset()
    env.render()
    while not env.is_done():
        action, _ = agent.act(env.get_state(), add_noise=True)
        print(env.get_reward(), env.get_state())
        env.step(action)
        env.render()

    env.close() 
    print(env.get_state())


if __name__ == "__main__":
    with tf.device('/CPU:0'):
        main()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf

from src.environment.virtual_enviroment import Seesaw
from src.agent.DDPG import Agent

def _get_last_model_path():
    BASEDIR= 'models'
    if not os.path.exists(BASEDIR):
        raise FileNotFoundError(f'Directory {BASEDIR} does not exist.')

    trained_models = os.listdir(BASEDIR)
    if not trained_models:
        raise FileNotFoundError('No trained models found in the models directory.')
    last_trained_model = max(trained_models)
    return os.path.join(BASEDIR, last_trained_model, 'actor.weights.h5')


def main():
    env = Seesaw(randomize_initial_state=False)
    agent = Agent()
    agent.actor.load(_get_last_model_path())
    env.reset()
    env.render()
    while True:
        action, _ = agent.act(env.get_state(), add_noise=False)
        env.step(action)
        env.render()


if __name__ == "__main__":
    with tf.device('/CPU:0'):
        main()

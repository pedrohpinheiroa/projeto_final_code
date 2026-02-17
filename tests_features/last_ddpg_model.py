import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
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
    env.visualizer.include_setpoint_slider = True
    agent = Agent()
    agent.actor.load(_get_last_model_path())
    env.reset()
    env.render()
    while True:
        set_point = env.visualizer.setpoint_slider.get_value()
        transformed_state = env.get_state()
        transformed_state['position'] = transformed_state['position'] - set_point
        transformed_state['position'] = np.clip(transformed_state['position'], -0.35, 0.35)
        action, _ = agent.act(transformed_state, add_noise=False)
        env.step(action)
        env.render()


if __name__ == "__main__":
    with tf.device('/CPU:0'):
        main()

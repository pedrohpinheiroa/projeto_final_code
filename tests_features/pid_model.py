import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.virtual_enviroment import Seesaw
from src.agent.PID import Agent


def main():
    agent = Agent()
    env = Seesaw(randomize_initial_state=False)
    env.visualizer.include_setpoint_slider = True
    env.reset()
    env.render()
    while True:
        set_point = env.visualizer.setpoint_slider.get_value()
        agent.controller.setpoint = set_point
        action, _ = agent.act(env.get_state(), add_noise=False)
        env.step(action)
        env.render()


if __name__ == "__main__":
    main()
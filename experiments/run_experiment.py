import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json

import matplotlib.pyplot as plt

from src.environment.virtual_enviroment import Seesaw
from src.agent.DDPG import Agent as DDPGAgent
from src.agent.PID import Agent as PIDAgent

def _get_last_model_path():
    BASEDIR= 'models'
    if not os.path.exists(BASEDIR):
        raise FileNotFoundError(f'Directory {BASEDIR} does not exist.')

    trained_models = os.listdir(BASEDIR)
    if not trained_models:
        raise FileNotFoundError('No trained models found in the models directory.')
    last_trained_model = max(trained_models)
    return os.path.join(BASEDIR, last_trained_model, 'actor.weights.h5')

def _create_plot(title, pid_data, ddpg_data, time_steps):
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, pid_data, label='PID Controller', color='blue')
    plt.plot(time_steps, ddpg_data, label='DDPG Agent', color='orange')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Position')
    plt.legend()
    plt.grid()
    plt.savefig(f"{title.replace(' ', '_').lower()}_plot.png")
    plt.close()

def init_experiment():
    ddpg = DDPGAgent()
    ddpg.actor.load(_get_last_model_path())
    ddpg_env = Seesaw(randomize_initial_state=False)
    ddpg_env.state.position = 0.35

    pid = PIDAgent()
    pid_env = Seesaw(randomize_initial_state=False)
    pid_env.state.position = ddpg_env.state.position
    return ddpg, ddpg_env, pid, pid_env

def apply_scenario(ddpg:DDPGAgent, ddpg_env:Seesaw, pid:PIDAgent, pid_env:Seesaw, scenario:str):
    configs_path = f"experiments/configs/{scenario}.json"
    if not os.path.exists(configs_path):
        raise FileNotFoundError(f"Configuration file for scenario '{scenario}' not found at {configs_path}")
    
    with open(configs_path, 'r') as file:
        scenario_configs = json.load(file)

    for key, value in scenario_configs.items():
        if "env" in key:
            setattr(ddpg_env, key.split("env_")[1], value)
            setattr(pid_env, key.split("env_")[1], value)

    return ddpg, ddpg_env, pid, pid_env

def save_experiment_data(ddpg_data, pid_data, scenario):
    base_output_dir = f"experiments/results"

    with open(os.path.join(base_output_dir, f'ddpg_data_{scenario}.json'), 'w') as file:
        json.dump(ddpg_data, file, indent=4)

    with open(os.path.join(base_output_dir, f'pid_data_{scenario}.json'), 'w') as file:
        json.dump(pid_data, file, indent=4)

    print(f"Experiment data from {scenario} saved to {base_output_dir}")

def save_experiment_plots(ddpg_data, pid_data, scenario):
    time_steps = []
    ddpg_positions = []
    pid_positions = []
    for i in range(len(ddpg_data)):
        ddpg_iteration = ddpg_data[i]
        pid_iteration = pid_data[i]
        time_steps.append(ddpg_iteration['time'])
        ddpg_positions.append(ddpg_iteration['position'])
        pid_positions.append(pid_iteration['position'])
    
    _create_plot(f"Posição", pid_positions, ddpg_positions, time_steps)
    

def run_experiment(scenario:str):
    ddpg, ddpg_env, pid, pid_env = init_experiment()
    # ddpg, ddpg_env, pid, pid_env = apply_scenario(ddpg, ddpg_env, pid, pid_env, scenario)

    ddpg_data = []
    pid_data = []
    total_pid_reward = 0
    total_ddpg_reward = 0
    while not ddpg_env.is_done() and not pid_env.is_done():
        ddpg_state = ddpg_env.get_state()
        ddpg_action, _ = ddpg.act(ddpg_state, add_noise=False)
        ddpg_env.step(ddpg_action)
        
        pid_state = pid_env.get_state()
        pid_action, _ = pid.act(pid_state, add_noise=False)
        pid_env.step(pid_action)

        ddpg_reward = ddpg_env.get_reward()
        pid_reward = pid_env.get_reward()
        
        ddpg_iteration = ddpg_state
        ddpg_iteration['action'] = ddpg_action[0]
        ddpg_iteration['reward'] = ddpg_reward
        ddpg_data.append(ddpg_iteration)

        pid_iteration = pid_state
        pid_iteration['action'] = pid_action[0]
        pid_iteration['reward'] = pid_reward
        pid_data.append(pid_iteration)
        total_pid_reward += pid_reward
        total_ddpg_reward += ddpg_reward
    
    # save_experiment_data(ddpg_data, pid_data, scenario)
    save_experiment_plots(ddpg_data, pid_data, scenario)
    print(f"Total DDPG Reward: {total_ddpg_reward}")
    print(f"Total PID Reward: {total_pid_reward}")


if __name__ == "__main__":
    import tensorflow as tf
    with tf.device('/CPU:0'):
        run_experiment("")
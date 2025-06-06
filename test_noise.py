import numpy as np
from src.agent.DDPG.noise import OUNoise

if __name__ == "__main__":
    action_dimension = 2  # igual ao seu ambiente
    noise = OUNoise(action_dimension)
    noise.reset()

    print("Testando magnitude do ruído Ornstein-Uhlenbeck: Decay e Amplitude")
    print(f"Initial Decay Factor: {noise.decay_factor:.4f}, Min Decay Factor: {noise.min_decay_factor:.4f}, Decay Rate: {noise.decay_rate:.6f}")
    for i in range(noise.iterations_to_minimum + 100):
        sample = noise.sample()
        noise.decay()
        print(f"Sample {i+1}: {sample}, Magnitude: {np.linalg.norm(sample)}, Decay: {noise.decay_factor:.4f}, ")
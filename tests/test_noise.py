import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.agent.DDPG.noise import OUNoise

if __name__ == "__main__":
    action_dimension = 1
    noise = OUNoise(action_dimension)
    noise.reset()

    print("Testando magnitude do ruído Ornstein-Uhlenbeck:")

    max_norm = 0.0 
    norm_sum = 0.0 
    norms = []     

    total_iterations = 500
    for i in range(total_iterations):
        sample = noise.sample()
        norm = np.linalg.norm(sample)
        norm_sum += norm  
        norms.append(norm)
        percentual = 100 * norm
    
        if norm > max_norm:
            max_norm = norm

        print(f"Iteração {i+1:2d}: Ruído = {sample}, Norma = {norm:.6f}, Percentual do máximo = {percentual:.2f}%")

    print("\nMaior valor do ruído (norma): {:.6f}".format(max_norm))

    avg_norm = norm_sum / total_iterations
    print("Média do módulo dos ruídos: {:.6f}".format(avg_norm))

    std_norm = np.std(norms)
    print("Desvio padrão do módulo dos ruídos: {:.6f}".format(std_norm))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, total_iterations + 1), norms, marker='o')
    plt.title('Iteração x Valor do Ruído (Norma)')
    plt.xlabel('Iteração')
    plt.ylabel('Norma do Ruído')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("noise_plot.png") 

import numpy as np
from .model import  Controller
from src.utils.noise import OUNoise

class Agent:

    def __init__(self):
        self.controller = Controller()
        self.noise = OUNoise(action_dimension=1, category='action')

    def reset(self):
        """
        Reseta o controlador e o ruído.
        """
        self.controller.reset()
        self.noise.reset()

    def act(self, state: dict, add_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Gera uma ação para o estado atual, com ou sem ruído de exploração.

        Parâmetros
        ----------
        state : dict
            Estado atual do ambiente.
        add_noise : bool, opcional
            Se True, adiciona ruído de exploração à ação (padrão: True).

        Retorna
        -------
        action : np.ndarray
            Ação sugerida pelo controlador (com ou sem ruído).
        noise : np.ndarray
            Ruído aplicado à ação.
        """
        state = (state.get('position'), state.get('velocity'))
        action = self.controller.act(state)
        noise = np.zeros(1, dtype=np.float32)
        if add_noise:
            noise = self.noise.sample()
            action = action + noise
        action = np.clip(action, self.controller.output_min, self.controller.output_max)
        action = np.round(action, 4)
        return action, noise
       
import numpy as np
import pygame

class Metrics:
    """
    Classe responsável por exibir métricas na tela (posição, tempo, etc).
    """
    def __init__(self, font_size=24, color=(0, 0, 0)):
        pygame.font.init()
        self.font = pygame.font.SysFont(None, font_size)
        self.color = color

    def draw(self, screen, metrics_dict, pos=(10, 10)):
        """
        metrics_dict: dicionário com nome e valor das métricas
        pos: posição inicial do texto
        """
        x, y = pos
        for key, value in metrics_dict.items():
            # Round float values to 4 decimal places
            if isinstance(value, (float, np.float64, np.float32, np.float16)):
                value = float(value)
            
            text = f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            img = self.font.render(text, True, self.color)
            screen.blit(img, (x, y))
            y += self.font.get_height() + 5

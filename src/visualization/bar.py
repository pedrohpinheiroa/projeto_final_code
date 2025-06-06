import pygame
import math

class SeesawBar:
    """
    Classe responsável por desenhar a barra (seesaw) na tela.
    """
    def __init__(self, center, length=500, thickness=20, color=(150, 150, 150), pivot_color=(0, 0, 0)):
        self.center = center
        self.length = length
        self.thickness = thickness
        self.color = color
        self.pivot_color = pivot_color

    def draw(self, screen, angle_rad=0.0):
        # Calcula os quatro vértices do retângulo rotacionado
        cx, cy = self.center
        l = self.length
        t = self.thickness
        # Coordenadas dos vértices em relação ao centro (antes da rotação)
        corners = [
            (-l/2, -t/2),
            ( l/2, -t/2),
            ( l/2,  t/2),
            (-l/2,  t/2)
        ]
        # Aplica rotação
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rotated = [
            (cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a)
            for (x, y) in corners
        ]
        pygame.draw.polygon(screen, self.color, rotated)
        pygame.draw.circle(screen, self.pivot_color, self.center, 15)

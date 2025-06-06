import pygame
import json

class Window:
    """
    Classe responsável por criar e gerenciar a janela do pygame.
    Agora lê as configurações do arquivo configs/window.json.
    """
    def __init__(self, width=800, height=600, title="Simulação Seesaw"):
        self.width = width
        self.height = height
        self.title = title
        self.screen = None
        self.running = False

    def init_window(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        self.running = True

    def fill(self, color):
        if self.screen:
            self.screen.fill(color)

    def update(self):
        pygame.display.flip()

    def close(self):
        pygame.quit()
        self.running = False

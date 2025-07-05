import pygame
import math
from abc import ABC, abstractmethod
from .window import Window
from .bar import SeesawBar
from .metrics import Metrics
from .setpoint import SetPointSlider

class BaseVisualizer(ABC):
    """
    Classe base abstrata para visualização de ambientes usando pygame.
    Subclasses devem implementar o método render.
    """
    def __init__(self, width=800, height=600, title="Visualização do Ambiente"):
        self.width = width
        self.height = height
        self.title = title
        self.screen = None
        self.running = False

    def init_window(self):
        """Inicializa a janela do pygame."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        self.running = True

    @abstractmethod
    def render(self, *args, **kwargs):
        """
        Renderiza o ambiente.
        Este método deve ser implementado pelas subclasses.
        """
        pass

    def close(self):
        """Encerra a janela do pygame."""
        pygame.quit()
        self.running = False

class SeesawVisualizer:
    """
    Visualizador para o ambiente Seesaw, recebe componentes de janela, barra e métricas.
    Agora inclui slider de setpoint.
    """
    def __init__(self, include_setpoint_slider=False):
        self.window = Window()
        self.bar = SeesawBar(center=(self.window.width // 2, self.window.height // 2))
        self.metrics = Metrics()
        self.setpoint_slider = SetPointSlider(x=self.window.width-320, y=20)
        self.bg_color = (240, 240, 240)
        self.include_setpoint_slider = include_setpoint_slider

    def render(self, angle_rad=0.0, metrics_dict=None):
        if not self.window.screen:
            self.window.init_window()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.window.close()
                return
            if self.include_setpoint_slider:
                self.setpoint_slider.handle_event(event)
        self.window.fill(self.bg_color)
        self.bar.draw(self.window.screen, angle_rad)
        if metrics_dict:
            self.metrics.draw(self.window.screen, metrics_dict)
        
        if self.include_setpoint_slider:
            self.setpoint_slider.draw(self.window.screen)
        
        self.window.update()

    def close(self):
        self.window.close()

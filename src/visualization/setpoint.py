import pygame

class SetPointSlider:
    """
    Slider para ajuste do setpoint na visualização.
    Permite variar de -0.35 até 0.35 e interage com uma variável global de setpoint.
    """
    def __init__(self, x, y, width=300, min_value=-0.35, max_value=0.35, initial_value=0.0, height=8, handle_radius=12, color=(100, 100, 100), handle_color=(0, 120, 255)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_value = min_value
        self.max_value = max_value
        self.value = initial_value
        self.handle_radius = handle_radius
        self.color = color
        self.handle_color = handle_color
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            hx = self.get_handle_pos()
            if abs(mx - hx) <= self.handle_radius and abs(my - self.y) <= self.handle_radius:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mx, _ = pygame.mouse.get_pos()
            rel_x = min(max(mx, self.x), self.x + self.width)
            percent = (rel_x - self.x) / self.width
            self.value = self.min_value + percent * (self.max_value - self.min_value)
            # Aqui você pode atualizar a variável global do setpoint
            # Exemplo: global_setpoint_var = self.value

    def get_handle_pos(self):
        percent = (self.value - self.min_value) / (self.max_value - self.min_value)
        return int(self.x + percent * self.width)

    def draw(self, screen):
        # Linha do slider
        pygame.draw.rect(screen, self.color, (self.x, self.y - self.height // 2, self.width, self.height), border_radius=4)
        # Handle
        hx = self.get_handle_pos()
        pygame.draw.circle(screen, self.handle_color, (hx, self.y), self.handle_radius)
        # Valor numérico
        font = pygame.font.SysFont(None, 24)
        text =f"Setpoint: {self.value:.3f}"
        value_text = font.render(text, True, (0, 0, 0))
        screen.blit(value_text, (self.x, self.y + 12))

    def get_value(self):
        """
        Retorna o valor atual do setpoint do slider.
        """
        return self.value
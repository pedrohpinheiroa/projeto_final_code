import time
import math
from src.visualization.window import Window
from src.visualization.bar import SeesawBar
from src.visualization.metrics import Metrics
from src.visualization import SeesawVisualizer

if __name__ == "__main__":
    # Parâmetros da janela
    width, height = 800, 600
    center = (width // 2, height // 2)

    # Instanciando os componentes
    window = Window(width, height, "Demo Seesaw")
    bar = SeesawBar(center=center)
    metrics = Metrics()
    visualizer = SeesawVisualizer(window, bar, metrics)

    # Inicializa a janela explicitamente
    window.init_window()

    angle = 0.0
    start_time = time.time()
    while window.running:
        # Atualiza o ângulo para girar a barra
        angle = math.sin(time.time() - start_time) * math.pi / 4  # oscila entre -45 e +45 graus
        # Exemplo de métricas
        metrics_dict = {
            "Ângulo (graus)": f"{math.degrees(angle):.2f}",
            "Tempo (s)": f"{time.time() - start_time:.1f}"
        }
        visualizer.render(angle_rad=angle, metrics_dict=metrics_dict)
        time.sleep(0.016)  # ~60 FPS
    visualizer.close()

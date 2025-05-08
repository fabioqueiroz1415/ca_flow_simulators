import matplotlib.pyplot as plt
import matplotlib.animation as anm
import numpy as np
import os

class DynamicPlot:
    def __init__(self, rules_function, n_iterations, cmap_elev="terrain", cmap_water="Blues"):
        """
        Inicializa dois gráficos: relevo original e espalhamento da água.
        
        Args:
            elevation_matrix (np.ndarray): Matriz de elevação original.
            water_matrix (np.ndarray): Matriz inicial da água.
            cmap_elev (str): Colormap para o relevo. Default: 'terrain'.
            cmap_water (str): Colormap para a água. Default: 'Blues'.
        """
        self.fig, (self.ax_elev, self.ax_water) = plt.subplots(1, 2, figsize=(12, 5))
        
        self.n_iterations = n_iterations
        self.rules_function = rules_function
        self.cmap_elev = cmap_elev
        self.cmap_water = cmap_water

        plt.ion()
        plt.tight_layout()
        plt.show()

    def set_plot(self, elevation_matrix, water_matrix, output_dir="default", video_name="output"):
        self.n, self.m = elevation_matrix.shape
        self.elevation_matrix = elevation_matrix
        self.water_matrix = water_matrix
        self.output_dir = f"./{output_dir}"
        self.video_name = video_name
        self.iteration = 1
        
        os.makedirs(self.output_dir, exist_ok=True)

        # Gráfico do relevo original
        self.cax_elev = self.ax_elev.matshow(elevation_matrix, cmap=self.cmap_elev)
        plt.colorbar(self.cax_elev, ax=self.ax_elev, label="Elevação")
        self.ax_elev.set_title("Relevo Original")
        
        # Gráfico da água
        self.cax_water = self.ax_water.matshow(water_matrix, cmap=self.cmap_water, vmin=0, vmax=50)
        plt.colorbar(self.cax_water, ax=self.ax_water, label="Água")
        self.ax_water.set_title("Água - Iteração 1")

    def update(self, new_water_matrix):
        """Atualiza apenas o gráfico da água."""
        self.iteration += 1
        self.cax_water.set_array(new_water_matrix)
        self.cax_water.set_clim(vmin=0, vmax=50)  # Mantém a escala fixa
        self.ax_water.set_title(f"Água - Iteração {self.iteration}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot(self):
        height_init = self.elevation_matrix
        height = height_init.copy()
        EV = self.water_matrix

        total_inicial = np.sum(EV) + np.sum(height - height_init)
        
        print(f"Total inicial: {total_inicial:.8f}")

        # Configurar vídeo writer
        output_path = os.path.join(self.output_dir, f"{self.video_name}.mp4")
        writer = anm.FFMpegWriter(fps=30, metadata=dict(artist='Simulation'), bitrate=1800)

        with writer.saving(self.fig, output_path, dpi=200):
            for _ in range(self.n_iterations):
                EV, height = self.rules_function(EV, height, self.n, self.m)
                
                # Verificação rigorosa de conservação de massa
                total_atual = np.sum(EV) + np.sum(height - height_init)
                if not np.isclose(total_atual, total_inicial, atol=1e-6):
                    print(f"ERRO: Perda de {total_inicial - total_atual:.10f} na iteração {_+1}")
                    break
                if abs(EV.sum()) < 1e-6:
                    print("Água esgotada, encerrando simulação.")
                    break
                print(f"Iteração {_+1}: EV: {np.sum(EV):.8f}, Agua acomodada: {np.sum(height - height_init):.8f}")
                
                self.update(EV + height - height_init)
                #frame_path = os.path.join(self.output_dir, f"frame_{_+1:06d}.png")
                #self.fig.savefig(frame_path, dpi=200, bbox_inches='tight')

                writer.grab_frame()  # salva o frame no vídeo

                plt.pause(0.00001)

        print("Simulação concluída com conservação de massa!")

    def close(self):
        plt.ioff()
        plt.close(self.fig)
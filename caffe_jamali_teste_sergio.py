import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os
import cv2

def bicubic_interpolation_opencv(image, scale_factor=10):
    return cv2.resize(
        image, 
        None, 
        fx=scale_factor, 
        fy=scale_factor, 
        interpolation=cv2.INTER_CUBIC
    )

def extract_elev_Tiff(filepath_tiff):
    with rasterio.open(filepath_tiff) as src:
        return src.read(1).astype(np.float64)  # Já converte para float

class DynamicPlot:
    def __init__(self, elevation_matrix, water_matrix, cmap_elev="terrain", cmap_water="Blues"):
        """
        Inicializa dois gráficos: relevo original e espalhamento da água.
        
        Args:
            elevation_matrix (np.ndarray): Matriz de elevação original.
            water_matrix (np.ndarray): Matriz inicial da água.
            cmap_elev (str): Colormap para o relevo. Default: 'terrain'.
            cmap_water (str): Colormap para a água. Default: 'Blues'.
        """
        self.fig, (self.ax_elev, self.ax_water) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico do relevo original
        self.cax_elev = self.ax_elev.matshow(elevation_matrix, cmap=cmap_elev)
        plt.colorbar(self.cax_elev, ax=self.ax_elev, label="Elevação")
        self.ax_elev.set_title("Relevo Original")
        
        # Gráfico da água
        self.cax_water = self.ax_water.matshow(water_matrix, cmap=cmap_water, vmin=0, vmax=10)
        plt.colorbar(self.cax_water, ax=self.ax_water, label="Água")
        self.ax_water.set_title("Água - Iteração 1")
        
        self.iteration = 1
        plt.ion()
        plt.tight_layout()
        plt.show()

    def update(self, new_water_matrix):
        """Atualiza apenas o gráfico da água."""
        self.iteration += 1
        self.cax_water.set_array(new_water_matrix)
        self.cax_water.set_clim(vmin=0, vmax=10)  # Mantém a escala fixa
        self.ax_water.set_title(f"Água - Iteração {self.iteration}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)

def applyRules(EV, height, n, m):
    EV_prev = EV.copy()
    height_prev = height.copy()
    new_EV = EV.copy()  # Mantemos o EV inicial
    new_height = height.copy()
    
    # Matrizes buffers para acumular mudanças em EV e height
    delta_EV = np.zeros_like(EV)
    delta_height = np.zeros_like(height)
    
    # Controle de precisão
    eps = 1e-10
    increment_constant = 0.1

    for i in range(1, n-1):
        for j in range(1, m-1):
            current_EV = EV_prev[i, j]
            current_height = height_prev[i, j]
            
            # ---- Regra 0:  ----
            # Se não há volume excedente, não faz nada
            if current_EV < eps:
                continue # não faz nada e continua na proxima iteração dos laços fo
            
            # ---- Regra 1: Equalização com vizinho mais baixo ----
            # SE A ALTURA DA CÉLULA CENTRAL É MENOR QUE A DE TODOS OS VIZINHOS:
            # TRANSFERE DO VOLUME EXCEDENTE PARA A CELULA CENTRAL
            # A QUANTIDADE TRANSFERIDA É REGULADA PELA DIFERENÇA DE ALTURA
            # ENTRE O VIZINHO MAIS BAIXO E A CÉLULA CENTRAL, NÃO PODENDO
            # SER MAIOR QUE O VOLUME EXCEDENTE. 
            neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
            neighbor_heights = [height_prev[x,y] for x,y in neighbors]
            min_height = min(neighbor_heights)
            
            if current_height < min_height:
                transfer = min(min_height - current_height, current_EV)
                delta_height[i,j] += transfer
                delta_EV[i,j] -= transfer
                continue

            # ---- Regra 2: Espalhamento em platôs - Spreading ----
            # SE A ALTURA DA CÉLULA CENTRAL É IGUAL A DE TODOS OS VIZINHOS:
            # ESPALHA O VOLUME EXCEDENTE IGUALMENTE NAS 5 CELULAS.
            # COMO OS VALORES SÃO DO TIPO FLOAT, A COMPARAÇÃO É FEITA COM TOLERÂNCIA
            if all(abs(h - current_height) < 0.001 for h in neighbor_heights):
                transfer = current_EV / 5.0 # Distribuição uniforme nas 5 células
                delta_EV[i,j] += transfer
                for x,y in neighbors:
                    delta_EV[x,y] += transfer
                delta_EV[i,j] -= current_EV  # Remove todo o EV
                continue

            # ---- Regra 3: Aumento de nível ----
            #CONDIÇÃO: CÉLULA CENTRAL É TEM MESMA ALTURA (HEIGHT) QUE UM, DOIS OU TRÊS VIZINHOS
            # E É MAIS BAIXA QUE OUTRAS
            #PROCEDIMENTO: SE HA EV[I][J], INCREMENTA A ALTURA DA CÉLULA CENTRAL
            #  POR UM increment_constant. O VALOR RESTANTE DE EV É DISTRIBUÍDO
            #  PARA AS CÉLULAS DE MESMA ALTURA.  
            same_level = [(x,y) for x,y in neighbors  
                         if abs(height_prev[x,y] - current_height) < 0.001] # vizinhos no mesmo nível
            non_same_level = [(x,y) for x,y in neighbors if (x,y) not in same_level]
            all_higher = all(height_prev[x,y] > current_height for x,y in non_same_level) # vizinhos mais altos
            if same_level and all_higher: #deve haver pelo menos um vizinho mais alto
                max_transfer = min(current_EV*0.2, 0.1)  # Transfere 10% do EV ou 0.1, o que for menor
                delta_height[i,j] += max_transfer
                remaining = current_EV - max_transfer
                
                # Distribui o restante
                per_neighbor = remaining / len(same_level)
                for x,y in same_level:
                    delta_EV[x,y] += per_neighbor # estava assim

                delta_EV[i,j] -= current_EV

                continue

            # ---- Regra 4: Fluxo DOWNSTREAM ----
            # A ALTURA DA CELULA CENTRAL E MAIOR A ALTURA DE PELO MENOS UM VIZINHOS
            # TODO O VOLUME EXCEDENTE FLOUIRÁ PARA AS CELULAS VIZINHAS MAIS BAIXAS PROPORCIONALMENTE AO DECLIVE
            a = 0.11
            b = 0.25
            hf = a * (current_EV ** b) # Friction head loss
            h_potencial = current_height + hf

            # Identifica vizinhos DOWNSTREAM (H_i < H0 + hf)
            diffs = []
            downstream_neighbors = []
            for x, y in neighbors:
                H_i = height_prev[x, y]
                if H_i < h_potencial:
                    diff = h_potencial - H_i
                    diffs.append(diff)
                    downstream_neighbors.append((x, y))
                else:
                    diffs.append(0)  # Vizinhos não downstream recebem peso 0

            total_diff = sum(diffs)

            if total_diff > eps:
                weights = np.array(diffs) / total_diff
                transferred = current_EV * weights

                # Atualiza buffers
                for k, (x, y) in enumerate(neighbors):
                    delta_EV[x, y] += transferred[k]

                # Remove todo o EV da célula central
                delta_EV[i, j] -= current_EV

    # Aplica as mudanças em EV e height após atualização de todas as células
    new_EV += delta_EV
    new_height += delta_height
    
    # Correção de conservação (garante que não há perda)
    total_antes = np.sum(EV_prev) + np.sum(height_prev)
    total_depois = np.sum(new_EV) + np.sum(new_height)
    
    # Garante que a soma total de EV e height antes e depois da iteração permanece constante
    # Se houver diferença, distribui uniformemente
    if not np.isclose(total_antes, total_depois, atol=1e-6):
        correction = (total_antes - total_depois) / (n*m)
        new_EV += correction  # Distribui diferença uniformemente
    
    return new_EV, new_height



def main():
    n_iterations = 2000
    scale_factor = 1

    output_dir = "./simulation_frames"
    
    # Cria diretório para salvar os frames
    os.makedirs(output_dir, exist_ok=True)

    # Carregar e preparar dados
    height_orig = extract_elev_Tiff('./elevacao_catalao.tif')


    max_height = np.max(height_orig)
    height = np.where(height_orig < 0, max_height, height_orig)
    #height = bicubic_interpolation_opencv(height, scale_factor=scale_factor)

    #height_orig = np.zeros((100, 100))  # Exemplo de matriz de elevação
    #height = height_orig.copy()
    #nx, ny = height.shape

    #EV = np.zeros((nx, ny))*0.0  # Inicialização da água
    #EV[nx//2-5:nx//2+5, ny//2-5:ny//2+5] = 100.0  # Região central com água


    height_init = height.copy()

    # Dimensoes de matriz
    n, m = height_orig.shape
    
    # Definir região inicial de água
    EV = np.random.rand(n, m)*0.4 # Chuva pancada aleatória até 40mm por célula
    EV = np.where(height_orig < 0, 0, EV) # Colocando água apenas na área da cidade de catalão
    #EV = bicubic_interpolation_opencv(EV, scale_factor=scale_factor)


    # Inicializar plot
    plotter = DynamicPlot(height_init, EV, cmap_elev="viridis", cmap_water="Blues")

    total_inicial = np.sum(EV) + np.sum(height - height_init)
    print(f"Total inicial: {total_inicial:.8f}")

    
    for _ in range(n_iterations):
        EV, height = applyRules(EV, height, n, m)
        
        # Verificação rigorosa de conservação de massa
        #total_atual = np.sum(EV) + np.sum(height - height_init)
        #if not np.isclose(total_atual, total_inicial, atol=1e-6):
        #    print(f"ERRO: Perda de {total_inicial - total_atual:.10f} na iteração {_+1}")
        #    break
        
        # Simula mais chuva

        EV_inc = np.random.rand(n, m)*0.1 # Incremento de chuva aleatória  
        EV_inc = np.where(height_orig < 0, 0, EV_inc) # Considerando chuva apenas na área da cidade de catalão
        EV += EV_inc # Incrementa a chuva no EV

        #EV_inc = np.zeros((nx, ny))*0.0  # Inicialização da água
        #EV_inc[nx//2-5:nx//2+5, ny//2-5:ny//2+5] = 1.0  # Região central com água
        #EV += EV_inc

        

        if abs(EV.sum()) < 1e-6:
            print("Água esgotada, encerrando simulação.")
            break
        print(f"Iteração {_+1}: EV: {np.sum(EV):.8f}, Agua acomodada: {np.sum(height - height_init):.8f}")
        
        plotter.update(EV + height - height_init)
        frame_path = os.path.join(output_dir, f"frame_{_+1:06d}.png")
        plotter.fig.savefig(frame_path, dpi=200, bbox_inches='tight')

        plt.pause(0.00001)

    print("Simulação concluída com conservação de massa!")


if __name__ == '__main__':
    main()
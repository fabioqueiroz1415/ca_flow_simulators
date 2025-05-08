import numpy as np

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
                delta_height[i,j] += transfer
                for x,y in neighbors:
                    delta_height[x,y] += transfer
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

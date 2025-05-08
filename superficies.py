#gera superficies
import numpy as np

def paraboloide(size=350, scale=1.0):
    # Cria um grid de coordenadas X e Y centrado
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Fórmula do paraboloide
    Z = scale * (X**2 + Y**2)
    return Z

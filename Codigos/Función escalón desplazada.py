# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 08:54:31 2023

@author: sebas
"""

import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.where(x >= 0, 1, 0)

# Generar valores de x en el rango de -10 a 10
x = np.linspace(-2, 2, 100)

# Calcular los valores de y para la función escalón
y1 = step_function(x)

# Calcular los valores de y para la función escalón desplazada
y2 = step_function(x - 0.5)
fig, ax = plt.subplots(figsize=(10, 6))
# Graficar ambas funciones en el mismo plot
#plt.plot(x, y1, color='blue', label='sin bias')
plt.plot(x, y2, color='red') #label='con bias')

# Agregar títulos y leyendas
plt.title('Gráfico de los valores de salida', fontweight='bold')
plt.xlabel('w*x', fontweight='bold')
plt.ylabel('Magnitud', fontweight='bold')
plt.legend()
ax.grid(True)
# Mostrar el gráfico
plt.show()

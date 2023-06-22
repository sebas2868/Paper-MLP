# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:43:05 2023

@author: sebas
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

caen = np.random.uniform(220, 298, 30)
distancia1 = np.random.uniform(500, 550, 30)
no_c = np.random.uniform(100, 185, 30)
distancia2 = np.random.uniform(565, 590, 30)
ones = np.ones((30))
zeros = np.zeros((30))
data_set = np.column_stack((np.concatenate((caen, no_c)), np.concatenate((distancia1, distancia2)), np.concatenate((ones, zeros))))


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data_set[0:30,1],data_set[0:30,0],color='red', edgecolors='k', alpha=0.9, marker='o', label='Cayó (1)')
ax.scatter(data_set[30:60,1],data_set[30:60,0],color='blue', edgecolors='k', alpha=0.9, marker='o', label='No Cayó (0)')
ax.set_xlabel('Altura Orbital (Km)', fontweight='bold',  fontsize=15)
ax.set_ylabel('Masa (Kg)', fontweight='bold', fontsize=15)
plt.title('Grafica del conjunto de datos', fontweight='bold', fontsize=15)
# Crear las leyendas para cada categoría
# Agregar la leyenda
plt.legend()
ax.grid(True)

plt.show()
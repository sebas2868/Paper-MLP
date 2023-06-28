# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:43:05 2023

@author: sebas
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
points = 30
caen = np.random.uniform(210, 278, points)
distancia1 = np.random.uniform(500, 530, points)
no_c = np.random.uniform(100, 185, points)
distancia2 = np.random.uniform(540, 570, points)
ones = np.ones((points))
zeros = np.zeros((points))
data_set = np.column_stack((np.concatenate((caen, no_c)), np.concatenate((distancia1, distancia2)), np.concatenate((ones, zeros))))

#datos de prueba: 
points = 10
caen = np.random.uniform(215, 250, points)
distancia1 = np.random.uniform(515, 535, points)
no_c = np.random.uniform(150, 200, points)
distancia2 = np.random.uniform(540, 560, points)
ones = np.ones((points))
zeros = np.zeros((points))
data_test = np.column_stack((np.concatenate((caen, no_c)), np.concatenate((distancia1, distancia2)), np.concatenate((ones, zeros))))
muestras = np.concatenate((data_set, data_test), axis=0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data_set[0:30,1],data_set[0:30,0],color='red', edgecolors='k', alpha=0.9, marker='o', label='Cayó (1)')
ax.scatter(data_set[30:60,1],data_set[30:60,0],color='blue', edgecolors='k', alpha=0.9, marker='o', label='No Cayó (0)')
ax.scatter(data_test[0:10,1],data_test[0:10,0],color='yellow', edgecolors='k', alpha=0.9, marker='o', label='Cayó (1)')
ax.scatter(data_test[10:20,1],data_test[10:20,0],color='green', edgecolors='k', alpha=0.9, marker='o', label='No Cayó (0)')
ax.set_xlabel('Altura Orbital (Km)', fontweight='bold',  fontsize=15)
ax.set_ylabel('Masa (Kg)', fontweight='bold', fontsize=15)
plt.title('Grafica del conjunto de datos', fontweight='bold', fontsize=15)
# Crear las leyendas para cada categoría
# Agregar la leyenda
plt.legend()
ax.grid(True)

plt.show()
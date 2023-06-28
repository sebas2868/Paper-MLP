# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:35:48 2023

@author: sebas
"""
import numpy as np #libreria para el manejo de matrices
import matplotlib.pyplot as plt #libreria utilizada para la graficación

x = np.random.uniform(0, 1, size=(2)) #Vector de entrada entre 1 y 0
yd = 1 #valor esperado
eta = 0.9
W = np.random.uniform(-0.5, 0.5, size=(1, 2)) #Vector de pesos sinapticos
W_new = []
Y_new = []
def sigmoid(x): #Sigmoide
    return 1 / (1 + np.exp(-x))

for i in range(10): #numero
    I = np.dot(W,x) #entrada neta a la neurona 
    Y = sigmoid(I) #Salida
    Y_new.append(Y)
    error = yd-Y # calculo del error
    print(error)
    print(W)
    W = W+error*x #regla de aprendizaje 
    W_new.append(W)

x_grap = list(range(1, len(Y_new) + 1))


# Datos de los puntos de inicio y final de la línea

plt.figure(figsize=(10, 6))
plt.xlim(0, 11)  # Limitar el eje x desde 1 hasta 4
plt.ylim(0.4, 1.1)  # Limitar el eje y desde 2 hasta 5
x_values = [-1, 12]
y_value = 1
plt.plot(x_values, [y_value, y_value], color='red', label='Salida deseada', linewidth=2.5)
x_values = [-1, 12]
y_value = 1
plt.plot(x_values, [y_value, y_value], color='blue', label='Salida real', linestyle='dashed', linewidth=2.5)
plt.xlabel("Iteración")
plt.ylabel("Valor salida")
plt.grid(True)
plt.legend()
# plt.scatter(W_new[0,0], W_new[0,1])
plt.title('Regla de aprendizaje', fontweight='bold')

plt.figure(figsize=(10, 6))
plt.xlim(0, 11)  # Limitar el eje x desde 1 hasta 4
plt.ylim(0.4, 1.1)  # Limitar el eje y desde 2 hasta 5
plt.scatter(x_grap, Y_new, label='Salida real')
plt.plot(x_values, [y_value, y_value], color='red', label='Salida deseada', linewidth=2.5)
plt.xlabel("Iteración")
plt.ylabel("Valor salida")
plt.grid(True)
plt.legend()
# plt.scatter(W_new[0,0], W_new[0,1])
plt.title('Regla de aprendizaje', fontweight='bold')

    
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:35:48 2023

@author: sebas
"""
import numpy as np #libreria para el manejo de matrices
import matplotlib.pyplot as plt #libreria utilizada para la graficación

x = np.array([1, 2]) #entrada
yd = 0.8 #valor esperado

W = np.random.uniform(-0.5, 0.5, size=(1, 2)) #Vector de pesos sinapticos
W_new = []
Y_new = []
def sigmoid(x): #Sigmoide
    return 1 / (1 + np.exp(-x))

for i in range(10):
    I = np.dot(W,x) #entrada neta a la neurona 
    Y = sigmoid(I) #Salida
    Y_new.append(Y)
    error = yd-Y # calculo del error
    print(error)
    print(W)
    W = W+error*x #regla de aprendizaje 
    W_new.append(W)

x_grap = list(range(1, len(Y_new) + 1))
    
plt.figure()
plt.scatter(x_grap, Y_new)
plt.xlabel("epoca")
plt.ylabel("Predicción")
# plt.scatter(W_new[0,0], W_new[0,1])


    
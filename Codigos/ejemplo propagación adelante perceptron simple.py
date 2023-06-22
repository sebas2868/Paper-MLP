# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:35:49 2023

@author: sebas
"""

#Librerias que vamos a usar: 
import matplotlib.pyplot as plt #libreria utilizada para la graficación
import numpy as np #libreria para el manejo de matrices
import pandas as pd #libreria para la exportación del data set

#Importación del data set
data_set = pd.read_excel(r'C:/Users/sebas/Desktop/Semillero/Neuro_paper/Data_sets/Data_set_perceptrón_simple_Basura_espacial.xlsx') #ruta del archivo de training 
data_set_values = data_set.values #obteniendo solo los valores 

#inclusión del bias: 
b=1 #valor del bias
data_set = np.column_stack((b*np.ones(len(data_set_values)), data_set_values)) #data set con bias

#Conjuto de enetrenamineto y de prueba:
train_set = data_set[0:60,:] #conjunto de entrenamiento
test_set = data_set[60:,:] #conjunto de prueba


#Valores del perceptrón:
W = np.random.uniform(-0.5, 0.5, size=(1, 3)) #inicialización pesos sinapticos
#Función de activación:
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = train_set[0,0:3] #vector de entrada 
#Propagación adelante
I = np.dot(W,x) #entrada neta a la neurona 
Y = sigmoid(I) #Salida

yd = data_set[0,3:] #salida esperada del vector de entrada
error = yd-Y # calculo del error


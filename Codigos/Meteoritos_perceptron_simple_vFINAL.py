# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:52:31 2023

@author: sebas
"""

#Librerias que vamos a usar: 
import matplotlib.pyplot as plt #libreria utilizada para la graficación
import numpy as np #libreria para el manejo de matrices
import pandas as pd #libreria para la exportación del data set


#Importación del data set
data_set = pd.read_excel(r'C:/Users/sebas/Desktop/Semillero/Neuro_paper/Data_sets/Data_set_perceptrón_simple.xlsx') #ruta del archivo de training 
data_set_values = data_set.values #obteniendo solo los valores 
data_set1 = data_set_values
#inclusión del bias: 
b=1 #valor del bias
data_set = np.column_stack((b*np.ones(len(data_set_values)), data_set_values)) #data set con bias

#Conjuto de enetrenamineto y de prueba:
train_set = data_set[0:60,:] #conjunto de entrenamiento
test_set = data_set[60:,:] #conjunto de prueba

#Valores del perceptrón:
W = np.random.uniform(0, 0.1, size=(1, 3)) #inicialización pesos sinapticos
eta_i = 0.001 #Tasa de aprendizaje valor incial
eta_f = 0.00001 #Tasa de aprendizaje valor final
nEpocas = 8 #numero de epocas
emedio = [0]*nEpocas #lista para almacenar el costo del entrenamiento
emedio_test= [0]*nEpocas #lista para almacanar el costo de prueba

#Función de activación:
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#datos de prueba
x_test = test_set[:,0:3] #entradas de prueba
y_test = test_set[:,3:] #salidas esperadas de prueba


for epoca in range(nEpocas): # Ciclo para número de épocas
    np.random.shuffle(train_set) # Mezclar las muestras de entrenamiento
    eta = eta_i+(eta_f - eta_i)*(epoca / nEpocas) # Descenso lineal
    x = train_set[:, 0:3]# Entradas de entrenamiento
    y = train_set[:, 3:]# Salidas de entrenamiento
    sumt = 0 #Suma del error de cada muestra de entrenamiento en una época
    sumt_test = 0 #Suma del error de cada muestra de prueba en una época
    
    for j in range(train_set.shape[0]): # Ciclo de muestras de entrenamiento
        I = np.dot(W, x[j, :]) # Potencial de activación
        Y = sigmoid(I) # Salida
        loss = 0.5 * (y[j, :] - Y) ** 2 # Pérdida
        error = y[j, :] - Y # Acumulación de la pérdida
        W = W + eta * error * x[j, :] # Actualización de los pesos
        sumt = sumt + loss # Acumulación de la pérdida de entrenamiento

    #ciclo para el test
    for i in range(test_set.shape[0]): #ciclo para prueba
        I = np.dot(W,x_test[i,:]) # Potencial de activación
        Y = sigmoid(I) # Salida
        loss_test = 0.5*(y_test[i,:]-Y)**2 # Pérdida
        sumt_test = sumt_test+loss_test # Acumulación de la pérdida
        
    emedio[epoca]=sumt/len(x) #obtención del coste de entrenamiento
    emedio_test[epoca]=sumt_test/len(x_test) #obtención del costo de prueba

epoca_x = list(range(1, nEpocas+1)) #arreglo con el numero de épocas
fig = plt.figure(figsize=(10, 6)) #definir el tamaño del plot
# Agregar el primer subplot
ax1 = fig.add_subplot(1, 2, 1) 
ax1.plot(epoca_x,emedio, label='coste_entrenamiento') #coste entrenamiento
ax1.plot(epoca_x,emedio_test, label='coste_prueba') #coste prueba
ax1.legend() 
ax1.set_xlabel("epoca", fontweight='bold', fontsize=15)
ax1.set_ylabel("MSE", fontweight='bold', fontsize=15)
ax1.set_title('Grafica de Desempeño, Épocas = {}'.format(nEpocas), fontweight='bold', fontsize=15)

ax2 = fig.add_subplot(1, 2, 2)
# #Graficación de las lienas
# x1 = np.random.uniform(150, 240, 100)
# x2 = -(W[0,0] + W[0,1]*x1)/W[0,2]
# ax2.plot(x1,x2, color='purple', linewidth=3.5)



ax2.set_xlim(98, 275)  # Limitar el eje x entre 2 y 4
ax2.set_ylim(490, 580)  # Limitar el eje y entre 4 y 8
ax2.scatter(data_set1[0:30,0],data_set1[0:30,1], color='red', label='Impacto (1)')
ax2.scatter(data_set1[30:60,0],data_set1[30:60,1], color='blue', label='NO Impacto (0)')
ax2.scatter(data_set1[60:70,0],data_set1[60:70,1], color='orange',label='Prueba (1)')
ax2.scatter(data_set1[70:,0],data_set1[70:,1], color='green',label='Prueba (0)')
ax2.set_ylabel('Altura Orbital (Km)', fontweight='bold',  fontsize=15)
ax2.set_xlabel('Masa (Kg)', fontweight='bold', fontsize=15)
ax2.set_title('Grafica del conjunto de datos', fontweight='bold', fontsize=15)
ax2.legend()
ax2.grid(True)


#Predecir:
xx, yy = np.meshgrid(np.arange(98, 275, 0.2),
                      np.arange(480, 580, 0.2))
mesh_points = np.c_[xx.ravel(), yy.ravel()]
test = np.column_stack((b*np.ones(len(mesh_points)), mesh_points))
value = np.zeros(len(mesh_points))
for j in range(len(test)):
      I = np.dot(W,test[j,:])
      Y = sigmoid(I)
      value[j] = Y
Z = value.reshape(xx.shape)
ax2.pcolormesh(xx, yy, Z, cmap='bwr', alpha=0.3)


# test = []
# for j in range(data_set.shape[0]):
#      I = np.dot(W,x[j,:])
#      Y = sigmoid(I)
#      test.append(Y)
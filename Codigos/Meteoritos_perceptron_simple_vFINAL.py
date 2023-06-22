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
data_set = pd.read_excel(r'C:/Users/sebas/Desktop/Semillero/Neuro_paper/Data_sets/Data_set_perceptrón_simple_Basura_espacial.xlsx') #ruta del archivo de training 
data_set_values = data_set.values #obteniendo solo los valores 
data_set1 = data_set_values
#inclusión del bias: 
b=1 #valor del bias
data_set = np.column_stack((b*np.ones(len(data_set_values)), data_set_values)) #data set con bias

#Conjuto de enetrenamineto y de prueba:
train_set = data_set[0:60,:] #conjunto de entrenamiento
test_set = data_set[60:,:] #conjunto de prueba


#Valores del perceptrón:
W = np.random.uniform(-0.5, 0.5, size=(1, 3)) #inicialización pesos sinapticos


eta_i = 0.01
eta_f = 0.00001
nEpocas = 20
epsilon = 1e-6
emedio = [0]*nEpocas
emedio_test= [0]*nEpocas
from scipy.special import expit #libreria para la obtencion

#Función de activación:
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_test = test_set[:,0:3]
y_test = test_set[:,3:]


for epoca in range(nEpocas):
    np.random.shuffle(data_set)
    eta = eta_i +(eta_f-eta_i)*(epoca/nEpocas)
    x = data_set[:,0:3]
    y = data_set[:,3:]
    sumt = 0
    sumt_test = 0
    for j in range(data_set.shape[0]):
        I = np.dot(W,x[j,:])
        Y = sigmoid(I)
        loss = 0.5*(y[j,:]-Y)**2
        error = y[j,:]-Y
        W = W+eta*error*x[j,:]
        sumt = sumt+loss
    #ciclo para el test
    for i in range(test_set.shape[0]):
        I = np.dot(W,x_test[i,:])
        Y = sigmoid(I)
        loss_test = y_test[i,:]-Y
        sumt_test = sumt_test+0.5*loss_test**2
    emedio[epoca]=sumt/len(x)
    emedio_test[epoca]=sumt_test/len(x_test)
    
plt.figure()
plt.plot(emedio, label='coste_entrenamiento')
plt.plot(emedio_test, label='coste_prueba')
plt.legend()
plt.xlabel("epoca")
plt.ylabel("MSE")

plt.figure()
#Graficación de las lienas
x1 = np.random.uniform(150, 210, 100)
x2 = -(W[0,0] + W[0,1]*x1)/W[0,2]
plt.figure(figsize=(10, 6))
plt.plot(x1,x2)
plt.show()


# plt.figure()
# plt.scatter(norm_data_set[:,1],norm_data_set[:,2])

#plt.figure()
plt.xlim(100, 275)  # Limitar el eje x entre 2 y 4
plt.ylim(510, 650)  # Limitar el eje y entre 4 y 8
plt.scatter(data_set1[0:30,0],data_set1[0:30,1], color='red', label='cayo (1)')
plt.scatter(data_set1[30:60,0],data_set1[30:60,1], color='blue', label='NO cayo (0)')
plt.scatter(data_set1[60:65,0],data_set1[60:65,1], color='purple')
plt.scatter(data_set1[65:,0],data_set1[65:,1], color='green')
plt.ylabel('Altura Orbital (Km)', fontweight='bold',  fontsize=15)
plt.xlabel('Masa (Kg)', fontweight='bold', fontsize=15)
plt.title('Grafica del conjunto de datos', fontweight='bold', fontsize=15)
plt.legend()
plt.grid(True)


#Predecir:
xx, yy = np.meshgrid(np.arange(98, 300, 0.2),
                      np.arange(480, 660, 0.2))
mesh_points = np.c_[xx.ravel(), yy.ravel()]
test = np.column_stack((b*np.ones(len(mesh_points)), mesh_points))
value = np.zeros(len(mesh_points))
for j in range(len(test)):
      I = np.dot(W,test[j,:])
      Y = sigmoid(I)
      value[j] = Y
Z = value.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)
plt.show()

# test = []
# for j in range(data_set.shape[0]):
#      I = np.dot(W,x[j,:])
#      Y = sigmoid(I)
#      test.append(Y)
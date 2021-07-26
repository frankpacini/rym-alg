import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

import csv
import math

def normX(X, data):
    X_0 = ((X[:,0] - np.min(data[:,0]))/math.sqrt(np.var(data[:,0])))[:,None]
    X_1 = ((X[:,1] - np.min(data[:,1]))/math.sqrt(np.var(data[:,1])))[:,None]
    # X_dist = X[:,2:] / data[:,1][:,None]
    return np.concatenate((X_0,X_1), 1)

def normY(y):
    return (np.mean(y) - y)/math.sqrt(np.var(y))

def getPredictions(model, X, y, data):
    return model.predict(normX(X, data)).flatten("F") * -math.sqrt(np.var(y)) + np.mean(y)

arr = np.array(list(csv.reader(open("rym-links-rand.csv", "r"), delimiter=",")))


data = np.array(list(csv.reader(open("rym.csv", "r"), delimiter=","))).astype("float")[:1075,:]
x = data[:,0]
y = data[:,1].astype("int")
z = arr[:,1].astype("int")[:x.shape[0]]
# z =  (z - np.mean(z)) / -math.sqrt(np.var(z)) 


# sx = np.arange(3.75, 4.85, 0.02)
# sy = np.arange(250, 60000, 10)
sx = np.arange(3.5, 4.9, 0.02)
sy = np.arange(150, 4000, 5)
sX, sY = np.meshgrid(sx, sy)

# print(sX[1800][11])
# print(sY[1800][11])

input = np.column_stack((np.ravel(sX), np.ravel(sY)))
# fill = np.ones((input.shape[0], input.shape[1]*5))
# input = np.column_stack((input, fill))

models = ["./model3"]

for m in models:
    model = keras.models.load_model(m)
    sz = np.array(getPredictions(model, input, z, data))
    sZ = sz.reshape(sX.shape)
    # print(sZ[1800][11])

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    # ax.scatter(x, y, z)
    ax.plot_surface(sX, sY, sZ)
    plt.title(m)

    plt.show()

# 1800, 11
# 3.97, 18250
# -1.4708769
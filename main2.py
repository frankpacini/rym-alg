from tensorflow import keras
import numpy as np
import csv
import math

arr = np.array(list(csv.reader(open("rym-links-rand.csv", "r"), delimiter=",")))
X = np.array(list(csv.reader(open("rym.csv", "r"), delimiter=","))).astype("float")[:1075,:]
y = arr[:,1].astype("int")[:X.shape[0]]

def normX(X):
    X_0 = ((X[:,0] - np.min(X[:,0]))/math.sqrt(np.var(X[:,0])))[:,None]
    X_1 = ((X[:,1] - np.min(X[:,1]))/math.sqrt(np.var(X[:,1])))[:,None]
    X_dist = X[:,2:] / X[:,1][:,None]
    return np.concatenate((X_0,X_1,X_dist), 1)

model = keras.models.load_model('./model3')

ex0 = [4.4, 3000]
ex1 = [4.5, 2503]
ex2 = []

for ex in exs:
    ex = [(ex[0] - np.min(X[:,0]))/math.sqrt(np.var(X[:,0])),(ex[1] - np.min(X[:,1]))/math.sqrt(np.var(X[:,1]))] + [e / ex[1] for e in ex[2:]] 
    print((model.predict(np.transpose(np.array(ex)[:,None])).flatten("F") * -math.sqrt(np.var(y)) + np.mean(y)).astype("int"))



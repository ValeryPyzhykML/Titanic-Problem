from os import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import re

from numpy.core.fromnumeric import size
from numpy.core.numeric import NaN

def sigmoid(z):
    return 1/(1+math.exp(-z))

vectorized_sigmoid = np.vectorize(sigmoid)

def cost(h,y):
    return (y.dot(np.log(h)) 
        + (np.ones(y.size)-y).dot(np.log(1-h))) *(-1) /len(y)

def loss_funtion_j(w, data, y):   
    if (len(data.columns) != len(w)):
        print("data.columns %i" % len(data.columns))
        print("w %i" % len(w))
        raise RuntimeError from None
    h = vectorized_sigmoid(data.to_numpy().dot(w))
    j = cost(h,y)
    return j

def gradient(w, data, y): 
    h = vectorized_sigmoid(data.to_numpy().dot(w))
    return data.transpose().dot(h-y) / len(y)


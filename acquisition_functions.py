import math
import numpy as np
import os
import random
from scipy.stats import mode

def bald(X_Pool_Dropout, num_classes, model, batch_size=32, dropout_iterations=10):

    print (X_Pool_Dropout[0].shape)
    score_All = np.zeros(shape=(X_Pool_Dropout[0].shape[0], num_classes))
    All_Entropy_Dropout = np.zeros(shape=X_Pool_Dropout[0].shape[0])

    for d in range(dropout_iterations):
        dropout_score = model.predict(X_Pool_Dropout, batch_size=batch_size, verbose=1)
        #computing Entropy_Average_Pi
        score_All += dropout_score
        #computing Average_Entropy
        dropout_score_log = np.log2(dropout_score)
        Entropy_Compute = - np.multiply(dropout_score, dropout_score_log)
        Entropy_Per_Dropout = np.sum(Entropy_Compute, axis=1)
        All_Entropy_Dropout += Entropy_Per_Dropout 


    Avg_Pi = np.divide(score_All, dropout_iterations)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

    G_X = Entropy_Average_Pi
    Average_Entropy = np.divide(All_Entropy_Dropout, dropout_iterations)
    uncertain_pool_points = Entropy_Average_Pi - Average_Entropy

    return uncertain_pool_points


def maxentropy(X_Pool_Dropout, num_classes, model, batch_size=32, dropout_iterations=10):

    print (X_Pool_Dropout[0].shape)
    score_All = np.zeros(shape=(X_Pool_Dropout[0].shape[0], num_classes))
    for d in range(dropout_iterations):
        dropout_score = model.predict(X_Pool_Dropout, batch_size=batch_size, verbose=1)
        score_All += dropout_score

    Avg_Pi = np.divide(score_All, dropout_iterations)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

    uncertain_pool_points = Entropy_Average_Pi

    return uncertain_pool_points

def varratio(X_Pool_Dropout, num_classes, model, batch_size=32, dropout_iterations=10):
    All_Dropout_Classes = np.zeros(shape=(X_Pool_Dropout[0].shape[0],1))

    for d in range(dropout_iterations):
        y_prob = model.predict(X_Pool_Dropout, batch_size=batch_size, verbose=1) 
        dropout_classes = y_prob.argmax(axis=-1)
        #dropout_classes = model.predict_classes(X_Pool_Dropout, batch_size=batch_size, verbose=1)
        dropout_classes = np.array([dropout_classes]).T
        All_Dropout_Classes = np.append(All_Dropout_Classes, dropout_classes, axis=1)

    uncertain_pool_points = np.zeros(shape=(X_Pool_Dropout[0].shape[0]))

    for t in range(X_Pool_Dropout[0].shape[0]):
        L = np.array([0])
        for d_iter in range(dropout_iterations):
            L = np.append(L, All_Dropout_Classes[t, d_iter+1])            
        Predicted_Class, Mode = mode(L[1:])
        v = np.array(  [1 - Mode/float(dropout_iterations)])
        uncertain_pool_points[t] = v

    return uncertain_pool_points

def random_acq(X_Pool_Dropout, num_classes, model, batch_size=32, dropout_iterations=10):
    #just assign everything a random value
    uncertain_pool_points = np.random.random(size=X_Pool_Dropout[0].shape[0])
    return uncertain_pool_points

import numpy as np
import random
from math import isnan
from sklearn import preprocessing
from numpy import linalg as LA


def tensor_fectorization(ratings, U, V, C, tensor, steps = 1000, alpha = 0.0002, beta = 0.0005):

    for step in range(steps):
        for rating in ratings:
            u_index = int(rating[0])
            r_index = int(rating[1])
            s_index = rating[2]
            r = rating[4]
            sum_tensor = 0
            for t in range(len(tensor)):
                sum_tensor += tensor[t] * U[u_index,t] * V[r_index,t] * C[s_index,t]
            #real value-predicted value
            eijk = r - sum_tensor
            for t in range(len(tensor)):
                U[u_index][t] = U[u_index][t] + alpha * (eijk * tensor[t] * V[r_index][t] * C[s_index][t] - beta * U[u_index][t])
                V[r_index][t] = V[r_index][t] + alpha * (eijk * tensor[t] * U[u_index][t] * C[s_index][t] - beta * V[r_index][t])
                C[s_index][t] = C[s_index][t] + alpha * (eijk * tensor[t] * U[u_index][t] * V[r_index][t] - beta * C[s_index][t])
                tensor[t] = tensor[t] + alpha * (eijk * U[u_index][t] * V[r_index][t] * C[s_index][t] - beta * tensor[t])

        e = 0
        for rating_ in ratings:
            u_index_ = int(rating_[0])
            r_index_ = int(rating_[1])
            s_index_ = rating_[2]
            r_ = rating_[4]
            sum_tensor_ = 0
            for t in range(len(tensor)):
                sum_tensor_ += tensor[t] * U[u_index_,t] * V[r_index_,t] * C[s_index_,t]
            e = e + (1 / 2) * pow((r_ - sum_tensor_), 2)
        e = e + (beta / 2) * (LA.norm(U) / len(U) + LA.norm(V) / len(V) + LA.norm(C) / len(C) + LA.norm(tensor) / len(tensor))

        print(step)
        print(e)

        if e < 0.01:
            print(step)
            print(e)
            break

    return U, V, C, tensor

def prediction_matrix(U, V, C, tensor):
    X_prediction = np.zeros([len(C), len(U), len(V)])

    for i in range(len(U)):
        for j in range(len(V)):
            for k in range(len(C)):
                sum_tensor = 0
                for t in range(len(tensor)):
                    sum_tensor += tensor[t] * U[i, t] * V[j, t] * C[k, t]
                X_prediction[k][i][j] = sum_tensor

    return X_prediction






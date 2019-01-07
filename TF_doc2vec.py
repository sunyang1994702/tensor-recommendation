import numpy as np
import random
from math import isnan
from sklearn import preprocessing
from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler



def value_transform(vector):
    scale = MinMaxScaler(feature_range=(0, 1))
    vector = np.array(vector, dtype='float64')
    re_vector = np.reshape(vector, (-1,1))
    trans_vector = scale.fit_transform(re_vector)

    return trans_vector.flatten()



def tensor_fectorization(ratings, U, V, C, tensor, model, steps=1000, alpha=0.0002, beta=0.0005):
    error_list = []
    for step in range(steps):
        for rating in ratings:
            u_index = int(rating[0])
            r_index = int(rating[1])
            s_index = int(rating[2])
            re_index = int(rating[3])
            r = rating[4]
            sum_tensor = 0
            for t in range(len(tensor)):
                sum_tensor += tensor[t] * U[u_index, t] * V[r_index, t] * C[s_index, t]
            # 真实值-预测值
            eijk = r - sum_tensor
            #reset the value in model from 0 to 1
            vector = value_transform(model[re_index])

            for t in range(len(tensor)):
                U[u_index][t] = U[u_index][t] + alpha * vector[t] * (
                            eijk * tensor[t] * V[r_index][t] * C[s_index][t] - beta * U[u_index][t])
                V[r_index][t] = V[r_index][t] + alpha * vector[t] * (
                            eijk * tensor[t] * U[u_index][t] * C[s_index][t] - beta * V[r_index][t])
                C[s_index][t] = C[s_index][t] + alpha * vector[t] * (
                            eijk * tensor[t] * U[u_index][t] * V[r_index][t] - beta * C[s_index][t])
        e = 0
        for rating_ in ratings:
            u_index_ = int(rating_[0])
            r_index_ = int(rating_[1])
            s_index_ = rating_[2]
            r_ = rating_[4]
            sum_tensor_ = 0
            for t in range(len(tensor)):
                sum_tensor_ += tensor[t] * U[u_index_, t] * V[r_index_, t] * C[s_index_, t]
            e = e + (1 / 2) * pow((r_ - sum_tensor_), 2)
        e = e + (beta / 2) * (
                    LA.norm(U) / len(U) + LA.norm(V) / len(V) + LA.norm(C) / len(C) + LA.norm(tensor) / len(tensor))
        error_list.append(e)
        print(step)
        print(e)

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


if __name__ == '__main__':
    K = 30
    file = "../file_package/RichmondHill_filtered.txt"
    file_LDA = "LDA_Topic_File/RichmondHill_doc_topics_" + str(K) + ".txt"
    rating_list, user_index, restaurant_index, season_index, reveiw_index = reconstruct_rating(file)
    print(rating_list)

    """
    X = [[[0, 0, 1, 0], [4, 0, 0, 0], [3, 0, 0, 0], [0, 4, 0, 0], [0, 0, 3, 1]],
         [[0, 0, 1, 0], [0, 2, 0, 0], [3, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
         [[1, 0, 2, 0], [0, 5, 4, 0], [0, 0, 0, 4], [0, 0, 0, 0], [2, 2, 0, 0]]]
    X = np.array(X)

    print(X)

    tensor = np.random.normal(scale=1/K, size=K)
    #size of U 171*K
    U = np.random.normal(scale=1/K, size=(len(user_index), K))
    #size of V 122*K
    V = np.random.normal(scale=1/K, size=(len(restaurant_index), K))
    #size of C 4*K
    C = np.random.normal(scale=1/K, size=(len(season_index), K))


    tensor = [0.001] * K
    U = [[0.001] * K for i in range(len(user_index))]
    U = np.array(U)
    V = [[0.001] * K for i in range(len(restaurant_index))]
    V = np.array(V)
    C = [[0.001] * K for i in range(len(season_index))]
    C = np.array(C)


    tensor = [random.uniform(0.1, 0.9) for rand in range(K)]
    U = np.random.rand(len(user_index), K)
    V = np.random.rand(len(restaurant_index), K)
    C = np.random.rand(len(season_index), K)


    U, V, C, tensor = tensor_fectorization(rating_list, U, V, C, tensor, doc_topics)

    X_prediction = prediction_matrix(U, V, C, tensor)

    print(X_prediction)
    """




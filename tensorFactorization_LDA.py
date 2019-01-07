import numpy as np
import random
from math import isnan
from sklearn import preprocessing
from numpy import linalg as LA

"""
包含正则项
anothor tpye of tensorFactorization
when k = 5
the result of tensor factorization: 
RMSE: [1.5838600872203294, 1.630691334896505, 1.6191176115025332, 1.6112139791558677, 1.6654770941432084] 
MAE: [1.2417945122133498, 1.262390772763441, 1.2572145933938197, 1.2571448030832106, 1.2941910373953611]
the average of RMSE :1.6220720213836888
the average of MAE :1.2625471437698363

when k = 10
the result of tensor factorization: 
RMSE: [1.7605347397193707, 1.678113131652284, 1.7025429329356832, 1.7085920904145468, 1.704786424934104] 
MAE: [1.3912222942339074, 1.3307104860514618, 1.3504077297946209, 1.357174364410061, 1.3347800216794095]
the average of RMSE :1.7109138639311976
the average of MAE :1.352858979233892
when K = 20
the result of tensor factorization: 
RMSE: [1.803321237628651, 1.8259566035779777, 1.8270254330871902, 1.8467445517989483, 1.8631356415637774] 
MAE: [1.4324283327501237, 1.4303330139734007, 1.4459860201281303, 1.4770538963284676, 1.4654358274079113]
the average of RMSE :1.833236693531309
the average of MAE :1.4502474181176068

"""


def reconstruct_rating(file):
    myfile = open(file, 'r')
    user_index = {}
    restaurant_index = {}
    season_index = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3}
    reveiw_index = {}
    u_index = 0
    r_index = 0
    re_index = 0
    rating_list = []
    for line in myfile.readlines():
        line_array = line.strip().split(',')
        user = str(line_array[0])
        restaurant = str(line_array[1])
        rating = int(line_array[2])
        season = int(line_array[3])
        review = str(line_array[4])
        if user not in user_index.keys():
            user_index[user] = u_index
            u_index += 1

        if restaurant not in restaurant_index.keys():
            restaurant_index[restaurant] = r_index
            r_index += 1

        if review not in reveiw_index.keys():
            reveiw_index[review] = re_index
            re_index += 1
        ## the format of rating list:  (u_index, r_index, season, re_index, rating)
        rating_list.append((user_index[user], restaurant_index[restaurant], season, reveiw_index[review], rating))

    return rating_list, user_index, restaurant_index, season_index, reveiw_index

def construct_tensor(file):
    myfile = open(file, 'r')
    doc_topics_dic = {}
    for line_index, line in enumerate(myfile.readlines()):
        each_doc_topics = []
        line = line.rstrip('\n')
        line = line.rstrip(',')
        line_array = line.split(',')
        for topic in line_array:
            each_doc_topics.append(float(topic))
        doc_topics_dic[line_index] = each_doc_topics

    return doc_topics_dic

def tensor_fectorization(ratings, U, V, C, tensor, doc_topics, steps=1000, alpha=0.0002, beta=0.0005):
    error_list = []
    for step in range(steps):
        for rating in ratings:
            u_index = int(rating[0])
            r_index = int(rating[1])
            s_index = rating[2]
            re_index = rating[3]
            r = rating[4]
            sum_tensor = 0
            for t in range(len(tensor)):
                sum_tensor += tensor[t] * U[u_index, t] * V[r_index, t] * C[s_index, t]
            # 真实值-预测值
            eijk = r - sum_tensor

            for t in range(len(tensor)):
                U[u_index][t] = U[u_index][t] + alpha * doc_topics[re_index][t] * (
                            eijk * tensor[t] * V[r_index][t] * C[s_index][t] - beta * U[u_index][t])
                V[r_index][t] = V[r_index][t] + alpha * doc_topics[re_index][t] * (
                            eijk * tensor[t] * U[u_index][t] * C[s_index][t] - beta * V[r_index][t])
                C[s_index][t] = C[s_index][t] + alpha * doc_topics[re_index][t] * (
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
    file = "file_package/RichmondHill_filtered.txt"
    file_LDA = "LDA_Topic_File/RichmondHill_doc_topics_" + str(K) + ".txt"
    rating_list, user_index, restaurant_index, season_index, reveiw_index = reconstruct_rating(file)
    print(reveiw_index)

    doc_topics = construct_tensor(file_LDA)
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
    """

    tensor = [random.uniform(0.1, 0.9) for rand in range(K)]
    U = np.random.rand(len(user_index), K)
    V = np.random.rand(len(restaurant_index), K)
    C = np.random.rand(len(season_index), K)


    U, V, C, tensor = tensor_fectorization(rating_list, U, V, C, tensor, doc_topics)

    X_prediction = prediction_matrix(U, V, C, tensor)

    print(X_prediction)





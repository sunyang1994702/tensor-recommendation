import numpy as np
from sklearn.model_selection import train_test_split
import random
import TF, TF_LDA, TF_doc2vec
import doc_topic, doc2vec_model


def calculate_MAE_RMSE(prediction_TF, prediction_TFLDA, rediction_TFD2V, test):
    sum_rmseTF = 0
    sum_maeTF = 0
    sum_rmseTFLDA = 0
    sum_maeTFLDA = 0
    sum_rmseTFD2V = 0
    sum_maeTFD2V = 0

    TF_ = []
    TF_LDA = []
    TF_D2V = []

    count_test = len(test)
    for each_test in test:
        user = int(each_test[0])
        restaurant = int(each_test[1])
        season = int(each_test[2])
        real_rating = float(each_test[4])

        rating_predict_TF = prediction_TF[season][user][restaurant]
        rating_predict_TFLDA = prediction_TFLDA[season][user][restaurant]
        rating_predict_TFD2V = prediction_TFD2V[season][user][restaurant]


        sum_rmseTF += np.square(real_rating - rating_predict_TF)
        sum_rmseTFLDA += np.square(real_rating - rating_predict_TFLDA)
        sum_rmseTFD2V += np.square(real_rating - rating_predict_TFD2V)
        sum_maeTF += np.abs(real_rating - rating_predict_TF)
        sum_maeTFLDA += np.abs(real_rating - rating_predict_TFLDA)
        sum_maeTFD2V += np.abs(real_rating - rating_predict_TFD2V)



    TF_.append((np.sqrt(sum_rmseTF / count_test), sum_maeTF / count_test))
    TF_LDA.append((np.sqrt(sum_rmseTFLDA / count_test), sum_maeTFLDA / count_test))
    TF_D2V.append((np.sqrt(sum_rmseTFD2V / count_test), sum_maeTFD2V / count_test))

    return TF_, TF_LDA, TF_D2V


file = "file_package/RichmondHill_filtered.txt"
rating_list, user_index, restaurant_index, season_index, reveiw_index = rating_construct.reconstruct_rating(file)
K_list = [5,10,15,20,25,30,35,40,45,50,55,60]
myfile = open("expriment_all(RichmondHill).txt", "w+")


for K in K_list:
    myfile.write("when the K is " + str(K) + ':' + '\n')
    RMSE_TF = []
    MAE_TF = []
    RMSE_TFLDA = []
    MAE_TFLDA = []
    RMSE_TFD2V = []
    MAE_TFD2V = []

    doc_topics = doc_topic.doc_topic_distribution(rating_list, K)
    model = doc2vec_model.get_doc2vec_model(rating_list, K)

    for i in range(5):
        train, test = train_test_split(rating_list, test_size=0.2, random_state=None)
        #TF
        tensor_TF = [random.uniform(0.1, 0.9) for tf in range(K)]
        U_TF = np.random.rand(len(user_index), K)
        V_TF = np.random.rand(len(restaurant_index), K)
        C_TF = np.random.rand(len(season_index), K)
        TF_U, TF_V, TF_C, TF_tensor = TF.tensor_fectorization(train, U_TF, V_TF, C_TF, tensor_TF)
        prediction_TF = TF.prediction_matrix(TF_U, TF_V, TF_C, TF_tensor)
        print("the time of training for TF method is :{}".format(i))
        print(prediction_TF)
        
        #TF_LDA
        tensor_TFLDA = [random.uniform(0.1, 0.9) for tf_lda in range(K)]
        U_TFLDA = np.random.rand(len(user_index), K)
        V_TFLDA = np.random.rand(len(restaurant_index), K)
        C_TFLDA = np.random.rand(len(season_index), K)
        TF_LDA_U, TF_LDA_V, TF_LDA_C, TF_LDA_tensor = TF_LDA.tensor_fectorization(train, U_TFLDA, V_TFLDA, C_TFLDA, tensor_TFLDA, doc_topics)
        prediction_TFLDA = TF_LDA.prediction_matrix(TF_LDA_U, TF_LDA_V, TF_LDA_C, TF_LDA_tensor)
        print("the time of training for TF_LDA method is :{}".format(i))
        print(prediction_TFLDA)


        # TF_D2V
        tensor_TFD2V = [random.uniform(0.1, 0.9) for tf_d2v in range(K)]
        U_TFD2V = np.random.rand(len(user_index), K)
        V_TFD2V = np.random.rand(len(restaurant_index), K)
        C_TFD2V = np.random.rand(len(season_index), K)
        TFD2V_U, TFD2V_V, TFD2V_C, TFD2V_tensor = TF_doc2vec.tensor_fectorization(train, U_TFD2V, V_TFD2V, C_TFD2V, tensor_TFD2V, model)
        prediction_TFD2V = TF_doc2vec.prediction_matrix(TFD2V_U, TFD2V_V, TFD2V_C, TFD2V_tensor)
        print("the time of training for TF method is :{}".format(i))
        print(prediction_TFD2V)

        #calculate MAE and RMSE
        TF_, TF_LDA, TF_D2V = calculate_MAE_RMSE(prediction_TF, prediction_TFLDA,
                               prediction_TFD2V, test)


        RMSE_TF.append(TF_[0][0])
        MAE_TF.append(TF_[0][1])
        RMSE_TFLDA.append(TF_LDA[0][0])
        MAE_TFLDA.append(TF_LDA[0][1])
        RMSE_TFD2V.append(TF_D2V[0][0])
        MAE_TFD2V.append(TF_D2V[0][1])

    myfile.write("the RMSE for TF is " + str(RMSE_TF) + '\n' + "the average :" + str(np.average(RMSE_TF)) + '\n')
    myfile.write("the RMSE for TFLDA is " + str(RMSE_TFLDA) + '\n' + "the average :" + str(np.average(RMSE_TFLDA)) + '\n')
    myfile.write("the RMSE for TFD2V is " + str(RMSE_TFD2V) + '\n' + "the average :" + str(np.average(RMSE_TFD2V)) + '\n')
    myfile.write("the MAE for TF is " + str(MAE_TF) + '\n' + "the average :" + str(np.average(MAE_TF)) + '\n')
    myfile.write("the MAE for TFLDA is " + str(MAE_TFLDA) + '\n' + "the average :" + str(np.average(MAE_TFLDA)) + '\n')
    myfile.write("the MAE for TFD2V is " + str(MAE_TFD2V) + '\n' + "the average :" + str(np.average(MAE_TFD2V)) + '\n')

    myfile.write("\n")
    myfile.write("\n")

myfile.close()



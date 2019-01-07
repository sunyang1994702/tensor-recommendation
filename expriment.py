from restaurantResSys.MF_and_TF import matrix_factorization, tensorFactorization1
from restaurantResSys.MF_LDA_and_TF_LDA import doc_topic, matrixFactorization_LDA, tensorFactorization_LDA
from restaurantResSys.MF_d2v_and_TF_d2v import doc2vec_model, MF_doc2vec, TF_doc2vec
from restaurantResSys import rating_construct
import numpy as np
from sklearn.model_selection import train_test_split
import random




file = "file_package/RichmondHill_filtered.txt"
rating_list, user_index, restaurant_index, season_index, reveiw_index = rating_construct.reconstruct_rating(file)


def calculate_MAE_RMSE(prediction_MF, prediction_TF, prediction_MFLDA, prediction_TFLDA, prediction_MFD2V, prediction_TFD2V, test):
    sum_rmseMF = 0
    sum_maeMF = 0
    sum_rmseTF = 0
    sum_maeTF = 0
    sum_rmseMFLDA = 0
    sum_maeMFLDA = 0
    sum_rmseTFLDA = 0
    sum_maeTFLDA = 0
    sum_rmseMFD2V = 0
    sum_maeMFD2V = 0
    sum_rmseTFD2V = 0
    sum_maeTFD2V = 0

    MF_ = []
    TF_ = []
    MF_LDA = []
    TF_LDA = []
    MF_D2V = []
    TF_D2V = []

    count_test = len(test)
    for each_test in test:
        user = int(each_test[0])
        restaurant = int(each_test[1])
        season = int(each_test[2])
        real_rating = float(each_test[4])

        rating_predict_MF = prediction_MF[user][restaurant]
        rating_predict_TF = prediction_TF[season][user][restaurant]
        rating_predict_MFLDA = prediction_MFLDA[user][restaurant]
        rating_predict_TFLDA = prediction_TFLDA[season][user][restaurant]
        rating_predict_MFD2V = prediction_MFD2V[user][restaurant]
        rating_predict_TFD2V = prediction_TFD2V[season][user][restaurant]

        sum_rmseMF += np.square(real_rating - rating_predict_MF)
        sum_rmseTF += np.square(real_rating - rating_predict_TF)
        sum_rmseMFLDA += np.square(real_rating - rating_predict_MFLDA)
        sum_rmseTFLDA += np.square(real_rating - rating_predict_TFLDA)
        sum_rmseMFD2V += np.square(real_rating - rating_predict_MFD2V)
        sum_rmseTFD2V += np.square(real_rating - rating_predict_TFD2V)
        sum_maeMF += np.abs(real_rating - rating_predict_MF)
        sum_maeTF += np.abs(real_rating - rating_predict_TF)
        sum_maeMFLDA += np.abs(real_rating - rating_predict_MFLDA)
        sum_maeTFLDA += np.abs(real_rating - rating_predict_TFLDA)
        sum_maeMFD2V += np.abs(real_rating - rating_predict_MFD2V)
        sum_maeTFD2V += np.abs(real_rating - rating_predict_TFD2V)


    MF_.append((np.sqrt(sum_rmseMF / count_test), sum_maeMF / count_test))
    TF_.append((np.sqrt(sum_rmseTF / count_test), sum_maeTF / count_test))
    MF_LDA.append((np.sqrt(sum_rmseMFLDA / count_test), sum_maeMFLDA / count_test))
    TF_LDA.append((np.sqrt(sum_rmseTFLDA / count_test), sum_maeTFLDA / count_test))
    MF_D2V.append((np.sqrt(sum_rmseMFD2V / count_test), sum_maeMFD2V / count_test))
    TF_D2V.append((np.sqrt(sum_rmseTFD2V / count_test), sum_maeTFD2V / count_test))

    return MF_, TF_, MF_LDA, TF_LDA, MF_D2V, TF_D2V

K_list = [5,10,15,20,25,30,35,40,45,50,55,60]
myfile = open("expriment_all(RichmondHill).txt", "w+")


for K in K_list:
    myfile.write("when the K is " + str(K) + ':' + '\n')
    RMSE_MF = []
    MAE_MF =[]
    RMSE_TF = []
    MAE_TF = []
    RMSE_MFLDA = []
    MAE_MFLDA = []
    RMSE_TFLDA = []
    MAE_TFLDA = []
    RMSE_MFD2V = []
    MAE_MFD2V = []
    RMSE_TFD2V = []
    MAE_TFD2V = []

    doc_topics = doc_topic.doc_topic_distribution(rating_list, K)
    model = doc2vec_model.get_doc2vec_model(rating_list, K)

    for i in range(5):
        train, test = train_test_split(rating_list, test_size=0.2, random_state=None)
        #MF
        U_MF = np.random.rand(len(user_index), K)
        V_MF = np.random.rand(len(restaurant_index), K)

        nP, nQ = matrix_factorization.matrixFactorization(train, U_MF, V_MF)

        prediction_MF = np.dot(nP, nQ)

        print("the time of training for MF method is :{}".format(i))
        print(prediction_MF)

        #TF
        tensor_TF = [random.uniform(0.1, 0.9) for tf in range(K)]
        U_TF = np.random.rand(len(user_index), K)
        V_TF = np.random.rand(len(restaurant_index), K)
        C_TF = np.random.rand(len(season_index), K)
        TF_U, TF_V, TF_C, TF_tensor = tensorFactorization1.tensor_fectorization(train, U_TF, V_TF, C_TF, tensor_TF)
        prediction_TF = tensorFactorization1.prediction_matrix(TF_U, TF_V, TF_C, TF_tensor)
        print("the time of training for TF method is :{}".format(i))
        print(prediction_TF)

        #MF_LDA
        U_MFLDA = np.random.rand(len(user_index), K)
        V_MFLDA = np.random.rand(len(restaurant_index), K)


        ave_rating, u_ave, r_ave = matrixFactorization_LDA.calculate_ave_ratings(train)
        predict_U, predict_V = matrixFactorization_LDA.matrixFactorization(train, U_MFLDA, V_MFLDA, K, doc_topics)
        prediction_MFLDA = matrixFactorization_LDA.get_predicted_matrix(ave_rating, u_ave, r_ave, predict_U, predict_V)

        print("the time of training for MF_LDA method is :{}".format(i))
        print(prediction_MFLDA)

        #TF_LDA
        tensor_TFLDA = [random.uniform(0.1, 0.9) for tf_lda in range(K)]
        U_TFLDA = np.random.rand(len(user_index), K)
        V_TFLDA = np.random.rand(len(restaurant_index), K)
        C_TFLDA = np.random.rand(len(season_index), K)
        TF_LDA_U, TF_LDA_V, TF_LDA_C, TF_LDA_tensor = tensorFactorization_LDA.tensor_fectorization(train, U_TFLDA, V_TFLDA, C_TFLDA, tensor_TFLDA, doc_topics)
        prediction_TFLDA = tensorFactorization_LDA.prediction_matrix(TF_LDA_U, TF_LDA_V, TF_LDA_C, TF_LDA_tensor)
        print("the time of training for TF_LDA method is :{}".format(i))
        print(prediction_TFLDA)

        # MF_D2V
        U_MFD2V = np.random.rand(len(user_index), K)
        V_MFD2V = np.random.rand(len(restaurant_index), K)

        nP_D2V, nQ_D2V = MF_doc2vec.matrixFactorization(train, U_MFD2V, V_MFD2V, K, model)
        prediction_MFD2V = MF_doc2vec.get_predicted_matrix(ave_rating, u_ave, r_ave, nP_D2V, nQ_D2V)
        print("the time of training for MF method is :{}".format(i))
        print(prediction_MFD2V)

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
        MF_, TF_, MF_LDA, TF_LDA, MF_D2V, TF_D2V = calculate_MAE_RMSE(prediction_MF, prediction_TF, prediction_MFLDA, prediction_TFLDA, prediction_MFD2V,
                               prediction_TFD2V, test)

        RMSE_MF.append(MF_[0][0])
        MAE_MF.append(MF_[0][1])
        RMSE_TF.append(TF_[0][0])
        MAE_TF.append(TF_[0][1])
        RMSE_MFLDA.append(MF_LDA[0][0])
        MAE_MFLDA.append(MF_LDA[0][1])
        RMSE_TFLDA.append(TF_LDA[0][0])
        MAE_TFLDA.append(TF_LDA[0][1])
        RMSE_MFD2V.append(MF_D2V[0][0])
        MAE_MFD2V.append(MF_D2V[0][1])
        RMSE_TFD2V.append(TF_D2V[0][0])
        MAE_TFD2V.append(TF_D2V[0][1])

    myfile.write("the RMSE for MF is " + str(RMSE_MF) + '\n' + "the average :" + str(np.average(RMSE_MF)) + '\n')
    myfile.write("the RMSE for TF is " + str(RMSE_TF) + '\n' + "the average :" + str(np.average(RMSE_TF)) + '\n')
    myfile.write("the RMSE for MFLDA is " + str(RMSE_MFLDA) + '\n' + "the average :" + str(np.average(RMSE_MFLDA)) + '\n')
    myfile.write("the RMSE for TFLDA is " + str(RMSE_TFLDA) + '\n' + "the average :" + str(np.average(RMSE_TFLDA)) + '\n')
    myfile.write("the RMSE for MFD2V is " + str(RMSE_MFD2V) + '\n' + "the average :" + str(np.average(RMSE_MFD2V)) + '\n')
    myfile.write("the RMSE for TFD2V is " + str(RMSE_TFD2V) + '\n' + "the average :" + str(np.average(RMSE_TFD2V)) + '\n')
    myfile.write("the MAE for MF is " + str(MAE_MF) + '\n' + "the average :" + str(np.average(MAE_MF)) + '\n')
    myfile.write("the MAE for TF is " + str(MAE_TF) + '\n' + "the average :" + str(np.average(MAE_TF)) + '\n')
    myfile.write("the MAE for MFLDA is " + str(MAE_MFLDA) + '\n' + "the average :" + str(np.average(MAE_MFLDA)) + '\n')
    myfile.write("the MAE for TFLDA is " + str(MAE_TFLDA) + '\n' + "the average :" + str(np.average(MAE_TFLDA)) + '\n')
    myfile.write("the MAE for MFD2V is " + str(MAE_MFD2V) + '\n' + "the average :" + str(np.average(MAE_MFD2V)) + '\n')
    myfile.write("the MAE for TFD2V is " + str(MAE_TFD2V) + '\n' + "the average :" + str(np.average(MAE_TFD2V)) + '\n')

    myfile.write("\n")
    myfile.write("\n")

myfile.close()

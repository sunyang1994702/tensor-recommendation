# tensor-recommendation
Using tensor factorization to improve user's rating for item

My project was using tensor factorization method to improve ratings of user for item. This method gets the inspiration from this paper<Rating Prediction with Topic Gradient Descent Method for Matrix Factorization in Recommendation>. 
On the basis of Matrix Factorization, I tried to use Tensor Factorization to achive rating prediction and improve the accuracy according to the index of RMSE and MAE.

This project was based on Python3 with using Yelp dataset:https://www.yelp.com/dataset/challenge

First of all, 
  using the normalization.py file to proccess the user's reviews. skipping the punctuation, stopwords and some useless words. and output in the form of arrays.

Secondlly,
  implementing TF, TF_LDA and TF_Doc2Vec method .
  
finally, 
  compare the result about this three method. comparing the most basic TF method. the tensor that used LDA and Doc2Vec improved significantly. it can be improved by 5-8% according to the index of RMSE and MAE



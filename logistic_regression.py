#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pickle 
from minst_data_set import minst_data_set
from sklearn.linear_model import LogisticRegression

def get_normalizers(data):

    mean = np.mean(data,axis =0).reshape((1,-1))
    std = np.std(data,axis =0).reshape((1,-1))
    return mean,std

def evaluate_predictions(y, y_predicted):
    total   = y.shape[0]
    correct = np.flatnonzero(np.array([y_i == predict_i for y_i,predict_i in zip(y,y_predicted)])).shape[0]
    return correct/total

if __name__ == '__main__':

    #Loading test and training data
    print("Loading data ....")
    with open('./saved_models/training.pickle', 'rb') as f:
        train_data = pickle.load(f)

    with open('./saved_models/test.pickle', 'rb') as f:
        test_data = pickle.load(f)

    x_test = test_data.x
    y_test = test_data.y
    x_train = train_data.x
    y_train = train_data.y

    #Normalize data
    x_train[-1,:] = x_train[-1,:] +np.array([0.0001])
    mean,std = get_normalizers(x_train)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) /std

    #Training
    logisticRegr = LogisticRegression(solver = 'lbfgs')
    print("training classifier ....")
    logisticRegr.fit(x_train, y_train)

    print("making predictions ....")
    predictions = logisticRegr.predict(x_test)

    #Saving classifier
    with open('./saved_models/classifier.pickle', 'wb') as f:
        pickle.dump(logisticRegr, f)

    #acc = evaluate_predictions(y_test,predictions)
    score_test = logisticRegr.score(x_test, y_test)
    score_train = logisticRegr.score(x_train, y_train)
    print("Classifier accuracy in training set: {:4.4f}".format(score_train))
    print("Classifier accuracy in test set: {:4.4f}".format(score_test))

    


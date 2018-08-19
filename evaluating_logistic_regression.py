#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pickle 
from minst_data_set import minst_data_set
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def evaluate_predictions(y, y_predicted):
    total   = y.shape[0]
    correct = np.flatnonzero(np.array([y_i == predict_i for y_i,predict_i in zip(y,y_predicted)])).shape[0]
    return correct/total

def get_normalizers(data):

    mean = np.mean(data,axis =0).reshape((1,-1))
    std = np.std(data,axis =0).reshape((1,-1))
    return mean,std

if __name__ == '__main__':

    #Loading test and training data
    with open('./saved_models/classifier.pickle', 'rb') as f:
        classifier = pickle.load(f)

    with open('./saved_models/training.pickle', 'rb') as f:
        train_data = pickle.load(f)

    with open('./saved_models/test.pickle', 'rb') as f:
        test_data = pickle.load(f)

    x_test = test_data.x
    y_test = test_data.y
    x_train = train_data.x
    y_train = train_data.y

    #Normalize data
    original_test = x_test
    #Trick to avoid zero standard deviation
    x_train[-1,:] = x_train[-1,:] +np.array([0.0001])
    mean,std = get_normalizers(x_train)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) /std

    #Predictions
    train_predictions = classifier.predict(x_train)
    test_predictions = classifier.predict(x_test)

    #Evaluate
    train_acc = evaluate_predictions(y_train,train_predictions)
    test_acc = evaluate_predictions(y_test,test_predictions)
    print("Classifier accuracy in training set: {:4.4f}".format(train_acc))
    print("Classifier accuracy in test set: {:4.4f}".format(test_acc))



    #Show some incorrect predictions
    misclassifiedIndexes = []
    for index,label, predict in zip(range(y_test.shape[0]),y_test, test_predictions):
        if label != predict: 
            misclassifiedIndexes.append(index)
            
    fig,axes = plt.subplots(1,5,figsize=(20,4))
    for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
        axes[plotIndex].imshow(np.reshape(original_test[badIndex], (28,28)), cmap=plt.cm.gray_r)
        axes[plotIndex].set_title("Predicted: {}, Actual: {}".format(test_predictions[badIndex], y_test[badIndex]), fontsize = 15)

    #Show confusion Matrix taken from 
    #https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
    cm = metrics.confusion_matrix(y_test, test_predictions)
    
    
    plt.figure(figsize=(9,9))
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size = 15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
    plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
    plt.ylabel('Actual label', size = 15)
    plt.xlabel('Predicted label', size = 15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                  horizontalalignment='center',
                  verticalalignment='center')
            
            
    


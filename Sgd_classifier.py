import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from HelperFunctions import ReadFromFolder, getBOWAndBER
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

if __name__== '__main__':
    path = sys.argv[1]
    if len(sys.argv) == 3:
        seedVal = int(sys.argv[2])
    else:
        seedVal = np.random.randint(low = 0, high = 1000)
    
    trainData = ReadFromFolder(path, test = False)
    testData = ReadFromFolder(path,test = True)

    np.random.seed(seedVal)
    np.random.shuffle(trainData)
    np.random.shuffle(testData)

    vectorizer = CountVectorizer()
    trainingData = vectorizer.fit_transform(trainData[:,0])
    trainingData = trainingData.toarray()
    testingData = vectorizer.transform(testData[:,0]).toarray()
    trainLabel = np.where(trainData[:,1]=='ham',1,0)
    clf = GridSearchCV(estimator=SGDClassifier(), param_grid= {'alpha': [0.001,0.01, 0.1, 1, 10], 'loss':["hinge", "log_loss"], 'max_iter':[100,200,500,800,1000],'penalty': ['l1', 'l2']}, n_jobs=-1)
    clf.fit(trainingData,trainLabel)
    print("Ideal parameters are: {}".format(clf.best_params_))

    predY = clf.predict(testingData)
    trueY = np.where(testData[:,1] =='ham',1,0)


    ##TODO 
    #Insert accuraccy and precision stuff
    accuracy = accuracy_score(trueY, predY)
    precision, recall, f1, support = precision_recall_fscore_support(trueY, predY)
    print('Algorithm: Multinomial Linear Regression: SGD Classifier')
    print('\tAccuracy:\t{}\n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1 Score:\t{}'.format(accuracy, precision, recall, f1))


    vectorizer = CountVectorizer(binary=True)
    trainingData = vectorizer.fit_transform(trainData[:,0])
    trainingData = trainingData.toarray()
    testingData = vectorizer.transform(testData[:,0]).toarray()
    trainLabel = np.where(trainData[:,1]=='ham',1,0)
    clf = GridSearchCV(estimator=SGDClassifier(), param_grid={'alpha': [0.001,0.01, 0.1, 1, 10], 'loss':["hinge", "log_loss"], 'max_iter':[100,200,500,800,1000],'penalty': ['l1', 'l2']}, n_jobs=-1)

    clf.fit(trainingData,trainLabel)
    print("Ideal parameters are: {}".format(clf.best_params_))

    predY = clf.predict(testingData)
    trueY = np.where(testData[:,1]=='ham', 1, 0)

    accuracy = accuracy_score(trueY,predY)
    precision, recall, f1, support = precision_recall_fscore_support(trueY, predY)
    print('Algorithm: Bernoulli Linear Regression: SGD Classifier')
    print('\tAccuracy:\t{}\n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1 Score:\t{}'.format(accuracy, precision, recall, f1))


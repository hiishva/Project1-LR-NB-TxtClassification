import numpy as np
import pandas as pd
import os
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import math
from scipy.special import expit
import sys
from helperFunctions import get_accuracy_precision_recall_f1_score, read_data_given_folder_and_label, get_bow_and_bernoulli

def LogRegTrain(trainNpArr, labelArr, learnRate, lambdaVal):
    weight = np.zeros((1,trainNpArr.shape[1]))
    noIters = 50
    for i in range(noIters):
        predict = expit(np.multiply(trainNpArr,weight).sum(axis=1))
        diffErr = labelArr - predict
        predictUpdt = np.multiply(trainNpArr, diffErr.reshap(-1,1))
        weightUpdt = predictUpdt.sum(axis=0)
        weightN = weight + ((learnRate * weightUpdt) - learnRate * lambdaVal * weight)
        weight = weightN.copy()
    return weight

def LogRegTest(weight, testData):
    predict = expit(np.multiply(testData[:, :-1],weight).sum(axis=1))
    pred = np.where(predict >= 0.5, 1, 0)
    return pred

def LogReg(trainData, splitVal, lambdaVal, bernFlg):
    splitParam = int(trainData.shape[0] * 0.7)

    validData = trainData[splitParam:].copy()
    trainDataAr = trainData[:splitParam].copy()

    if bernFlg == True:
        vectorizer = CountVectorizer(binary=True)
    else:
        vectorizer = CountVectorizer()
    
    bow = vectorizer.fit_transform(trainDataAr[:0])
    bow = bow.toarray()

    label = np.where(trainDataAr[:,1] == 'ham',1,0).reshape(-1,1)
    biasW = np.ones((bow,biasW))
    trainNp = np.hstack((bow, biasW))
    trainNp = np.hstack(trainNp,label)

    acc = 0

    for lambdas in lambdaVal:
        weight = LogRegTrain(trainNp[:,:-1], trainNp[:,-1],learnRate=0.01, lambdaVal=lambdas)

        validataion = vectorizer.transform(validData[:,0]).toarray()
        label = np.where(validData[:,1]=='ham', 1, 0).reshape(-1,1)
        biasW = np.ones((validData.shape[0],1))

        validataionArr = np.hstack(validataion,biasW)
        validataionArr = np.hstack(validataionArr,label)

        pred = LogRegTest(weight,validataionArr)
        curAcc = (pred == np.where(validData[:,1]=='ham',1,0)).sum()/pred.shape[0]

        if curAcc > acc:
            acc = curAcc
            weightF = weight
            lambdaF = lambdas
    if bernFlag == True:
        vectorizerN = CountVectorizer(binary=True)
    else:
        vectorizerN = CountVectorizer()

    bowOrBern = vectorizerN.fit_transform(trainData[:,0]).toarray
    biasW = np.ones((bowOrBern.shape[0], 1))
    trainLables = np.where(trainData[:,1]=='ham',1,0)
    trainNp = np.hstack((bowOrBern,biasW))
    weightF = LogRegTrain(trainNp, trainLables,learnRate=0.01, lambdaVal=lambdaF)

    return weightF, vectorizerN

path =  sys.argv[1]
if len(sys.argv) == 3:
    seedVal = int(sys.argv[2])
else:
    seedVal = 1000

trainData = read_data_given_folder_and_label(path, test = False)
testData = read_data_given_folder_and_label(path, test = True )
np.random.seed(seedVal)
np.random.shuffle(trainData)

splitVal = 0.7
lambdaVals = [0.0001, .001, .1, 1, 5]

bernFlag = False
weightF, vectorizerN = LogReg(trainData,splitVal, lambdaVals, bernFlag)

testDf = vectorizerN.transform(testData[:,0]).toarray()
labels = np.where(testData[:,1]=='ham',1,0).reshape(-1,1)
biasW = np.ones((testDf.shape[0],1))
testDataNpAr = np.hstack((testDf,biasW))
testDataNpAr = np.hstack((testDataNpAr, labels))
preds = LogRegTest(weightF, testDataNpAr)

testLables = np.where(testData[:,1]=='ham',1,0)
accuracy = accuracy_score(testLables,)
print('Analysis of Logistic Regression:')
print('---------------------------------------')
print('predict = {}'.format(preds))


import numpy as np
import os
import string
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def ReadFromFolder(foldPath, test):
    if test == True:
        pathSub = 'test/'
    else:
        pathSub = 'train/'
    
    label = ['ham', 'spam']
    retData = []
    for hamSpam in label:
        data = []
        labelArr = []
        foldPathN = foldPath + pathSub + hamSpam + '/'
        print('Reading from folder:', foldPathN)
        
        for files in os.listdir(foldPathN):
            str_r = open(foldPathN + files, 'r', errors='ignore').read()
            str_r = str_r.translate(str.maketrans('', '', string.punctuation))
            str_r = re.sub("[a^a-zA-Z'-]+", ' ', str_r)
            data.append(str_r)
            labelArr.append(hamSpam)
        retData.append(np.column_stack((data,labelArr)))
        print("Files are loaded")
    finData = np.vstack((retData[0],retData[1]))
    return finData

def getBOWAndBER(trainDat, BernFlg):
    if BernFlg == True:
        vectorizer = CountVectorizer(binary=True)
    else:
        vectorizer = CountVectorizer
    
    npArr = vectorizer.fit_transform(trainDat[:,0])
    bow = npArr.toarray()
    UniqWord = pd.DataFrame(vectorizer.get_feature_names())
    return bow, UniqWord

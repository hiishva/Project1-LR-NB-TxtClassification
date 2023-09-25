import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from HelperFunctions import ReadFromFolder, getBOWAndBER

def trainMultinomialNB(trainData, uniqueCorpus):
    classes = ['ham','spam']
    priors = pd.DataFrame(trainData[:,1]).value_counts(normalize=True)
    priors = priors.to_numpy()
    conditionalProbabilities = []
    
    for each in classes:
        spam_ham_text = trainData[np.where(trainData[:,1]==each)]
        tokenVectorizer = CountVectorizer()
        tokenVector = tokenVectorizer.fit_transform(spam_ham_text[:,0]).toarray()
        
        wordCount = tokenVector.sum(axis=0).reshape(-1,1)
        uniqueWords = np.array(tokenVectorizer.get_feature_names()).reshape(-1,1).astype('object')

        Tct = np.hstack((uniqueWords, wordCount))

        countDataFrame = pd.DataFrame(Tct)
        TctWithV = uniqueCorpus.merge(countDataFrame, how='left').fillna(0)
        TctWithV.columns = ['word', 'freq']
        st = 'condtl_prob_given_' + each
        TctWithV[st] = (TctWithV['freq'] + 1)/(TctWithV['freq'].sum() + uniqueCorpus.shape[0])
        TctWithV.drop('frequency', axis=1, inplace=True)
        conditionalProbabilities.append(TctWithV)
    
    condtlProbMatrix = conditionalProbabilities[0].merge(conditionalProbabilities[1])
    condtlProbMatrix['condtl_prob_given_ham'] = np.log(condtlProbMatrix['condtl_prob_given_ham'])
    condtlProbMatrix['condtl_prob_given_spam'] = np.log(condtlProbMatrix['condtl_prob_given_spam'])
    
    return condtlProbMatrix, priors

def testMultinomialNB(testData, condtlProbMatrix, priors):
    yPred = []
    yTrue = []

    for data in testData:
        words = data[0].split(' ')
        if data[1]=='ham':
            yTrue.append(1)
        else:
            yTrue.append(0)

        occurringWordCondtlProb = condtlProbMatrix[condtlProbMatrix['words'].isin(words)].reset_index(drop=True)
        
        classAndCondtlProb = occurringWordCondtlProb.sum(axis=0)[1:]
        classAndCondtlProb[0] += np.log(priors[0]) # ham
        classAndCondtlProb[1] += np.log(priors[1]) # spam

        if classAndCondtlProb[0] > classAndCondtlProb[1]:
            yPred.append(1)
        else:
            yPred.append(0)
        
    return {'yPred':yPred, 'yTrue':yTrue}

if __name__ == '__main__':
    path = sys.argv[1]
    trainData = ReadFromFolder(path, test=False)
    testData = ReadFromFolder(path, test=True)

    bagOfWords, uniqueCorpusDF = getBOWAndBER(trainData, False)

    condtlProbMatrix, priors = trainMultinomialNB(trainData, uniqueCorpusDF)
    yDict = testMultinomialNB(testData, condtlProbMatrix, priors)

    accuracy = accuracy_score(yDict['yTrue'], yDict['yPred'])
    precision, recall, f1, support = precision_recall_fscore_support(yDict['yTrue'], yDict['yPred'])

    print('Algorithm: Multinomial Naive Bayes')
    print('\tAccuracy:\t{}\n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1 Score:\t{}'.format(accuracy, precision, recall, f1))
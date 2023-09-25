import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from helper_functions import read_data_given_folder_and_label, get_bow_and_bernoulli

def trainBernoulliNB(trainData, uniqueCorpusDataFrame):
    classes = ['ham','spam']
    priors = pd.DataFrame(trainData[:,1]).value_counts(normalize=True)
    priors = priors.to_numpy()
    conditionalProbabilities = []
    for each in classes:
        spam_ham_text = trainData[np.where(trainData[:,1]==each)]
        tokenVectorizer = CountVectorizer()
        tokenVector = tokenVectorizer.fit_transform(spam_ham_text[:,0]).toarray()
        tokenVector = np.where(tokenVector>0, 1, 0)
        wordCount = tokenVector.sum(axis=0).reshape(-1,1)
        uniqueWords = np.array(tokenVectorizer.get_feature_names()).reshape(-1,1).astype('object')

        Tct = np.hstack((uniqueWords, wordCount))

        countDataFrame = pd.DataFrame(Tct)
        TctWithV = uniqueCorpusDataFrame.merge(countDataFrame, how='left').fillna(0)
        TctWithV.columns = ['word', 'freq']
        st = 'condtl_prob_given_' + each
        TctWithV[st] = (TctWithV['freq'] + 1)/(TctWithV['freq'].sum() + uniqueCorpusDataFrame.shape[0])
        TctWithV.drop('frequency', axis=1, inplace=True)
        conditionalProbabilities.append(TctWithV)
    
    condtlProbMatrix = conditionalProbabilities[0].merge(conditionalProbabilities[1])
    return condtlProbMatrix, priors

def testBernoulliNB(test_data, condtitionalProbMatrix, priors):
    yPred = []
    yTrue = []
    
    for data in test_data:
        words = data[0].split(' ')
        
        if data[1]=='ham':
            yTrue.append(1)
        else:
            yTrue.append(0)
        
        occurringWordCondtlProb = condtitionalProbMatrix[condtitionalProbMatrix['words'].isin(words)].reset_index(drop=True)
        
        nonOccurringWordCondtlProb = condtitionalProbMatrix[~condtitionalProbMatrix['words'].isin(words)].reset_index(drop=True)
        nonOccurringWordCondtlProb['condtl_prob_given_ham'] = 1 - nonOccurringWordCondtlProb['condtl_prob_given_ham']
        nonOccurringWordCondtlProb['condtl_prob_given_spam'] = 1 - nonOccurringWordCondtlProb['condtl_prob_given_spam']

        pHam = np.log(occurringWordCondtlProb['condtl_prob_given_ham']).sum() + np.log(nonOccurringWordCondtlProb['condtl_prob_given_ham']).sum() + np.log(priors[0])
        pSpam = np.log(occurringWordCondtlProb['condtl_prob_given_spam']).sum() + np.log(nonOccurringWordCondtlProb['condtl_prob_given_spam']).sum() + np.log(priors[1])

        if pHam > pSpam:
            yPred.append(1)
        else:
            yPred.append(0)
        
    return {'yPred':yPred, 'yTrue':yTrue}
    
if __name__ == '__main__':
    path = sys.argv[1]
    trainData = read_data_given_folder_and_label(path, test=False)
    testData = read_data_given_folder_and_label(path, test=True)

    bagOfWords, uniqueCorpusDF = get_bow_and_bernoulli(trainData, True)

    condtlProbMatrix, priors = trainBernoulliNB(trainData, uniqueCorpusDF)
    yDict = testBernoulliNB(testData, condtlProbMatrix, priors)

    accuracy = accuracy_score(yDict['yTrue'], yDict['yPred'])
    precision, recall, f1, support = precision_recall_fscore_support(yDict['yTrue'], yDict['yPred'])

    print('Algorithm: Bernoulli Naive Bayes')
    print('\tAccuracy:\t{}\n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1 Score:\t{}'.format(accuracy, precision, recall, f1))
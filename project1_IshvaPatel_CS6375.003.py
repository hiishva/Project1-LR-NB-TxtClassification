from math import *
import numpy as np
import sys
import os
import re

from sklearn import metrics as met

reg = re.compile('[@_!#$%^&*()<>?/\|}{~:]')

## File paths 
datIdx = sys.argv[1]

cwd = os.getcwd()
HamTrainPath = 'project1_datasets/enron{}/train/ham'.format(datIdx)
SpamTrainPath = 'project1_datasets/enron{}/train/spam'.format(datIdx)
HamTestPath = 'project1_datasets/enron{}/test/ham'.format(datIdx)
SpamTestPath = 'project1_datasets/enron{}/test/spam'.format(datIdx)

hamTrainDir= "/".join([cwd, HamTrainPath])
spamTrainDir = "/".join([cwd, SpamTrainPath])
hamTestDir = "/".join([cwd, HamTestPath])
spamTestDir ="/".join([cwd, SpamTestPath])

trainDirs = [hamTrainDir, spamTrainDir]
testDirs = [hamTestDir, spamTestDir]

# def CreateVocab(directories):
#     totalVocab = []
#     hamVocab = []
#     spamVocab = []
#     for dir in directories:
#         with os.scandir(dir) as dir:
#             for file in dir:
#                 try:
#                     f = open(file,'r', encoding='latin-1')
#                     lines = f.readlines()
#                     for line in lines:
#                         words = line.strip().upper().split(" ")
#                         for word in words:
#                             if (reg.search(word) == None and word not in totalVocab):
#                                 totalVocab.append(word)
#                             if (reg.search(word) == None and dir.split('/')[-1] == 'ham' and word not in hamVocab):
#                                 hamVocab.append(word)
#                             elif (reg.search(word) == None and dir.split('/')[-1] == 'spam' and word not in spamVocab):
#                                 spamVocab.append(word)
#                     f.close()
#                 except UnicodeDecodeError as err:
#                     print('Error: {}'.format(err))
#     return(totalVocab, hamVocab, spamVocab)


## CREATING THE DATASETS
def CreateBagOfWords(directories):
    #Bag of Words model dataset
    spamDict = {}
    hamDict = {}
    totalDict = {}
    hamDocs = 0
    spamDocs = 0

    for dirs in directories:
        with os.scandir(dirs) as dir:
            for file in dir:
                if dirs.split('/')[-1] == 'ham':
                    hamDocs += 1
                if dirs.split('/')[-1] == 'spam':
                    spamDocs += 1
                try:
                    f = open(file,'r', encoding='latin-1')
                    lines = f.readlines()
                    for line in lines:
                        words = line.strip().upper().split(" ")
                        for word in words:
                            if word in totalDict:
                                totalDict[word] += 1
                            else:
                                totalDict[word] = 1
                            if (dirs.split('/')[-1] == 'ham'):
                                if word in hamDict:
                                    hamDict[word] += 1
                                else:
                                    hamDict[word] = 1
                            elif (dirs.split('/')[-1] == 'spam'):
                                if word in spamDict:
                                    spamDict[word] += 1
                                else:
                                    spamDict[word] = 1
                    f.close()
                except UnicodeDecodeError as err:
                    print('Error: {}'.format(err))
    return (totalDict, hamDict, spamDict, hamDocs, spamDocs)

def CreateBernoulli(directories):
    spamDict = {}
    hamDict = {}
    totalDict = {}
    hamDocs = 0
    spamDocs = 0
    
    for dirs in directories:
        with os.scandir(dirs) as dir:
            for file in dir:
                if dirs.split('/')[-1] == 'ham':
                    hamDocs += 1
                if dirs.split('/')[-1] == 'spam':
                    spamDocs += 1
                try:
                    f = open(file,'r', encoding='latin-1')
                    lines = f.readlines()
                    for line in lines:
                        words = line.strip().upper().split(" ")
                        for word in words:
                            totalDict[word] = 1
                            if (dirs.split('/')[-1] == 'ham'):
                                hamDict[word] = 1
                            elif (dirs.split('/')[-1] == 'spam'):
                                spamDict[word] = 1
                    f.close()
                except UnicodeDecodeError as err:
                    print('Error: {}'.format(err))
    return (totalDict, hamDict, spamDict, hamDocs, spamDocs)

# Multinomial NB
def TrainMultinomialNB(totalDict, hamDict, spamDict, hamDocs, spamDocs):
    prior = []
    cndtlProb = {}
    cndtlProb['spam'] = {}
    cndtlProb['ham'] = {}
    cndtlProbNoWord = {}
    cndtlProbNoWord['spam'] = {}
    cndtlProbNoWord['ham'] = {}


    #PRIORS
    prior.append(log(hamDocs/(hamDocs + spamDocs)))
    prior.append(log(spamDocs/(hamDocs + spamDocs)))

    #CONDITIONAL PROBS
    numSpamWords = len(hamDict.keys())
    numHamWords = len(spamDict.keys())
    numWords = len(totalDict.keys())
    
    for word in spamDict.keys():
        cndtlProb['spam'][word] = log((spamDict[word]+1) / (numSpamWords + numWords))

    for word in hamDict.keys():
        cndtlProb['ham'][word] = log((hamDict[word] + 1) / (numHamWords + numWords))
    

    cndtlProbNoWord['ham'] = log(1/(numHamWords + numWords))
    cndtlProbNoWord['spam'] = log(1/(numSpamWords + numWords))

    return prior, cndtlProb, cndtlProbNoWord

def TrainDiscreteNaiveBayes(totalDict, hamDict, spamDict, hamDocs, spamDocs):
    prior = []
    cndtlProb = {}
    cndtlProb['spam'] = {}
    cndtlProb['ham'] = {}
    cndtlProbNoWord = {}
    cndtlProbNoWord['spam'] = {}
    cndtlProbNoWord['ham'] = {}

    #PRIORS
    prior.append(log(hamDocs/(float(hamDocs + spamDocs))))
    prior.append(log(spamDocs/(float(hamDocs + spamDocs))))

    #CONDITIONAL PROBS
    numSpamWords = len(hamDict.keys())
    numHamWords = len(spamDict.keys())
    numWords = len(totalDict.keys())

    # For words in training set
    for word in spamDict.keys():
        cndtlProb['spam'][word] = log((1 + spamDict[word])/(float(spamDocs + 2)))
    
    for word in hamDict.keys():
        cndtlProb['ham'][word] = log((1 + hamDict[word])/(float(hamDocs + 2)))
    
    # For words not in training set:
    cndtlProbNoWord['spam'] = log((1/(float(spamDocs + 2))))
    cndtlProbNoWord['ham'] = log((1/(float(hamDocs + 2))))
    
    return prior, cndtlProb, cndtlProbNoWord

def TestMultinomialNB(directories, prior, cndtlProb, cndtlProbNoWord):
    y_test = []
    y_true = []
    for dirs in directories:
        with os.scandir(dirs) as dir:
            for file in dir:
                # Log true value
                if dirs.split('/')[-1] == 'ham':
                    y_true.append(1)
                if dirs.split('/')[-1] == 'spam':
                    y_true.append(0)

                # Calculate test value
                pHam = prior[0]
                pSpam = prior[1]
                
                try:
                    f = open(file,'r', encoding='latin-1')
                    lines = f.readlines()
                    for line in lines:
                        words = line.strip().upper().split(" ")
                        for word in words:
                            try:
                                pHam += cndtlProb['ham'][word]
                                pSpam += cndtlProb['spam'][word]
                            except KeyError as e:
                                pHam += cndtlProbNoWord['ham']
                                pSpam += cndtlProbNoWord['spam']
                    f.close()
                    if pHam < pSpam:
                        y_test.append(1)
                    else:
                        y_test.append(0)
                except UnicodeDecodeError as err:
                    print('Error: {}'.format(err))

    return y_test, y_true

def TestDiscreteNB(directories, prior, cndtlProb, cndtlProbNoWord):
    y_test = []
    y_true = []
    # score={}
    for dirs in directories:
        with os.scandir(dirs) as dir:
            for file in dir:
                # Log true value
                if dirs.split('/')[-1] == 'ham':
                    y_true.append(1)
                if dirs.split('/')[-1] == 'spam':
                    y_true.append(0)

                # Calculate test value
                pHam = prior[0]
                pSpam = prior[1]
                
                try:
                    f = open(file,'r', encoding='latin-1')
                    lines = f.readlines()
                    for line in lines:
                        words = line.strip().upper().split(" ")
                        for word in words:
                            try:
                                pHam += cndtlProb['ham'][word]
                                pSpam += cndtlProb['spam'][word]
                            except KeyError as e:
                                pHam += cndtlProbNoWord['ham']
                                pSpam += cndtlProbNoWord['spam']
                    f.close()
                    #ham is 1 spam is 0
                    if abs(pSpam) < abs(pHam):
                        y_test.append(1)
                    else:
                        y_test.append(0)
                except UnicodeDecodeError as err:
                    print('Error: {}'.format(err))
    return (y_test, y_true)






# Create dataset models
totalDictBOW, hamDictBOW, spamDictBOW, hamDocsBOW, spamDocsBOW = CreateBagOfWords(trainDirs)
totalDictBER, hamDictBER, spamDictBER, hamDocsBER, spamDocsBER = CreateBernoulli(trainDirs)

# Train and Test Naive Bayes
mn_prior, mn_cndtlProb, mn_cndtlProbNoWord = TrainMultinomialNB(totalDictBOW, hamDictBOW, spamDictBOW, hamDocsBOW, spamDocsBOW)
disc_prior, disc_cndtlProb, disc_cndtlProbNoWord = TrainDiscreteNaiveBayes(totalDictBER, hamDictBER, spamDictBER, hamDocsBER, spamDocsBER)

mn_ytest, mn_ytrue = TestMultinomialNB(testDirs, mn_prior, mn_cndtlProb, mn_cndtlProbNoWord)
#print(mn_ytrue)

disc_ytest, disc_ytrue = TestDiscreteNB(testDirs,disc_prior,disc_cndtlProb,disc_cndtlProbNoWord)
print(disc_ytrue)
print(disc_ytest)
disc_precision,disc_recall, disc_fscore, disc_support = met.precision_recall_fscore_support(disc_ytrue,disc_ytest)
mn_precision, mn_recall, mn_fscore, mn_support = met.precision_recall_fscore_support(mn_ytrue, mn_ytest)
print('Multinomial Naive Bayes:\n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1-score:\t{}'.format(mn_precision, mn_recall, mn_fscore))
print()
print('Discrete Naive Bayes: \n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1-score:\t{}'.format(disc_precision,disc_recall,disc_fscore))
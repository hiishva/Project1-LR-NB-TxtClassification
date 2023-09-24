from math import *
import numpy as np
import sys
import os
import re

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

def CreateVocab(directories):
    vocab = []
    for dir in directories:
        with os.scandir(dir) as dir:
            for file in dir:
                try:
                    f = open(file,'r', encoding='latin-1')
                    lines = f.readlines()
                    for line in lines:
                        words = line.strip().upper().split(" ")
                        for word in words:
                            if (reg.search(word) == None):
                                if(word not in vocab):
                                    vocab.append(word)
                    f.close()
                except UnicodeDecodeError as err:
                    print('Error: {}'.format(err))
    return(vocab)


## CREATING THE DATASETS
def CreateBagOfWords(directories, vocab):
    #Bag of Words model dataset
    dataset = np.zeros((len(vocab),1))
    fileSpamOrHam = 0
    spamOrHam = []
    for dir in directories:
        if dir.split('/')[-1] == 'ham':
            fileSpamOrHam = 1
        else:
            fileSpamOrHam = 0
        with os.scandir(dir) as dir:
            for file in dir:
                fileBag = np.zeros((len(vocab),1))
                try:
                    f = open(file,'r', encoding='latin-1')
                    lines = f.readlines()
                    for line in lines:
                        words = line.strip().upper().split(" ")
                        for word in words:
                            if (reg.search(word) == None and word in vocab):
                                idx = vocab.index(word)     # Get index from input vocab
                                fileBag[idx] += 1           # Increment word by 1 if present in message
                    f.close()
                    dataset = np.append(dataset, fileBag, axis=1)
                    spamOrHam.append(fileSpamOrHam)
                except UnicodeDecodeError as err:
                    print('Error: {}'.format(err))
    # print(dataset.shape)
    return(dataset, spamOrHam)

def CreateBernoulli(directories, vocab):
    #Bag of Words model dataset
    dataset = np.zeros((len(vocab),1))
    fileSpamOrHam = 0
    spamOrHam = []
    for dir in directories:
        if dir.split('/')[-1] == 'ham':
            fileSpamOrHam = 1
        else:
            fileSpamOrHam = 0
        with os.scandir(dir) as dir:
            for file in dir:
                fileBag = np.zeros((len(vocab),1))
                try:
                    f = open(file,'r', encoding='latin-1')
                    lines = f.readlines()
                    for line in lines:
                        words = line.strip().upper().split(" ")
                        for word in words:
                            if (reg.search(word) == None and word in vocab):
                                idx = vocab.index(word)     # Get index from input vocab
                                fileBag[idx] = 1            # Set word to 1 if present in message
                    f.close()
                    dataset = np.append(dataset, fileBag, axis=1)
                    spamOrHam.append(fileSpamOrHam)
                except UnicodeDecodeError as err:
                    print('Error: {}'.format(err))
    return(dataset, spamOrHam)


train_vocab = CreateVocab(trainDirs)
print("Created Vocabulary...")
XTrainBOW, yTrainBOW = CreateBagOfWords(trainDirs, train_vocab)
XTrainBernoulli, yTrainBernoulli = CreateBernoulli(trainDirs, train_vocab)
print("Created training sets...")
print("training dataset shapes:\n\tBag of words: {}\n\tBernoulli: {}".format(XTrainBOW.shape, XTrainBernoulli.shape))
print(XTrainBOW)
print(yTrainBOW)
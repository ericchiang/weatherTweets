#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

import csv
import sys
import json
import re
import string
import numpy as np
from sklearn import cross_validation
from naivebayes import BagOfWordsBayes
from utils import printInfo,meanSquaredError # utils.py

meta_tags = ['@mention','{link}']

def usage():
    print """
        Usage:
        %s [train.csv] > [featurefile]
        """ % (sys.argv[0],)

"""
Parse CSV file into data matrix
"""
def parseTraining(tweet_train):
    data = []
    tweet_reader = csv.reader(tweet_train, delimiter=',', quotechar='"')
    for row in tweet_reader:
        data.append(row)
    return data

"""
Parse tweet into bag of words
"""
def parseTweet(tweet):
    words = []
    for tag in meta_tags:
        tweet = tweet.replace(tag,'')
    tweet = tweet.split()
    for i in range(len(tweet)):
        word = tweet[i].lower()
        while True:
            last_word = word
            word = word.strip(string.punctuation)
            if word == last_word:
                break
        if word:
            words.append(word)
    return words


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
        sys.exit(2)

    try:
        tweet_train = open(sys.argv[1], 'rb')
    except IOError:
        sys.stderr.write("[Error] Could not open file '%s'" % (sys.argv[1],))
        sys.exit(14)

    # Parse csv file
    data = parseTraining(tweet_train)
    # Get header data
    headers = data[0]

    # Parse tweets
    tweet_data = []
    for row in data:
        tweet_data.append(parseTweet(row[1]))
    assert len(tweet_data) == len(data)    
    printInfo("%d tweets parsed" % (len(tweet_data),))

    label = 's1'
    # Get index of desired label
    label_index = headers.index(label)

    printInfo("Preforming training on class '%s'" % (label,))

    # Generate training data
    X = tweet_data[1:]
    y = []
    for row in data[1:]:
        y.append(row[label_index])

    X = np.array(X)
    y = np.array(y)

    naiveBayes = BagOfWordsBayes()

    printInfo("Preforming cross validation")

    num_folds = 10
    k_fold = cross_validation.KFold(n=len(X), n_folds=num_folds, indices=True)

    y_pred = [''] * len(y)

    fold_number = 1
    for train_indices,test_indices in k_fold:
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test  = X[test_indices]
        y_test  = y[test_indices]

        progress = (fold_number,num_folds,)
        printInfo("%2d/%d Training" % progress)
        naiveBayes = BagOfWordsBayes()
        naiveBayes.fit(X_train,y_train)
        printInfo("%2d/%d Predicing" % progress)
        fold_y_pred = naiveBayes.predict(X_test)
        print fold_y_pred
        for i in range(len(test_indices)):
            assert y[test_indices[i]] == y_test[i]
            y_pred[test_indices[i]] = fold_y_pred[i]
        fold_number += 1
    printInfo("MSE %s" % (meanSquaredError(y_pred,y),))

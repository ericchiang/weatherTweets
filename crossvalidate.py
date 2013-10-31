#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

import csv
import sys
import re
import math
import numpy as np
from sklearn import cross_validation
from naivebayesV2 import BagOfWordsBayes
from utils import printInfo,meanSquaredError,rootMeanSquaredError # utils.py
from featextract import * # featextract.py
from hmm import HiddenMarkovModel

def usage():
    print """
        Usage:
        %s [train.csv] [var names... ] 
        """ % (sys.argv[0],)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)

    try:
        tweet_train = open(sys.argv[1], 'rb')
    except IOError:
        sys.stderr.write("[Error] Could not open file '%s'" % (sys.argv[1],))
        sys.exit(14)

    if len(sys.argv) > 2:
        var_names = sys.argv[2:]
    else:
        var_names = ['s1','s2','s3','s4','s5',
                     'w1','w2','w3','w4',
                     'k1','k2','k3','k4','k5','k6','k7','k8',
                     'k9','k10','k11','k12','k13','k14','k15']

    # Parse csv file
    data = parseTwitterCSV(tweet_train)
    # Get header data
    headers = data[0]

    data = data[1:]

    # Parse tweets
    tweet_data = []
    for row in data:
        t_data = parseTweet(row[1])
        tweet_data.append(t_data)
    assert len(tweet_data) == len(data)    
    printInfo("%d tweets parsed" % (len(tweet_data),))

    mses = []

    for var_name in var_names:
        # Get index of desired label
        var_index = headers.index(var_name)
        if not var_index:
            printInfo("No variable '%s' found..." % (var_name))
            continue
 
        # printInfo("Calculating scores for variable '%s'" % (var_name,))
    
        # Generate training data
        X = tweet_data
        y = []
        for row in data:
            y.append(row[var_index])
    
        X = np.array(X)
        y = np.array(y)
    
        naiveBayes = BagOfWordsBayes()
    
        # printInfo("Preforming cross validation")
    
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
            classifier = BagOfWordsBayes()
            classifier.fit(X_train,y_train)
            fold_y_pred = classifier.predict(X_test)
            for i in range(len(test_indices)):
                assert y[test_indices[i]] == y_test[i]
                y_pred[test_indices[i]] = fold_y_pred[i]
            printInfo(" %2s/%s" % (fold_number,num_folds,))
            fold_number += 1
        mse = meanSquaredError(y_pred,y)
        printInfo("%s MSE: %s" % (var_name,mse))
        mses.append(mse)
    totalMSE = sum(mses) / len(mses)
    printInfo("Total MSE: %s" % (totalMSE,))
    printInfo("Total RMSE: %s" % (math.sqrt(totalMSE),))

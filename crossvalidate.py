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
import math
import numpy as np
from sklearn import cross_validation
from naivebayes import BagOfWordsBayes
from utils import printInfo,meanSquaredError,rootMeanSquaredError # utils.py
from featextract import * # featextract.py


def usage():
    print """
        Usage:
        %s [train.csv] [var names... ] 
        """ % (sys.argv[0],)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        sys.exit(2)

    try:
        tweet_train = open(sys.argv[1], 'rb')
    except IOError:
        sys.stderr.write("[Error] Could not open file '%s'" % (sys.argv[1],))
        sys.exit(14)

    var_names = sys.argv[2:]

    # Parse csv file
    data = parseTwitterCSV(tweet_train)
    # Get header data
    headers = data[0]

    data = data[1:]

    # Parse tweets
    tweet_data = []
    for row in data:
        tweet_data.append(parseTweet(row[1]))
    assert len(tweet_data) == len(data)    
    printInfo("%d tweets parsed" % (len(tweet_data),))

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
            naiveBayes = BagOfWordsBayes()
            naiveBayes.fit(X_train,y_train)
            fold_y_pred = naiveBayes.predict(X_test)
            for i in range(len(test_indices)):
                assert y[test_indices[i]] == y_test[i]
                y_pred[test_indices[i]] = fold_y_pred[i]
            #printInfo(" %2s/%s" % (fold_number,num_folds,))
            fold_number += 1
        mse = meanSquaredError(y_pred,y)
        print("%s %s" % (var_name,mse))

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
from naivebayes import BagOfWordsBayes

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
    for tag in meta_tags:
        tweet = tweet.replace(tag,'')
    tweet = tweet.split()
    for i in range(len(tweet)):
        tweet[i] = tweet[i].lower().strip(string.punctuation)
    return tweet 


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
    print "[+] %d tweets parsed" % (len(tweet_data),)

    label = 's3'
    # Get index of desired label
    label_index = headers.index(label)

    print "[+] Preforming training on class '%s'" % (label,)

    n_test = 200

    # Generate training data
    train_x = tweet_data[1:-n_test]
    train_y = []
    for row in data[1:-n_test]:
        train_y.append(row[label_index])


    # Generate test data
    test_x = tweet_data[-n_test:]
    test_y = []
    for row in data[-n_test:]:
        test_y.append(row[label_index])

    # Model file information
    model_file_name = 'model/bayesModel.data'
    read_model_from_file = True

    naiveBayes = BagOfWordsBayes()

    if read_model_from_file:
        try:
            model_file = open(model_file_name,'r')
            naiveBayes.readInModel(model_file)
            model_file.close()
        except IOError:
            print "[-] Error reading model from file"
            sys.exit(3)
    else:
        # Initialize bayes classifier
        naiveBayes = BagOfWordsBayes()
        # Train bayes classifier
        print "[+] Training classifier..."
        naiveBayes.train(train_x,train_y)
        print "[+] Classifier trained"
    
        print "[+] Writing classifier model to file"
        try:
            model_file = open(model_file_name,'w')
            naiveBayes.writeOutModel(model_file)
            model_file.close()
            print "[+] Classifier model written"
        except IOError:
            print "[-] Error writing classifier to file"

    print "[+] Fitting data..."
    pred_y = naiveBayes.fit(test_x)
    print "[+] Data fit!"
    for i in range(len(test_y)):
        print "%s\t%s\t%s" % (test_y[i],pred_y[i][0],pred_y[i][1])

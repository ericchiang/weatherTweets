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
    print "%d tweets parsed" % (len(tweet_data),)

    label = 's3'
    # Get index of desired label
    label_index = headers.index(label)

    print headers
    print headers[label_index]
    # Generate training data
    train_x = tweet_data[1:-100]
    train_y = []
    for row in data[1:-100]:
        conf = row[label_index]
        if float(conf) > 0.3:
            train_y.append('1')
        else:
            train_y.append('0')
 
    print set(train_y)

    if len(set(train_y)) > 5:
        sys.exit(0)

    # Generate test data
    test_x = tweet_data[-100:]
    test_y = []
    for row in data[-100:]:
        conf = row[label_index]
        test_y.append(conf)

    # Initialize bayes classifier
    naiveBayes = BagOfWordsBayes()
    # Train bayes classifier
    naiveBayes.train(train_x,train_y)
    print "Classifier trained"
    pred_y = naiveBayes.fit(test_x)
    print "Results created!"
    for i in range(len(test_y)):
        print "%s\t%s" % (test_y[i],pred_y[i])

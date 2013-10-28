
import json
import sys
from naivebayes import BagOfWordsBayes
from featextract import *

def usage():
    print """
        Usage:
        python %s [train.csv] [test.csv] [varname] > [jsonfile]
        """ % (sys.argv[0],)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        usage()
        sys.exit(2)

    try:
        train_file = open(sys.argv[1], 'rb')
    except IOError:
        sys.stderr.write("[Error] Could not open file '%s'" % (sys.argv[1],))
        sys.exit(14)

    try:
        test_file = open(sys.argv[2], 'rb')
    except IOError:
        sys.stderr.write("[Error] Could not open file '%s'" % (sys.argv[2],))
        sys.exit(14)

    var_name = sys.argv[3]

    train_data = parseTwitterCSV(train_file)
    test_data = parseTwitterCSV(test_file)

    train_header = train_data[0]
    test_header = test_data[0]

    train_data = train_data[1:]
    test_data = test_data[1:]

    tweet_index_train = train_header.index('tweet')
    tweet_index_test = test_header.index('tweet')

    var_index = train_header.index(var_name)

    X_train = []
    X_test = []
    y_train = []

    for row in train_data:
        X_train.append(parseTweet(row[tweet_index_train]))
        y_train.append(row[var_index])

    for row in test_data:
        X_test.append(parseTweet(row[tweet_index_train]))

    naiveBayes = BagOfWordsBayes()
    naiveBayes.fit(X_train,y_train)
    y_pred = list(naiveBayes.predict(X_test))
    assert len(y_pred) == len(X_test)

    print json.dumps(y_pred)


import sys
from naivebayes import BagOfWordsBayes
from featextract import *

def usage():
    print """
        Usage:
        python %s [train.csv] [test.csv] > [csvfile]
        """ % (sys.argv[0],)

variable_names = ['s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3',
                  'k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14',
                  'k15']


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
    id_index = test_header.index('id')

    X_train = []
    X_test = []
    y_train = []

    test_ids = []

    for row in train_data:
        X_train.append(parseTweet(row[tweet_index_train]))
        y_train.append(row[var_index])

    for row in test_data:
        X_test.append(parseTweet(row[tweet_index_train]))
        test_ids.append(row[id_index])

    naiveBayes = BagOfWordsBayes()
    naiveBayes.fit(X_train,y_train)
    y_pred = naiveBayes.predict(X_test)
    assert len(y_pred) == len(X_test)


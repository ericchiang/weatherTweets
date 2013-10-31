import sys
import math
import numpy as np
import random
from sklearn import linear_model

class HiddenMarkovModel:

    """
    Fit data
    """
    def fit(self,X,y,holdout_percent=0.05):

        assert len(X) == len(y), \
              "Bags and intervals have different lengths"
        for i in range(len(y)):
            conf_inter = float(y[i])
            assert conf_inter >= 0.0 and conf_inter <= 1.0, \
                  "Bad CI '%s' at index [%s]" % (conf_inter, i)
            y[i] = conf_inter

        num_datapoints = len(X)
        indices = range(num_datapoints)
        random.shuffle(indices)
        num_holdout = int(num_datapoints * holdout_percent)

        train_indices = indices[num_holdout:]
        holdout_indices = indices[:num_holdout]

        X = np.array(X)
        y = np.array(y)

        X_train   = X[train_indices]
        X_holdout = X[holdout_indices]
        y_train   = map(np.float,y[train_indices])
        y_holdout = map(np.float,y[holdout_indices])


        self.in_ngrams = self.generateNGrams(X_train,y_train)
        self.out_ngrams = self.generateNGrams(X_train,map(one_minus_n,y_train))

        self.internal_classifier.fit(self.genLogProbs(X_holdout),y_holdout)


    def generateNGrams(self,X,y):
        rare_words = self.getRareWords(X,y)
        # uni, bi and tri grams. index 0 is for word counts
        ngrams = [0.0,{},{},{}] 
        
        for i in range(len(X)):
            sent = [' * ',' * ']
            # replace rare words
            for word in X[i]:
                try:
                    _ = rare_words[word]
                    sent.append(self._handleRareWord(word))
                except:
                    sent.append(word)
            sent.append(' STOP ')
            prob = y[i]
            ngrams[0] += prob * len(sent)
            for n in (1,2,3):
                grams = [sent[i:i+n] for i in range(len(sent)-n+1)]
                for gram in grams:
                    gram = tuple(gram)
                    try:
                        ngrams[n][gram] += prob
                    except:
                        ngrams[n][gram] = prob
        return ngrams


    def getRareWords(self,X,y):
        rare_words = {} # use dict for speed
        word_freq = {}
        for i in range(len(X)):
            sent,freq = X[i],y[i]
            for word in sent:
                try:
                    word_freq[word] += freq
                except:
                    word_freq[word] = freq

        for word in word_freq.keys():
            if word_freq[word] < self.rare:
                rare_words[word] = True

        return rare_words

    """
    To prevent zero probabilities replace rare words with a unique tag
    """
    def _handleRareWord(self,word):
        return ' RARE ' 

    def predict(self,X_test):
        y = self.internal_classifier.predict(self.genLogProbs(X_test))
        for i in range(len(y)):
            if y[i] < 0.0:
                y[i] = 0.0
            elif y[i] > 1.0:
                y[i] = 1.0
        return y

    def genLogProbs(self,X):
        log_probs = []
        for sent in X:
            log_probs.append([math.log(self.genClassProb(sent,'in_class')),
                              math.log(self.genClassProb(sent,'out_class'))])
        return log_probs

    def genProbs(self,X):
        probs = []
        for sent in X:
            probs.append([self.genClassProb(sent,'in_class'),
                          self.genClassProb(sent,'out_class')])
        return probs

    """
    Generate in class and out class probabilities for a given sentence
    """
    def genClassProb(self,sent,class_name):
        if class_name == 'in_class':
            ngrams = self.in_ngrams
        elif class_name == 'out_class':
            ngrams = self.out_ngrams
        else:
            raise Exception("Bad class name: '%s'" % (class_name,))

        prob = 1.0
        s = [' * ',' * ']
        for word in sent:
            try:
                _ = ngrams[1][(word)] # Check if word is in unigrams
                s.append(word)      # If it is keep going
            except:
                s.append(self._handleRareWord(word))
        s.append(' STOP ')

        word_count = ngrams[0]

        for i in range(len(s))[2:]:
            try:
                unigram_freq = ngrams[1][tuple(s[i:i+1])]
            except:
                print s[i:i+1]
                print s
                print sent
                print ngrams[1][tuple(s[i:i+1])]
                sys.exit(2)
            try:
                bigram_freq = ngrams[2][tuple(s[i-1:i+1])]
                if bigram_freq == 0.0:
                    print s[i-1:i+1]
                    raise Exception()
            except:
                prob *= self.lambdas[0] * (unigram_freq / word_count)
                continue
            try:
                trigram_freq = ngrams[3][tuple(s[i-2:i+1])]
            except:
                prob *= (self.lambdas[0] * (unigram_freq / word_count)) + \
                        (self.lambdas[1] * (bigram_freq / unigram_freq))
                continue
            prob *= (self.lambdas[0] * (unigram_freq / word_count)) + \
                    (self.lambdas[1] * (bigram_freq / unigram_freq)) + \
                    (self.lambdas[2] * (trigram_freq / bigram_freq))
        assert prob != 0.0
        return prob
                    

    def __init__(self,rare=3.0,lambdas=(0.3,0.5,0.2)):
        self.rare = rare
        self.lambdas = lambdas
        self.in_num_words = 0.0
        self.out_num_words = 0.0
        self.in_ngrams = {}
        self.out_ngrams = {}
        self.internal_classifier = linear_model.LinearRegression()

def one_minus_n(n):
    return 1.0 - n

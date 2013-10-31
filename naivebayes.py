#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0" 

import math
import sys
import json
import numpy as np
import random
from threading import Thread
from Queue import Queue
from sklearn import linear_model

import utils # utils.py


"""
Binary Bag of Words Naive Bayes algorithm. Takes and produces inclass 
confidence intervals. 
"""
class BagOfWordsBayes(object):

    """
    Train the classifier. X is a list containing bags of words. Y is a list of 
    confidence intervals corresponding to the confidence that a specific bag
    is inclass.
    """
    def fit(self, x, y, num_threads=2,verbose=False,holdout_percent=0.01):

        assert len(x) == len(y), "Bags and intervals have different lengths"
        for i in range(len(y)):
            conf_inter = float(y[i])
            assert conf_inter >= 0.0 and conf_inter <= 1.0, \
                  "Bad CI '%s' at index [%s]" % (conf_inter, i)
            y[i] = conf_inter

        if verbose:
            utils.printInfo("Calculating class probabilities")

        num_datapoints = len(x)
        indices = range(num_datapoints)
        random.shuffle(indices)
        num_holdout = int(num_datapoints * holdout_percent)

        train_indices = indices[num_holdout:]
        holdout_indices = indices[:num_holdout]

        x = np.array(x)
        y = np.array(y)

        x_train   = x[train_indices]
        x_holdout = x[holdout_indices]
        y_train   = map(np.float,y[train_indices])
        y_holdout = map(np.float,y[holdout_indices])

        # Calculate priori class probabilities
        total_freq = float(len(x_train))
        inclass_freq  = sum(y_train)
        outclass_freq = total_freq - inclass_freq
        self.class_freqs[self.inclass]   = inclass_freq
        self.class_freqs[self.outclass]  = outclass_freq
        self.class_priori[self.inclass]  = inclass_freq  / total_freq
        self.class_priori[self.outclass] = outclass_freq / total_freq

        if verbose:
            utils.printInfo("Generating word frequency tuples")

        for i in range(len(x_train)):
            inclass_freq = y_train[i]
            outclass_freq = 1.0 - inclass_freq
            for word in set(x_train[i]):
                try:
                    self.word_freqs[word][0] += inclass_freq
                    self.word_freqs[word][1] += outclass_freq
                except KeyError:
                    self.word_freqs[word] = [inclass_freq,outclass_freq]



        """
        Train internal classifier
        """
        self.internal_classifier.fit(self._genLogProbs(x_holdout),y_holdout)

    """
    Calculate mean squared error
    """
    def score(self,x,y):
        return untils.meanSquaredError(self.predict(x),y)

    """
    Run internal 
    """
    def predict(self,x):
        log_probs = self._genLogProbs(x)
        y = self.internal_classifier.predict(log_probs)
        for i in range(len(y)):
            if y[i] < 0.0:
                y[i] = 0.0
            elif y[i] > 1.0:
                y[i] = 1.0
        return y

    """
    Fit a bag of words. Returns inclass and outclass probabilities
    """  
    def _genLogProbs(self,x):
       probs = []
       for i in range(len(x)):
          bag_of_words = x[i]
          probs.append(self._bagLogProbs(bag_of_words))
       assert len(probs) == len(x)
       return probs

    """
    Classify a bag of words. Probabilities are calculated in log space to avoid
    float underflow
    """
    def _bagLogProbs(self,bag_of_words):
        p = [0.0,0.0]
        n = [0.0,0.0]
        for word in bag_of_words:
            p_in  = self._correctedClassProb(self.inclass,word)
            p_out = self._correctedClassProb(self.outclass,word)
            assert p_in  < 1.0 and p_in  > 0.0, \
                  "%s %s" % (word,p_in)
            assert p_out < 1.0 and p_out > 0.0, \
                  "%s %s" % (word,p_out)
            n[0] += math.log(1.0 - p_in) - math.log(p_in)
            n[1] += math.log(1.0 - p_out) - math.log(p_out)
        p[0] = 1.0 / (1.0 + math.exp(n[0]))
        p[1] = 1.0 / (1.0 + math.exp(n[1]))
        p[0] = math.log(p[0])
        p[1] = math.log(p[1])
        return p


    """
    Calculate corrected probability using s (training strength) vairable. That
    probability reflects the probability of a class given a word. The corrected
    probability is used to deal with both rare words and to account for the
    random nature of predictive words.
  
    Calculation to be performed:
    Pr'(C|W) = s * Pr(C) + n * Pr(C|W) / s + n

    Pr'(C|W)  corrected probability
    s         strength of background information
    Pr(C)     probability of class
    n         number of occurrences of word in training
    Pr(C|W)   probability of class given word
    """
    def _correctedClassProb(self,class_name,word):
        assert class_name == self.inclass or class_name == self.outclass
        try:
            in_word_freq,out_word_freq = self.word_freqs[word]
            total_word_freq = in_word_freq + out_word_freq
            if class_name == self.inclass:
                class_word_freq = in_word_freq
            else:
                class_word_freq = out_word_freq
        except KeyError:
            total_word_freq = 0.0
            class_word_freq = 0.0
 
        # Probability of class given word
        prCW = class_word_freq / self.class_freqs[class_name]
        assert prCW <= 1.0, "%s %s" % \
              (class_word_freq,self.class_freqs[class_name])

        # Strength variable
        s = self.s
        # Priori probability of class
        prC = self.class_priori[class_name]

        # Number of times the word appeared in training
        n = total_word_freq
    
        return ((s * prC) + (n * prCW)) / (s + n)

    """
    Create an instance of this classifier
    """
    def __init__(self,s=1):
        # s is the strength of the training data
        assert s > 0, "S must have a value greater than 0"
        self.s = s
        self.word_freqs = {}
        self.class_priori = {}
        self.class_freqs = {}
        self.inclass = "Inclass"
        self.outclass = "Outclass"
        self.internal_classifier = linear_model.LinearRegression()

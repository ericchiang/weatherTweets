#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0" 

import math
import sys
import json

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
    def train(self, x, y):
        assert len(x) == len(y), "Bags and intervals have different lengths"
        for ci in y:
           ci = float(ci)
           assert ci >= 0 and ci <= 1, \
                 "Bad confidence interval %f" % (ci,)

        num_bags = len(x)
        for i in range(num_bags):
            if (i + i) % (num_bags / 20) == 0:
                print "%d/%d trained" % (i, num_bags)
            bag_of_words = set(x[i]) # Only care about unique words 
            inclass_prob = float(y[i])
            outclass_prob = 1.0 - inclass_prob

            """
            Because this classifier deals with confidence values rather than
            discrete classes, frequencies of words and classes are computed 
            using the inclass probabilities. 
            """
            self.class_freqs[self.inclass]  += inclass_prob
            self.class_freqs[self.outclass] += outclass_prob
            for word in bag_of_words:
                if word not in self.word_freqs.keys():
                    self.word_freqs[word] = [0.0,0.0]
                self.word_freqs[word][self.inclass]  += inclass_prob
                self.word_freqs[word][self.outclass] += outclass_prob

        # Calculate priori likelihood of inclass and outclass
        self.class_priori[self.inclass]  = \
                                   self.class_freqs[self.inclass] / num_bags
        self.class_priori[self.outclass] = \
                                   self.class_freqs[self.outclass] / num_bags


    """
    Fit a bag of words. Returns inclass and outclass probabilities
    """  
    def fit(self,x):
       y = ['']*len(x)
       for i in range(len(x)):
          bag_of_words = x[i]
          y[i] = self.classifyBag(bag_of_words)
       return y


    """
    Classify a bag of words. Probabilities are calculated in log space to avoid
    float underflow
    """
    def classifyBag(self,bag_of_words):
        p = [0.0,0.0]
        n = [0.0,0.0]
        for word in bag_of_words:
            p_in  = self.correctedClassProb(self.inclass,word)
            p_out = self.correctedClassProb(self.outclass,word)
            assert p_in  < 1.0 and p_in  > 0.0
            assert p_out < 1.0 and p_out > 0.0
            n[self.inclass]  += math.log(1.0 - p_in)  - math.log(p_in)
            n[self.outclass] += math.log(1.0 - p_out) - math.log(p_out)
        p[self.inclass]  = 1.0 / (1.0 + math.exp(n[self.inclass]))
        p[self.outclass] = 1.0 / (1.0 + math.exp(n[self.outclass]))
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
    def correctedClassProb(self,class_index,word):
        if word in self.word_freqs.keys():
            word_freq = self.word_freqs[word]
            total_word_freq = sum(word_freq)
            class_word_freq = word_freq[class_index]
        else:
            total_word_freq = 0
            class_word_freq = 0
 
        # Probability of class given word
        prCW = class_word_freq / self.class_freqs[class_index]
        # Strength variable
        s = self.s
        # Priori probability of class
        prC = self.class_priori[class_index]
        # Number of times the word appeared in training
        n = total_word_freq
    
        return ((s * prC) + (n * prCW)) / (s + n)

    def writeOutModel(self,f):
        model = [self.s, self.word_freqs, self.class_freqs, \
                  self.class_priori, self.outclass, self.inclass]
        f.write(json.dumps(model))

    def readInModel(self,f):
        model = json.loads(f.read())
        self.s            = model[0]
        self.word_freqs   = model[1]
        self.class_freqs  = model[2]
        self.class_priori = model[3]
        self.outclass     = model[4]
        self.inclass      = model[5]

    """
    Create an instance of this classifier
    """
    def __init__(self,s=2):
        # s is the strength of the training data
        assert s > 0, "S must have a value greater than 0"
        self.s = s
        self.word_freqs = {}
        self.class_freqs = [0.0,0.0]
        self.class_priori = [0.0,0.0]
        self.outclass = 0
        self.inclass = 1

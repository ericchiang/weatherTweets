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

    A majority of this function is dedicated to parallelization 
    """
    def fit(self, x, y, num_threads=2,verbose=False,holdout_percent=0.1):

        assert len(x) == len(y), "Bags and intervals have different lengths"
        for i in range(len(y)):
            conf_inter = float(y[i])
            assert conf_inter >= 0.0 and conf_inter <= 1.0
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

        """
        Create (word, inclass frequency, outclass frequency) tuples
        """
        queue = Queue() 
        threads = []
        x_chunks = utils.chunkList(x_train,num_threads)
        y_chunks = utils.chunkList(y_train,num_threads)
        for i in range(num_threads):
            x_chunk = x_chunks[i]
            y_chunk = y_chunks[i]
            t = Thread(target=self._extractWordFreq, args=(x_chunk,y_chunk,queue))
            t.deamon = True
            t.start()
            threads.append(t)

        # Wait for threads to finish
        for t in threads:
            t.join()

        assert queue.qsize() == num_threads

        # Collect results from each sort
        word_freq_tuples = []
        while not queue.empty():
            word_freq_tuples.extend(queue.get())

        if verbose:
            utils.printInfo("Sorting word frequency tuples")

        """
        Sort word frequency tuples so summing of frequencies
        can be done in linear time
        """
        # Sort data
        utils.parallelSort(word_freq_tuples,num_threads)

        if verbose:
            utils.printInfo("Combining word frequency tuples")

        """
        Combine word frequency tuples
        """
        threads =[]
        queue = Queue()
        tuple_chunks = utils.chunkList(word_freq_tuples,num_threads)
        for i in range(num_threads):
            tuple_chunk = tuple_chunks[i]
            t = Thread(target=self._combineWordFreq, \
                       args=(i,tuple_chunks[i],queue))
            t.deamon = True
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        assert queue.qsize() == num_threads

        comb_word_freq_chunks = []
        while not queue.empty():
            comb_word_freq_chunks.append(queue.get())

        comb_word_freq_chunks.sort() # sort by index

        """
        Combine combined chunks. If a word was assigned to two different
        chunks, combine those frequencies.
        """
        combined_word_freqs = []
        last_word = -1
        last_index = -1
        for index,word_freqs in comb_word_freq_chunks:
            assert index > last_index
            last_index = index
            if word_freqs[0][0] == last_word:
                in_word_freq = word_freqs[0][1]  + combined_word_freqs[-1][1]
                out_word_freq = word_freqs[0][2] + combined_word_freqs[-1][2]
                new_word_freq_tuple = (last_word,in_word_freq,out_word_freq)
                combined_word_freqs[-1] = new_word_freq_tuple
                word_freqs = word_freqs[1:]
            last_word = word_freqs[-1][0]
            combined_word_freqs.extend(word_freqs)

        self.word_freqs = combined_word_freqs

        if verbose:
            utils.printInfo("Training internal classifier")

        """
        Train internal classifier
        """
        self.internal_classifier.fit(self._genLogProbs(x_holdout),y_holdout)

    """
    Parallelized calculation method
    """
    def _combineWordFreq(self,index,bag_data,queue):
        combined_bag_data = []
        last_word    = bag_data[0][0]
        in_freq_sum  = bag_data[0][1]
        out_freq_sum = bag_data[0][2]
        for i in range(len(bag_data))[1:]:
            curr_word = bag_data[i][0]
            curr_in_freq  = bag_data[i][1]
            curr_out_freq = bag_data[i][2]
            if curr_word == last_word:
                in_freq_sum += curr_in_freq
                out_freq_sum += curr_out_freq
            else:
                word_freq_tuple = (last_word,in_freq_sum,out_freq_sum)
                combined_bag_data.append(word_freq_tuple)
                last_word = curr_word
                in_freq_sum = curr_in_freq
                out_freq_sum = curr_out_freq
        word_freq_tuple = (last_word,in_freq_sum,out_freq_sum)
        combined_bag_data.append(word_freq_tuple)


        queue.put((index,combined_bag_data))

    """
    Write training data to queue
    """
    def _extractWordFreq(self,x,y,queue):
        assert len(x) == len(y)
        word_freqs = []
        for i in range(len(x)):
            bag = x[i]
            inclass_prob = y[i]
            for word in set(bag):
                inclass_prob = self._probModifier(inclass_prob)
                outclass_prob = 1.0 - inclass_prob
                freq_tuple = (word,inclass_prob,outclass_prob)
                word_freqs.append(freq_tuple)
        queue.put(word_freqs)

    def _probModifier(self,confid_inter):
        assert confid_inter <= 1.0 and confid_inter >= 0.0, \
              "Bad confidence interval: %s" % confid_inter
        confid_inter = (2.0 * confid_inter) - 1.0
        confid_inter = confid_inter ** 3
        confid_inter += 1.0
        return confid_inter / 2.0

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
            assert p_in  < 1.0 and p_in  > 0.0
            assert p_out < 1.0 and p_out > 0.0
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
        word_index = utils.binarySearchByKey(self.word_freqs,word)

        if word_index: # if word is in words
            w,in_word_freq,out_word_freq = self.word_freqs[word_index]
            assert word == w
            total_word_freq = in_word_freq + out_word_freq
            if class_name == self.inclass:
                class_word_freq = in_word_freq
            else:
                class_word_freq = out_word_freq
        else:
            total_word_freq = 0.0
            class_word_freq = 0.0
 
        # Probability of class given word
        prCW = class_word_freq / self.class_freqs[class_name]
        # Strength variable
        s = self.s
        # Priori probability of class
        prC = self.class_priori[class_name]
        # Number of times the word appeared in training
        n = total_word_freq
    
        return ((s * prC) + (n * prCW)) / (s + n)

    def writeOutModel(self,f):
        model = [self.s, self.word_freqs, self.class_priori,
                 self.internal_classifier]
        f.write(json.dumps(model))

    def readInModel(self,f):
        model = json.loads(f.read())
        self.s                   = model[0]
        self.word_freqs          = model[1]
        self.class_priori        = model[2]
        self.internal_classifier = model[3]

    """
    Create an instance of this classifier
    """
    def __init__(self,s=2):
        # s is the strength of the training data
        assert s > 0, "S must have a value greater than 0"
        self.s = s
        self.word_freqs = []
        self.class_priori = {}
        self.class_freqs = {}
        self.inclass = "Inclass"
        self.outclass = "Outclass"
        self.internal_classifier = linear_model.Lasso()

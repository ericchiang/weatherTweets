#!/usr/bin/python
import math
import sys

"""
Naive Bayes algorithm using word information. Rather than numerical features
each observation contains a bag of words which are used to train the algorithm.
"""
class BagOfWordsBayes(object):

  """
  Train the classifier. X is a list containing bags of words. Y are class
  labels to allow supervised learning.
  """
  def train(self, x, y):
    assert len(x) == len(y), "Features and labels have different lengths"
    for class_name in set(y):          # Use set to find unique labels
      self.classes.append(class_name)  # Add class name to classes

    num_classes = len(self.classes)
    self.class_freqs = [0]*num_classes
    self.class_priori = [0.0]*num_classes

    num_input = len(x)

    for i in range(num_input):
      bag_of_words = set(x[i])               # words for each observation
      class_index = self.classes.index(y[i]) # get label index
      self.class_freqs[class_index] += 1
      for word in bag_of_words:              # For each unique word
        try:
          self.word_freqs[word][class_index] += 1
        except:
          self.word_freqs[word] = [0]*num_classes
          self.word_freqs[word][class_index] = 1

    # Calculate priori probabilities for each class
    for class_index in range(num_classes):
      self.class_priori[class_index] = self.class_freqs[class_index] / \
                                        float(num_input)
      assert self.class_priori[class_index] < 1

  """
  Fit a bag of words. Returns predictied classes.
  """  
  def fit(self,x):
    y = ['']*len(x)
    for i in range(len(x)):
      bag_of_words = x[i]
      y[i] = self.classifyBag(bag_of_words)
      print "%d/%d classified" % (i + 1, len(x))
    return y


  """
  Classify a bag of words
  """
  def classifyBag(self,bag_of_words):
    p = [0.0]*len(self.classes) # probability of each class
    for class_index in range(len(self.classes)):
      # Calculate probability of class in log space to avoid float underflow
      n = 0.0
      for word in bag_of_words:
        # Calculate corrected probability of word
        pi = self.correctedClassProb(class_index,word)
        assert pi < 1.0 and pi > 0.0, "[ERROR] pi value is %d" % (pi,)
        n += math.log(1.0 - pi) - math.log(pi)
      p[class_index] = 1.0 / (1 + math.exp(n))
    pred_class_index = p.index(max(p))
    return self.classes[pred_class_index]


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
    prCW = class_word_freq / float(self.class_freqs[class_index])
    # Strength variable
    s = self.s
    # Priori probability of class
    prC = self.class_priori[class_index]
    # Number of times the word appeared in training
    n = total_word_freq

    return ((s * prC) + (n * prCW)) / (s + n)

  """
  Create an instance of a class
  """
  def __init__(self,s=2):
    # s is the strength of the training data
    assert s > 0, "S must have a value greater than 0"
    self.classes = []
    self.s = s
    self.word_freqs = {}
    self.class_freqs = []
    self.class_priori = []

#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

import csv
import string

meta_tags = ['@mention','{link}']
alpha_numeric_word = ['ALPHA_NUMERIC_WORD']
numeric_word = ['NUMERIC_WORD']


"""
Parse CSV file into data matrix
"""
def parseTwitterCSV(tweet_csv):
    data = []
    tweet_reader = csv.reader(tweet_csv, delimiter=',', quotechar='"')
    for row in tweet_reader:
        data.append(row)
    return data

"""
Parse tweet into bag of words
"""
def parseTweet(tweet):
    bag_of_words = []
    for tag in meta_tags:
        tweet = tweet.replace(tag,'')
    tweet = tweet.split()
    for i in range(len(tweet)):
        word = tweet[i].lower()
        while True:
            last_word = word
            word = word.strip(string.punctuation)
            if word == last_word:
                break
        if word:
            bag_of_words.append(word)
    return bag_of_words

      

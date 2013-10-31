#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

import csv
import re
import string

meta_tags = ['@mention','{link}']
alphanumeric_word = 'ALPHANUMERIC_WORD'
numeric_word = 'NUMERIC_WORD'

re_number = re.compile(r'^\d*\.?\d+$')
re_contains_num = re.compile('\d')
punct_no_apost = string.punctuation.replace("'",'')

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
        tweet = tweet.replace(tag,' ')

    for punch_char in punct_no_apost:
        tweet = tweet.replace(punch_char,' ')

    tweet = tweet.split()
    for i in range(len(tweet)):
        word = tweet[i].lower()
        if word:
            if re_number.match(word):
                word = numeric_word
            elif re_contains_num.match(word):
                word = alphanumeric_word
            bag_of_words.append(word)
    return bag_of_words

      

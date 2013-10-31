#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

import unittest
import random
import sys
try:
    sys.path.insert(0,'..')
    from utils import *
except:
    sys.path.insert(0,'.')
    from utils import *

class TestUtilFunctions(unittest.TestCase):

    def test_binarySearch(self):
        m = 10000
        seq = range(m)
        for i in range(m):
            self.assertEqual(i,binarySearch(seq,i))
        self.assertEqual(None,binarySearch(seq,m+1))

    def test_binarySearchByKey(self):
        m = 10000
        seq = []
        rev_seq = range(m)
        rev_seq.reverse()
        for i in range(m):
            seq.append((i,rev_seq[i]))
        for i in range(m):
            index = binarySearchByKey(seq,i)
            self.assertEqual(i,index)
            self.assertEqual(seq[index],(i,rev_seq[i]))

    def test_sortList(self):
        m = 10000
        seq = range(m)
        random.shuffle(seq)
        sortList(seq)
        self.assertEqual(seq,range(m))

    def test_chunkList(self):
        for n in range(1050)[1:]:
            seqs = chunkList(range(1000),n)
            self.assertEqual(len(seqs),n)
            seq = []
            for s in seqs:
                seq.extend(s)
            self.assertEqual(seq,range(1000))

    def test_mergeSortedLists(self):
        m = 1000
        n = 3
        seq = range(m)
        random.shuffle(seq)
        seqs = chunkList(seq,3)
        for seq in seqs:
            sortList(seq)
        seq = mergeSortedLists(seqs)
        self.assertEqual(seq,range(m))


    def test_parallelSort(self):
        m = 10000
        for n in range(20)[1:]:
            seq = range(m)
            random.shuffle(seq)
            parallelSort(seq,n)
            self.assertEqual(seq,range(m)) 


        
if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtilFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)


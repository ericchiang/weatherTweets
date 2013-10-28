#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

from threading import Thread
from datetime import datetime
import math

def sortList(l):
    l.sort()

"""
Heap sort
"""
def sortListByKey(l):
    for i in range(len(l)):
        _addToHeap(l,i)
    heap_size = len(l)
    while heap_size:
        _removeFromHeap(l,heap_size)
        heap_size -= 1

# Heap sort helper function
def _addToHeap(l,i):
    if i == 0:
        return
    item_key = l[i][0]
    parent_i = (i - 1) / 2
    parent_key = l[parent_i][0]
    if item_key > parent_key:
        swap(l,i,parent_i)
        _addToHeap(l,parent_i)

# Heap sort helper function
def _removeFromHeap(l,heap_size):
    new_size = heap_size - 1
    swap(l,0,new_size)
    _pushDownItem(l,0,new_size)
    
# Heap sort helper function
def _pushDownItem(l,i,heap_size):
    item_key = l[i][0]
    c1_index = (2 * i) + 1
    c2_index = 2 * (i + 1)
    if c1_index >= heap_size:
        return
    c1_key = l[c1_index][0]
    if c2_index >= heap_size:
        sawp_key = c1_key
        swap_index = c1_index
    else:
        c2_key = l[c2_index][0]
        if c2_key > c1_key:
            swap_key = c2_key
            swap_index = c2_index
        else:
            swap_key = c1_key
            swap_index = c1_index
    if swap_key > item_key:
        swap(l,i,swap_index)
        _pushDownItem(l,swap_index,heap_size)



def swap(l,i1,i2):
    l[i1],l[i2] = l[i2],l[i1]


def binarySearch(l,item):
    low = 0
    high = len(l) - 1
    while low <= high:
        mid = (low + high) / 2
        mid_item = l[mid]
        if item < mid_item:
            high = mid - 1
        elif item > mid_item:
            low = mid + 1
        else: # item == mid_item
            return mid
    return None

def binarySearchByKey(l,key):
    low = 0
    high = len(l) - 1
    while low <= high:
        mid = (low + high) / 2
        mid_key = l[mid][0]
        if key < mid_key:
            high = mid - 1
        elif key > mid_key:
            low = mid + 1
        else: # key == mid_key
            return mid
    return None


def rootMeanSquaredError(pred,actual):
    return math.sqrt(meanSquaredError(pred,actual))

"""
Calculate mean squared error for a set of values
"""
def meanSquaredError(pred,actual):
    if len(pred) != len(actual):
        raise Exception("Mismatched predicted and actual values")
    n = 0.0
    for i in range(len(pred)):
        n += (float(pred[i]) - float(actual[i]))**2
    return n / len(pred)

"""
Chunk a list into n constituent lists
"""
def chunkList(l,n):
    chunks = []
    chunk_size = int(math.ceil(len(l) / float(n)))
    for i in range(n):
        chunks.append(l[i * chunk_size : (i + 1) * chunk_size])
    return chunks

"""
Parallelized list sorting
"""
def parallelSort(l,num_threads):
    threads = []
    chunked_list = chunkList(l,num_threads)
    for chunk in chunked_list:
        t = Thread(target=sortList, args=(chunk,))
        t.deamon = True
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    mergeSortedLists(chunked_list,write_list=l)


"""
Merge a collection of sorted lists. Lists must not contain item 'None'.
"""
def mergeSortedLists(lists,write_list=[]):
    min_items    = []
    list_indexes = []
    expected_len = 0
    for l in lists:
        min_items.append(l[0])
        list_indexes.append(0)
        expected_len += len(l)

    num_lists = len(lists)

    num_sorted = 0
    while True:
        # Min item beings as None
        min_item = None
        for i in range(len(min_items)):
             item = min_items[i]
             if item is None:
                 continue
             elif min_item is None:
                 min_item = item
                 min_index = i
             elif item < min_item:
                 min_item = item
                 min_index = i

        # If all the items in the min list than the lists have been merged
        if min_item is None:
            break

        if num_sorted >= len(write_list):
            write_list.append(min_item)
        else:
            write_list[num_sorted] = min_item

        num_sorted += 1

        list_indexes[min_index] += 1
        list_index = list_indexes[min_index]
        active_list = lists[min_index]
        try:
            new_item = active_list[list_index]
        except IndexError:
            new_item = None
        min_items[min_index] = new_item

    return write_list

def printInfo(message):
    t = datetime.now().isoformat().replace('T',' ').split('.')[0]
    print '[INFO %s] %s' % (t,message,)

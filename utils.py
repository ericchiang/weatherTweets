from threading import Thread


def sort(l):
    l.sort()


def binarySearch(l,item):
    raise Exception("Binary Search hasn't been implemented")

def parallelSort(l,num_threads)
    threads = []
    chunks = []
    # Parallel sort
    # Chunk for sorting
    chunk_size = int(math.ceil(len(results) / float(num_theads)))
    for i in range(num_threads):
        result_chunk = results[i * chunk_size : (i + 1) * chunk_size]
        chunks.append(result_chunk)
        t = Thread(target=utils.sort, args=(result_chunk))
        t.deamon = True
        t.start()
        threads.append(t)

    return mergeLists(chunks)

"""

"""
def mergeLists(lists):
    max_items    = []
    top_indexes  = []
    merged_list  = []
    expected_len = 0
    for l in lists:
        max_items.append(l[0])
        top_indexes.append(0)
        expected_len += len(l)

    num_lists = len(lists)

    while True:
        i = 0
        max_item = max_items[i]
        max_index = i
        while i < num_lists:
            if max_items[i] > max_item:
                max_item = max_items[i]
                max_index = i
            i += 1
        if max_item is None:
            break
        merged_list.append(max_item)
        top_indexes[max_index] += 1
        top_list = lists[max_index]
        try:
            new_item = top_list[top_indexes[max_index]]
        except IndexError:
            new_item = None
        max_items[max_index] = new_item

    assert len(merged_list) == expected_len

    return merged_list

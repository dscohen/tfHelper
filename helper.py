import math
import time
import datetime
import sys

import numpy as np
import tensorflow as tf

# def listify(input):
#     ti = []
#     for i in input:
#         if type(j) == list or if:
#             ti.append(j)
from tflearn.data_utils import shuffle, pad_sequences


def get_last_timestep(tensor):
    tensor = tf.transpose(tensor,[1,0,2])
    return tf.gather(tensor,tensor.get_shape()[0]-1)

def mvg_avg(n,point,avg):
    if math.isnan(point):
        return point
    return (avg*n + point)/(n+1.)

def read_time(time):
    time_str = ''
    hr = time/(60*60)
    minute = time/60
    sec = time%60
    return map(int,[hr,minute,sec])

def printProgress (iteration, total, t,time, prefix = '', suffix = '', decimals = 1, barLength = 100,elapsed_t = None,loss=None,metric = None,eta = None,feed_dict = None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    time_in = time[0]
    time_out = time[1]
    time = (time_out - time_in) * (total-iteration)
    time = read_time(time)
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = 'X' * filledLength + '-' * (barLength - filledLength)
    loss_val        = loss.eval(feed_dict = feed_dict) if loss is not None else float('nan')
    metric_val      = metric.eval(feed_dict = feed_dict) if metric is not None else float('nan')
    t               = [loss_val,metric_val]
    # t               = [mvg_avg(iteration,loss_val,t[0]), mvg_avg(iteration,metric_val,t[1]),time]
    sys.stdout.write('\r%s |%s| %s%s %s Loss: %.2f Accuracy: %.6f ETA: %s:%s:%s' % (prefix, bar, percents, '%', '',t[0],t[1],time[0],time[1],time[2])),
    sys.stdout.flush()
    # if iteration == total:
    #     sys.stdout.write('\n')
    #     sys.stdout.flush()
    return t


def blockshaped(arr, maxlen):
    """
    Return list of arr [batch X maxlen]

    """

    divide_points = [i * maxlen for i in range(1, int(math.ceil(1. * arr.shape[1] / maxlen)) + 1)]
    old_point = 0
    new_arrs = []
    print divide_points,maxlen
    for div in divide_points:
        new_arrs.append(arr[:, old_point:div])
        old_point = div
    return new_arrs


def prune(lst, pruned=None):
    """
    assumes input of [data,mask]
    :param list:
    :return: pruned data, pruned mask
    """
    lst1 = [[] for s in lst]
    if pruned is not None:
        # 'pruning outside of batch creation and serves as helper function'
        for i, state in enumerate(lst):
            lst1[i] = [state[s] for s in range(len(state)) if s in pruned]

        return lst1
    mask = lst
    counts = np.sum(mask, axis=-1)
    pruned = np.where(counts > 0)
    return mask[pruned[0]], pruned[0]


def next_batch(data_in, targets=None, batch_size=128, random=False, vocab=None, limit=1600, use_vocab=[]):
    """

    :param data_item: list of data to batch, outputs in order of input
    :param targets:
    :param batch_size:
    :param random: not impemented yet, but shuffles the batch each epoch
    :param vocab: Optional arg, if used will convert to embedding index
    :return: for each item in data_in: (inp, mask, lengths,inp1,mask1,len1),TARGETS
    """

    def get_lengths(mask):
        """input is masks [batch x length"""

    def grouper(input):
        """

        :param input: [ [in1,m1],[in12,m12],[in21,m21],[in22,m22] ]
        :return:
        """
        batches = []
        for minibatch in input:
            mini_length_batch = []
            # [[i1,m1],[i12,m12]]
            # [i1,m1]
            mini_length_batch.append([minibatch[0], minibatch[1], np.expand_dims(np.sum(minibatch[1], axis=-1), -1)])
            # [i1,m1,l1]
            batches.append(mini_length_batch)
        return batches

    def sequential_pruning(mask, previous_pruned, create=True):
        for i in previous_pruned:
            mask = mask[i]
        if create:
            new_mask, current_pruned = prune(mask)
        else:
            new_mask = mask
            current_pruned = []
        return new_mask, current_pruned

    def splitter(batch_in, max_len, limit=1e9):
        sliced_arrs = []
        for input in batch_in:
            sliced_arrs.append(blockshaped(input, max_len))
        # minis is (i,m)(i1,m1),...
        minibatches = map(None, sliced_arrs[0], sliced_arrs[1])
        pruned_batches = []
        # use all batches at beginning
        previous_pruned = [i for i in range(len(minibatches[0][0]))]
        pruned_list = []
        pruned_list.append(previous_pruned)
        for inp, mask in minibatches:
            # inp = prune([inp], previous_pruned)
            mask, previous_pruned = sequential_pruning(mask, pruned_list)
            # pruned returns it rapped in list
            pruned_list.append(previous_pruned)
            inp, _ = sequential_pruning(inp, pruned_list, create=False)
            pruned_batches.append([inp, mask])

        return pruned_batches, pruned_list[1:]

    def make_div_2(list_in):
        assert isinstance(list_in, int), 'splitter must run on ints'
        return list_in + 1 if list_in % 2 != 0 else list_in

    assert isinstance(data_in,
                      list), 'deprecated use, assure data_in is in list, even if list if of length 1: [data_in]'
    i = 0
    debug_cnt = 0
    if random: data_in = shuffle(data_in)

    while True:
        if debug_cnt == 2:
            debug_hold = 5

        data_item_output = []
        pruned_list = []
        for z, data_item in enumerate(data_in):
            inp = data_item[i:i + batch_size]
            max_len = max(map(len, inp))
            max_len = make_div_2(max_len)
            if vocab:
                # only convert items we want (documents are embedded already)
                if len(use_vocab) == 0 or use_vocab[z]:
                    vocab.convert_to_embed_idx(inp)
            inp = pad_sequences(inp, maxlen=max_len, padding='post')
            mask = np.ma.make_mask(inp, copy=True, dtype=np.int32).astype(int)
            # lengths = np.expand_dims(np.sum(mask, axis=-1), -1)
            mini_item = [inp, mask]
                # splitter outputs [[in1,m1],[in2,m2]],[[p1],[p2]]
            mini_item, pruned = splitter(mini_item, limit)
            pruned_list += pruned

            data_item_output += mini_item
        data_item_output = grouper(data_item_output)
        output = None
        if targets:
            output = targets[i:i + batch_size]
            data_item_output


        i = (i + batch_size) % len(data_item)
        debug_cnt += 1
        yield data_item_output, output, pruned_list
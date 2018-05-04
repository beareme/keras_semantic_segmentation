# coding=utf-8

import json
import re
import os
import numpy as np
from toolz import itemmap

from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD


PADDING = '<pad>'
UNKNOWN = 'UNK'
EOA = '<eos>'       # end of answer
EOQ = '<eoq>'       # end of question
EXTRA_WORDS_NAMES = [PADDING, UNKNOWN, EOA, EOQ]
EXTRA_WORDS = {PADDING:0, UNKNOWN:1, EOA:2, EOQ:3}
EXTRA_WORDS_ID = itemmap(reversed, EXTRA_WORDS)
MAXLEN = 50

OPTIMIZERS = { \
        'sgd':SGD,
        'adagrad':Adagrad,
        'adadelta':Adadelta,
        'rmsprop':RMSprop,
        'adam':Adam,
        }
###
# Functions
###
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(counter=len(EXTRA_WORDS))
def _myinc(d):
    """
    Gets a tuple d, and returns d[0]: id.
    """
    x = d[0]
    _myinc.counter += 1
    return (x, _myinc.counter - 1)


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        print 'creating directory %s'%directory
        os.makedirs(directory)
    else:
        print "%s already exists!"%directory

def preprocess_line(line):
    cap_tmp = line.strip().decode('utf-8').lower().encode('utf8')
    return cap_tmp

def save_txt_answers(samples, savefile='./sample', whichset='val', step=''):
        with open(savefile + '_' + whichset + '_samples_' + str(step) + '.json', 'w') as f:
            print >>f, '\n'.join(samples)


def save_json_answers(samples, savefile='./sample', whichset='val', step=''):
        with open(savefile + '_' + whichset + '_samples_' + str(step) + '.json', 'w') as f:
            json.dump(samples, f)

def index_sequence(x, word2index):
    """
    Converts list of words into a list of its indices wrt. word2index, that is into
    index encoded sequence.

    In:
        x - list of lines
        word2index - mapping from words to indices

    Out:
        a list of the list of indices that encode the words
    """
    one_hot_x = []
    for line in x:
        line_list = []
        for w in line.split():
            w = w.strip()
            if w in word2index: this_ind = word2index[w]
            else: this_ind = word2index[UNKNOWN]
            line_list.append(this_ind)
        one_hot_x.append(line_list)
    return one_hot_x

def get_num_captions(caption_file):
    """
    From a text file with format:
        <image_id>#<n_cap>
    Gets a list of the number of captions of each <image_id>
    :param caption_file: Text format with the above format
    :return: List of number of captions
    """
    with open (caption_file, 'r') as f:
        id = ''
        n_cap = 0
        n_cap_list =[]
        for line in f:
            caption_id = line.split('\t')[0]
            image_id = caption_id.split('#')[0]
            if id != image_id:
                id = image_id
                n_cap_list.append(n_cap + 1)
            n_cap = int(caption_id.split('#')[1])
        n_cap_list = n_cap_list[1:] + [n_cap + 1]
    return n_cap_list
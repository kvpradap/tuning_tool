import sys
import pandas as pd

sys.path.append('/Users/pradap/Documents/Research/Python-Package/scaling/tuning_tool/')
from downsample.sampler import *
from downsample.tuner import *

from collections import OrderedDict


A = pd.read_csv('../datasets/acm_demo.csv')
B = pd.read_csv('../datasets/dblp_demo.csv')

A.reset_index(inplace=True, drop=True)
B.reset_index(inplace=True, drop=True)

stopwords =  ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
              'he', 'in', 'is', 'it', 'its', 'on', 'that', 'the', 'to', 'of', 'was',
              'were', 'will', 'with']


# def downsample_dk(ltable, rtable, lid, rid, size, y, lstopwords=[], rstopwords=[], nlchunks=1, nrchunks=1, scheduler=threaded.get, compute=True, swap=False):

input_tables = OrderedDict()
input_tables['ltable'] = A
input_tables['rtable'] = B

input_args = OrderedDict()
input_args['lid'] = 'id'
input_args['rid'] = 'id'
input_args['size'] = 100
input_args['y'] = 1
input_args['lstopwords'] = stopwords
input_args['rstopwords'] = stopwords
input_args['compute'] = True
input_args['scheduler'] = threaded.get

param_grid = OrderedDict()
param_grid['nlchunks'] = [1, 2, 4, 8, 16, 32]
param_grid['nrchunks'] = [1, 2, 4, 8, 16, 32]

# dblp['id'] = list(range(len(dblp)))
# acm['id'] = list(range(len(acm)))
#
# l_sample = sample_stratified_length(dblp, 'id')
# stopwords =  ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
#               'he', 'in', 'is', 'it', 'its', 'on', 'that', 'the', 'to', 'of', 'was',
#               'were', 'will', 'with']
# r_sample = sample_stratified_probelen(acm, l_sample, 'id', 0.1, stopwords, stopwords)
# print('hi')

tune_for_ds(dblp, acm, )
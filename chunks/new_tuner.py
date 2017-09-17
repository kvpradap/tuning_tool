import pandas as pd
import string
import math
import numpy as np

import dmagellan
from dmagellan.utils.py_utils.utils import build_inv_index, tokenize_strings
from dmagellan.tokenizer.whitespacetokenizer import WhiteSpaceTokenizer
from dmagellan.utils.cy_utils.stringcontainer import StringContainer
from dmagellan.blocker.overlap.overlapblocker import OverlapBlocker

def remove_stopwords(tokens, stopwords):
    out_tokens = []
    for token in tokens:
        if not stopwords.has_key(token):
            out_tokens.append(token)
    return out_tokens

def process_column(column, stop_words):
    column = column.str.translate(None, string.punctuation)
    column = column.str.lower()
    if stop_words:
        dict_stopwords = dict(zip(self.stop_words, [0] * len(self.stop_words)))
        partial_rm_stopwords_fn = partial(remove_stopwords,
                                          stopwords=dict_stopwords)
        column = column.str.split().map(partial_rm_stopwords_fn).str.join(' ')
    return column


def sample_ltable(ltable, lid, l_block_attr, nbins, sample_proportion, seed=0,
                  stopwords=[]):
    ltbl = ltable[[lid, l_block_attr]]
    ltbl = ltbl[~ltbl[l_block_attr].isnull()]
    #process columns
    ltbl[l_block_attr] = process_column(ltbl[l_block_attr], stopwords)
    n = int(math.floor(sample_proportion*len(ltable)))


    # get the string lengths
    ltbl['str_len'] = ltbl.title.str.len()
    groups = ltbl.groupby('str_len')

    len_ids = {}
    for gid, g in groups:
        len_ids[gid] = list(g[lid].values)
    strlens = list(ltbl['str_len'].values)
    strlens += [max(strlens) + 1]

    freq, edges = np.histogram(strlens, bins=nbins)

    bins = [[] for _ in range(nbins)]
    keys = sorted(len_ids.keys())
    positions = np.digitize(keys, edges)

    for i in range(len(keys)):
        k, p = keys[i], positions[i]
        bins[p - 1].extend(len_ids[k])
    len_bins = [len(bins[i]) for i in range(len(bins))]

    weights = [len_bins[i] / float(sum(len_bins)) for i in range(len(bins))]
    numtups = [int(math.ceil(weights[i] * n)) for i in range(len(weights))]
    # numtups, sum(numtups)
    sampled = []
    for i in range(len(bins)):
        nt = numtups[i]
        np.random.seed(0)
        if len(bins[i]):
            np.random.seed(seed)
            tmp = np.random.choice(bins[i], nt)
            if len(tmp):
                sampled.extend(tmp)
    ltable.set_index(lid, inplace=True, drop=False)
    ltable['_pos'] = list(range(len(ltable)))
    s_ltable = ltable.loc[sampled]
    s_ltable = s_ltable.sort_values(['_pos'])
    s_ltable.reset_index(drop=True, inplace=True)
    s_ltable.drop(['_pos'], axis=1, inplace=True)
    ltable.drop(['_pos'], axis=1, inplace=True)
    return s_ltable

# def process_ltable(table, id, block_attr, tok, stopwords):
#     ob = OverlapBlocker()
#     p = ob.process_and_tokenize_ltable(table, id, block_attr, tok, stopwords)
#     inv_index = build_inv_index([p])


def sample_rtable(rtable, rid, r_block_attr, tok, nbins, sample_proportion, inv_index,
                  seed=0,
                  stopwords=[]):
    ob = OverlapBlocker()
    # rtbl = rtable.reset_index(drop=True)
    rtbl = rtable[[rid, r_block_attr]]
    # rtbl['_pos'] = list(range(len(tbl)))
    p = ob.process_and_tokenize_ltable(rtbl, rid, r_block_attr, tok, stopwords)

    tok_cnt = {}
    tok_map = {}
    for i in range(p.size()):
        tid, tokens = p.get(i)
        cnt = 0
        for tok in tokens:
            if tok not in tok_map:
                tok_map[tok] = len(inv_index.values(tok))
            cnt += tok_map[tok]
        tok_cnt[tid] = cnt
    df =  pd.DataFrame(tok_cnt.items(), columns=['id', 'count'])
    groups = df.groupby('count')
    cnt_ids = {}
    for gid, g in groups:
        cnt_ids[gid] = list(g[lid].values)

    cnts = list(df['count'].values)
    cnts += [max(cnts) + 1]
    freq, edges = np.histogram(cnts, bins=nbins)
    n = int(math.floor(sample_proportion * len(rtable)))
    bins = [[] for _ in range(nbins)]
    keys = sorted(cnt_ids.keys())
    positions = np.digitize(keys, edges)

    for i in range(len(keys)):
        k, p = keys[i], positions[i]
        bins[p - 1].extend(cnt_ids[k])
    len_bins = [len(bins[i]) for i in range(len(bins))]

    weights = [len_bins[i] / float(sum(len_bins)) for i in range(len(bins))]
    numtups = [int(math.ceil(weights[i] * n)) for i in range(len(weights))]

    sampled = []
    for i in range(len(bins)):
        nt = numtups[i]
        np.random.seed(seed)
        if len(bins[i]):
            tmp = np.random.choice(bins[i], nt)
            if len(tmp):
                sampled.extend(tmp)
    rtable['_pos'] = list(range(len(rtable)))
    rtable.set_index(rid, inplace=True, drop=False)
    s_rtable = rtable.loc[sampled]
    s_rtable = s_rtable.sort_values('_pos')
    s_rtable.drop(['_pos'], axis=1, inplace=True)
    rtable.drop(['_pos'], axis=1, inplace=True)
    return s_rtable







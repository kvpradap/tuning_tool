import pandas as pd
import numpy as np
import math
import string
from dmagellan.utils.py_utils.utils import get_str_cols, str2bytes, sample, split_df, \
    tokenize_strings_wsp, build_inv_index, get_stopwords_for_downsample
from dmagellan.blocker.overlap.overlapblocker import OverlapBlocker
from dmagellan.tokenizer.whitespacetokenizer import WhiteSpaceTokenizer
from dmagellan.sampler.downsample.downsample import preprocess_table

def sample_stratified_length(table, idcol, lencol='strlen', sample_proportion=0.1,
                             nbins=10, seed=0):
    df = concat_string_attrs_comp_len(table, idcol, lencol)
    sampled = stratify_on_len(table, df, idcol, lencol, sample_proportion, nbins,
                              seed=seed)
    return sampled


def stratify_on_len(table, stat_table, idcol, lencol, sample_proportion, nbins, seed):
    groups = stat_table.groupby(lencol)
    len_ids = {}
    for gid, g in groups:
        len_ids[gid] = list(g[idcol].values)
    strlens = list(stat_table[lencol].values)
    strlens += [max(strlens) + 1]
    freq, edges = np.histogram(strlens, bins=nbins)
    n = int(math.floor(sample_proportion * len(table)))
    bins = [[] for _ in range(nbins)]
    keys = sorted(len_ids.keys())
    positions = np.digitize(keys, edges)
    for i in range(len(keys)):
        k, p = keys[i], positions[i]
        bins[p - 1].extend(len_ids[k])
    len_bins = [len(bins[i]) for i in range(len(bins))]
    weights = [len_bins[i] / float(sum(len_bins)) for i in range(len(bins))]
    numtups = [int(math.ceil(weights[i] * n)) for i in range(len(weights))]

    sampled = []
    for i in range(len(bins)):
        nt = numtups[i]
        if len(bins[i]):
            np.random.seed(seed)
            tmp = np.random.choice(bins[i], nt)
            if len(tmp):
                sampled.extend(tmp)
    table.set_index(idcol, inplace=True, drop=False)
    table['_pos'] = list(range(len(stat_table)))
    s_ltable = table.loc[sampled]
    s_ltable = s_ltable.sort_values(['_pos'])
    s_ltable.reset_index(drop=True, inplace=True)
    s_ltable.drop(['_pos'], axis=1, inplace=True)
    return s_ltable

def concat_string_attrs_comp_len(table, idcol, lencol):
    strcols = list(get_str_cols(table))
    strcols.append(idcol)
    projdf = table[strcols]
    t_dict = {}
    for row in projdf.itertuples():
        colvalues = row[1:-1]
        uid = row[-1]
        strings = [colvalue.strip() for colvalue in colvalues if not pd.isnull(colvalue)]
        concatrow = ' '.join(strings).lower()
        concatrow = concatrow.translate(None, string.punctuation)
        t_dict[uid] = len(concatrow)

    return pd.DataFrame(t_dict.items(), columns=[idcol, lencol])
##########
def concat_strings(table, idcol, concatcol):
    strcols = list(get_str_cols(table))
    strcols.append(idcol)
    projdf = table[strcols]
    t_dict = {}
    for row in projdf.itertuples():
        colvalues = row[1:-1]
        uid = row[-1]
        strings = [colvalue.strip() for colvalue in colvalues if not pd.isnull(colvalue)]
        concatrow = ' '.join(strings).lower()
        concatrow = concatrow.translate(None, string.punctuation)
        t_dict[uid] = concatrow

    return pd.DataFrame(t_dict.items(), columns=[idcol, concatcol])



def sample_stratified_probelen(table, othertable, idcol, oidcol,
                               sample_proportion=10, lstopwords=[],
                               rstopwords=[],
                               lenprobes='probelen',
                               nbins=10,
                               seed=0):
    df = concat_strings_comp_probelen(table, othertable, oidcol, lenprobes, lstopwords,
                                      rstopwords)
    s_table = stratify_on_probelen(table, df, idcol, lenprobes, sample_proportion, nbins,
                                   seed)
    return s_table


def concat_strings_comp_probelen(table, othertable, idcol, lenprobes, lstopwords,
                                 rstopwords):
    tok = WhiteSpaceTokenizer()
    concatcol = 'concatcol'
    odf = concat_strings(othertable, idcol, concatcol)
    ob = OverlapBlocker()
    p = ob.process_and_tokenize_ltable(odf, idcol, concatcol,
                                       tok, lstopwords)
    inv_index = build_inv_index([p])
    df = concat_strings(table, idcol, concatcol)
    p = ob.process_and_tokenize_ltable(df, idcol, concatcol, tok, rstopwords)
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
    df = pd.DataFrame(tok_cnt.items(), columns=[idcol, lenprobes])
    return df


def stratify_on_probelen(table, stat_table, idcol, probelen, sample_proportion, nbins,
                         seed):
    groups = stat_table.groupby(probelen)
    cnt_ids = {}
    for gid, g in groups:
        cnt_ids[gid] = list(g[idcol].values)
    cnts = list(stat_table[probelen].values)
    cnts += [max(cnts) + 1]
    freq, edges = np.histogram(cnts, bins=nbins)
    n = int(math.floor(sample_proportion * len(stat_table)))
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
    table['_pos'] = list(range(len(stat_table)))
    table.set_index(idcol, inplace=True, drop=False)
    s_rtable = table.loc[sampled]
    s_rtable = s_rtable.sort_values('_pos')
    s_rtable.drop(['_pos'], axis=1, inplace=True)
    # rtable.drop(['_pos'], axis=1, inplace=True)
    return s_rtable
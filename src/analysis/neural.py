import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn import metrics
from itertools import product
from utils.utils import to_sqnp, to_np, to_sqpth, to_pth
from analysis import compute_stats
from models.DND import compute_similarities, transform_similarities


def compute_trsm(activation_tensor):
    """compute TR-TR neural similarity for the input tensor

    Parameters
    ----------
    activation_tensor : 3d array, (n_examples, n_timepoints, n_dim)
        neural activity

    Returns
    -------
    2d array, (n_timepoints, n_timepoints)
        similarity array

    """
    n_examples, n_timepoints, n_dim = np.shape(activation_tensor)
    trsm_ = np.zeros((n_timepoints, n_timepoints))
    for data_i_ in activation_tensor:
        trsm_ += np.corrcoef(data_i_)
    return trsm_ / n_examples


def compute_cell_memory_similarity(
        C, V, inpt, leak, comp,
        kernel='cosine', recall_func='LCA'
):
    n_examples, n_timepoints, n_dim = np.shape(C)
    n_memories = len(V[0])
    # prealloc
    sim_raw = np.zeros((n_examples, n_timepoints, n_memories))
    sim_lca = np.zeros((n_examples, n_timepoints, n_memories))
    for i in range(n_examples):
        # compute similarity
        for t in range(n_timepoints):
            # compute raw similarity
            sim_raw[i, t, :] = to_np(compute_similarities(
                to_pth(C[i, t]), V[i], kernel))
            # compute LCA similarity
            sim_lca[i, t, :] = transform_similarities(
                to_pth(sim_raw[i, t, :]), recall_func,
                leak=to_pth(leak[i, t]),
                comp=to_pth(comp[i, t]),
                w_input=to_pth(inpt[i, t])
            )
    return sim_raw, sim_lca


def create_sim_dict(sim, cond_ids, n_targ=1):
    """split data according to condition, and target vs. lures

    Parameters
    ----------
    sim : np array
        output of `compute_cell_memory_similarity`
    cond_ids : dict
        trial type info in the form of {cond_name: condition_id_array}
    n_targ : int
        number of target memories, assumed to "sit at the end"

    Returns
    -------
    dict
        similatity values

    """
    sim_dict_ = {cn: sim[cids] for cn, cids in cond_ids.items()}
    sim_dict = {cn: {'targ': None, 'lure': None} for cn in cond_ids.keys()}
    sim_dict['NM']['lure'] = sim_dict_['NM']
    for cn in ['RM', 'DM']:
        sim_dict[cn]['targ'] = np.atleast_3d(sim_dict_[cn][:, :, -n_targ:])
        sim_dict[cn]['lure'] = np.atleast_3d(sim_dict_[cn][:, :, :-n_targ])
    return sim_dict


def compute_cell_memory_similarity_stats(sim_dict, cond_ids):
    # re-organize as hierachical dict
    sim_stats = {cn: {'targ': {}, 'lure': {}} for cn in cond_ids.keys()}
    # for DM, RM conditions...
    for cond in ['DM', 'RM']:
        # compute stats for target & lure activation
        for m_type in sim_stats[cond].keys():
            s_ = compute_stats(np.mean(sim_dict[cond][m_type], axis=-1))
            sim_stats[cond][m_type]['mu'], sim_stats[cond][m_type]['er'] = s_
    # for NM trials, only compute lure activations, since there is no target
    s_ = compute_stats(np.mean(sim_dict['NM']['lure'], axis=-1))
    sim_stats['NM']['lure']['mu'], sim_stats['NM']['lure']['er'] = s_
    return sim_stats


def compute_roc(distrib_noise, distrib_signal):
    """compute ROC given the two distribributions
    assuming the distributions are the output of np.histogram

    example:
    dist_l, _ = np.histogram(acts_l, bins=n_bins, range=histrange)
    dist_r, _ = np.histogram(acts_r, bins=n_bins, range=histrange)
    tprs, fprs = compute_roc(dist_l, dist_r)

    Parameters
    ----------
    distrib_noise : 1d array
        the noise distribution
    distrib_signal : 1d array
        the noise+signal distribution

    Returns
    -------
    1d array, 1d array
        the roc curve: true positive rate, and false positive rate

    """
    # assert len(distrib_noise) == len(distrib_signal)
    # assert np.sum(distrib_noise) == np.sum(distrib_signal)
    n_pts = len(distrib_noise)
    tpr, fpr = np.zeros(n_pts), np.zeros(n_pts)
    # slide the decision boundary from left to right
    for b in range(n_pts):
        fn, tp = np.sum(distrib_signal[:b]), np.sum(distrib_signal[b:])
        tn, fp = np.sum(distrib_noise[:b]), np.sum(distrib_noise[b:])
        # calculate TP rate and FP rate
        tpr[b] = tp / (tp + fn)
        fpr[b] = fp / (tn + fp)
    return tpr, fpr


def compute_auc_over_time(
        acts_l, acts_r,
        n_bins=100, histrange=(0, 1)
):
    """compute roc, auc, over time
    - given the activity for the two conditions
    - compute roc, auc for all time points
    *depends on analysis.neural.compute_roc()

    Parameters
    ----------
    acts_l : 2d array, (T x n_examples)
        the left distribution
    acts_r : 2d array, (T x n_examples)
        the right distribution
    n_bins : int
        histogram bin
    histrange : 2-tuple
        histogram range

    Returns
    -------
    arrays
        roc, auc, over time

    """
    event_len, n_examples = np.shape(acts_l)
    # compute fpr, tpr
    tprs = np.zeros((event_len, n_bins))
    fprs = np.zeros((event_len, n_bins))
    for t in range(event_len):
        # compute the bin counts for each condition
        dist_l, _ = np.histogram(acts_l[t, :], bins=n_bins, range=histrange)
        dist_r, _ = np.histogram(acts_r[t, :], bins=n_bins, range=histrange)
        tprs[t], fprs[t] = compute_roc(dist_l, dist_r)
    # compute area under roc curves
    auc = [metrics.auc(fprs[t], tprs[t]) for t in range(event_len)]
    return tprs, fprs, auc

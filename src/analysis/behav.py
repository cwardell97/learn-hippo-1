import numpy as np
import torch
from scipy.stats import sem
from utils.utils import to_sqnp, to_np


def compute_acc(Y, log_dist_a, n_se=2, return_er=False):
    """compute the accuracy of the prediction, over time
    - optionally return the standard error
    - assume n_action == y_dim + 1

    Parameters
    ----------
    Y : 3d tensor
        [n_examples, T_total, y_dim]
    log_dist_a : 3d tensor
        [n_examples, T_total, n_action]
    n_se : int
        number of SE
    return_er : bool
        whether to return SEs

    Returns
    -------
    1d array(s)
        stats for state prediction accuracy

    """
    # argmax the action distribution (don't know unit included)
    argmax_dist_a = np.argmax(log_dist_a, axis=2)
    # argmax the targets one hot vecs
    argmax_Y = np.argmax(to_np(Y), axis=2)
    # compute matches
    corrects = argmax_Y == argmax_dist_a
    # compute stats across trials
    acc_mu_ = np.mean(corrects, axis=0)
    acc_er_ = sem(corrects, axis=0) * n_se
    if return_er:
        return acc_mu_, acc_er_
    return acc_mu_


def compute_mistake(Y, log_dist_a, n_se=2, return_er=False):
    # argmax the action distribution (don't know unit included)
    argmax_dist_a = np.argmax(log_dist_a, axis=2)
    # argmax the targets one hot vecs
    argmax_Y = np.argmax(to_np(Y), axis=2)
    # compute the difference
    diff = argmax_Y != argmax_dist_a
    # get don't knows
    dk = _compute_dk(log_dist_a)
    # mistake := different from target and not dk
    mistakes = np.logical_and(diff, ~dk)
    # compute stats across trials
    mis_mu_ = np.mean(mistakes, axis=0)
    mis_er_ = sem(mistakes, axis=0) * n_se
    if return_er:
        return mis_mu_, mis_er_
    return mis_mu_


def compute_dk(log_dist_a, n_se=2, return_er=False):
    """compute P(don't know) over time
    - optionally return the standard error
    - assume don't know is the last action dimension
    - assume n_action == y_dim + 1

    Parameters
    ----------
    log_dist_a : 3d tensor
        [n_examples, T_total, n_action]
    n_se : int
        number of SE
    return_er : bool
        whether to return SEs

    Returns
    -------
    1d array(s)
        stats for P(don't know)

    """
    # get don't know actions
    dk = _compute_dk(log_dist_a)
    # compute stats
    dk_mu_ = np.mean(dk, axis=0)
    dk_er_ = sem(dk, axis=0) * n_se
    if return_er:
        return dk_mu_, dk_er_
    return dk_mu_


def _compute_dk(log_dist_a):
    # compute the dk dim
    a_dim = np.shape(log_dist_a)[-1]
    dk_id = a_dim-1
    # argmax to get the responses
    argmax_dist_a = np.argmax(log_dist_a, axis=2)
    # get don't know actions
    dk = argmax_dist_a == dk_id
    return dk


def average_by_part(time_course, p):
    """take average within each part of the (multi-part) sequence

    Parameters
    ----------
    time_course : 1d array
        a sequence of values; e.g. accuracy
    p : the param class
        simulation parameters

    Returns
    -------
    list
        a list of averaged values

    """
    return [np.mean(time_course[get_tps_for_ith_part(ip, p.env.tz.T_part)])
            for ip in range(p.env.tz.n_mvs)]


def get_tps_for_ith_part(ip, T_part):
    """get the time range (a list of time points) for the i-th movie part

    Parameters
    ----------
    ip : int
        the index of movie part
    T_part : int
        the length of one movie part

    Returns
    -------
    1d array
        a range of time points

    """
    return np.arange(T_part*ip, T_part*(ip+1))


def entropy(probs):
    """calculate entropy.
    I'm using log base 2!

    Parameters
    ----------
    probs : a torch vector
        a prob distribution

    Returns
    -------
    torch scalar
        the entropy of the distribution

    """
    return - torch.stack([pi * torch.log2(pi) for pi in probs]).sum()


def compute_behav_metrics(Y, log_dist_a, p, average_bp=True):
    # compute performance
    acc_mu_ = compute_acc(Y, log_dist_a)
    mis_mu_ = compute_mistake(Y, log_dist_a)
    dk_mu_ = compute_dk(log_dist_a)
    if average_bp:
        # split by movie parts
        acc_mu_parts = average_by_part(acc_mu_, p)
        mis_mu_parts = average_by_part(mis_mu_, p)
        dk_mu_parts = average_by_part(dk_mu_, p)
        return acc_mu_parts, mis_mu_parts, dk_mu_parts
    return acc_mu_, mis_mu_, dk_mu_


def get_baseline(T, chance):
    """compute the observation-only (no memory) baseline performance

    Parameters
    ----------
    T : int
    event length
    chance : float [0,1]
    chance performance, 1 / n branches

    Returns
    -------
    np.array (T+1,)
    baseline performance accuracy

    """
    return np.array([chance * (T-t)/T + t/T for t in range(T+1)])

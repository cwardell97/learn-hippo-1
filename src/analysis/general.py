import torch
import numpy as np
from scipy.stats import sem


def compute_stats(matrix, axis=0, n_se=2):
    """compute mean and errorbar w.r.t to SE

    Parameters
    ----------
    matrix : type
        Description of parameter `matrix`.
    axis : type
        Description of parameter `axis`.
    n_se : type
        Description of parameter `n_se`.

    Returns
    -------
    type
        Description of returned object.

    """
    mu_ = np.mean(matrix, axis=axis)
    er_ = sem(matrix, axis=axis) * n_se
    return mu_, er_


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


def cosine_similarity(u, v):
    """compute the cosine similarity of two vectors

    Parameters
    ----------
    u : type
        Description of parameter `u`.
    v : type
        Description of parameter `v`.

    Returns
    -------
    scalar
        cosine similarity

    """
    return u @ v / np.linalg.norm(u) / np.linalg.norm(v)


def compute_stats_max(matrix, axis=0, n_se=2):
    """compute mean and errorbar w.r.t to SE

    Parameters
    ----------
    matrix : type
        Description of parameter `matrix`.
    axis : type
        Description of parameter `axis`.
    n_se : type
        Description of parameter `n_se`.

    Returns
    -------
    type
        Description of returned object.

    """
    mu_ = np.amax(matrix, axis=axis)
    er_ = sem(matrix, axis=axis) * n_se
    return mu_, er_


def n_epoch_inpt_calc(inpts, sems, n, axis=0):
    """compute mean inpt gate activation curve of first and last n epochs

    Parameters
    ----------
    matrix : type
        Description of parameter `matrix`.
    axis : type
        Description of parameter `axis`.
    n : type
        number of first and last epochs to be grabbed.

    Returns
    -------
    type
        Description of returned object.

    """
    mu_first = np.mean(inpts[:n,:], axis=axis)
    er_first = np.mean(sems[:n,:], axis=axis)

    mu_last = np.mean(inpts[-n:,:], axis=axis)
    er_last = np.mean(sems[-n:,:], axis=axis)


    return mu_first, er_first, mu_last, er_last

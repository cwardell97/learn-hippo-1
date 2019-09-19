import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import dabest

from collections import defaultdict
# from itertools import product
# from scipy.stats import pearsonr
from task import SequenceLearning
from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, load_ckpt, pickle_load_dict, \
    get_test_data_dir, get_test_data_fname
from analysis import compute_acc, compute_dk, compute_stats, \
    compute_trsm, compute_cell_memory_similarity, create_sim_dict, \
    compute_auc_over_time, compute_event_similarity, batch_compute_true_dk, \
    process_cache, get_trial_cond_ids, compute_n_trials_to_skip,\
    compute_cell_memory_similarity_stats, sep_by_qsource, prop_true, \
    get_qsource, trim_data, make_df

# from vis import plot_pred_acc_full, plot_pred_acc_rcl, get_ylim_bonds,\
#     plot_time_course_for_all_conds
from matplotlib.ticker import FormatStrFormatter
# from sklearn.decomposition.pca import PCA
from itertools import combinations
from scipy.special import comb
from brainiak.funcalign.srm import SRM
# plt.switch_backend('agg')

sns.set(style='white', palette='colorblind', context='talk')

log_root = '../log/'
# exp_name = 'penalty-random-continuous'
# subj_ids = np.arange(6)
exp_name = 'penalty-fixed-discrete-simple_'
subj_ids = np.arange(10)
n_subjs = len(subj_ids)

supervised_epoch = 300
epoch_load = 600
learning_rate = 1e-3

n_branch = 3
n_param = 16
enc_size = 16
n_event_remember = 2
def_prob = None

n_hidden = 194
n_hidden_dec = 128
eta = .1

penalty_random = 0
# testing param, ortho to the training directory
penalty_discrete = 1
penalty_onehot = 0
normalize_return = 1

# loading params
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = .3
pad_len_load = -1
penalty_train = 4
# testing params
p_test = 0
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test
pad_len_test = 0
penalty_test = 4

n_examples_test = 256
fix_cond = 'RM'
scramble = False

srt_dict = {'control': None, 'patient': np.arange(n_param)}
group_names = list(srt_dict.keys())
# CM = {gn: [None] * n_subjs for gn in group_names}
# DA = {gn: [None] * n_subjs for gn in group_names}
CM = defaultdict(list)
DA = defaultdict(list)

for g_name, srt in srt_dict.items():
    print(f'\ngroup_name: {g_name}')
    for subj_id in subj_ids:
        print(f'{subj_id} ', end='')

        np.random.seed(subj_id)
        p = P(
            exp_name=exp_name, sup_epoch=supervised_epoch,
            n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
            enc_size=enc_size, n_event_remember=n_event_remember,
            penalty=penalty_train, penalty_random=penalty_random,
            penalty_onehot=penalty_onehot, penalty_discrete=penalty_discrete,
            normalize_return=normalize_return,
            p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
            n_hidden=n_hidden, n_hidden_dec=n_hidden_dec,
            lr=learning_rate, eta=eta,
        )
        # init env
        task = SequenceLearning(
            n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
            p_rm_ob_enc=p_rm_ob_enc_test, p_rm_ob_rcl=p_rm_ob_rcl_test,
        )
        # create logging dirs
        log_path, log_subpath = build_log_path(
            subj_id, p, log_root=log_root, verbose=False
        )

        test_params = [penalty_test, pad_len_test, srt]
        test_data_dir, test_data_subdir = get_test_data_dir(
            log_subpath, epoch_load, test_params)
        test_data_fname = get_test_data_fname(
            n_examples_test, fix_cond, scramble)
        fpath = os.path.join(test_data_dir, test_data_fname)

        test_data_dict = pickle_load_dict(fpath)
        results = test_data_dict['results']
        XY = test_data_dict['XY']

        [dist_a_, Y_, log_cache_, log_cond_] = results
        [X_raw, Y_raw] = XY

        # compute ground truth / objective uncertainty (delay phase removed)
        true_dk_wm_, true_dk_em_ = batch_compute_true_dk(X_raw, task)

        '''precompute some constants'''
        # figure out max n-time-steps across for all trials
        T_part = n_param + pad_len_test
        T_total = T_part * task.n_parts
        #
        n_conds = len(TZ_COND_DICT)
        memory_types = ['targ', 'lure']
        ts_predict = np.array(
            [t % T_part >= pad_len_test for t in range(T_total)])

        '''organize results to analyzable form'''
        # skip examples untill EM is full
        n_examples_skip = n_event_remember
        n_examples = n_examples_test - n_examples_skip
        data_to_trim = [
            dist_a_, Y_, log_cond_, log_cache_, true_dk_wm_, true_dk_em_
        ]
        [dist_a, Y, log_cond, log_cache, true_dk_wm, true_dk_em] = trim_data(
            n_examples_skip, data_to_trim)
        # process the data
        cond_ids = get_trial_cond_ids(log_cond)
        activity_, ctrl_param_ = process_cache(log_cache, T_total, p)
        [C_, H_, M_, CM_, DA_, V_] = activity_
        CM[g_name].append(CM_)
        DA[g_name].append(DA_)
        # [inpt, leak, comp] = ctrl_param_

'''analysis'''
# from sklearn.preprocessing import StandardScaler
dim_srm = 64
test_prop = .5
n_examples_tr = int(n_examples * (1-test_prop))
n_examples_te = n_examples-n_examples_tr

# data = CM
data = DA

# reshape data to fit SRM
_, _, nTR, nH = np.shape(data[g_name])
X_train = []
X_test = []
X_intercepts = []
for g_name in group_names:
    print(f'\n{g_name}: {np.shape(data[g_name])}', end='')
    for subj_id in subj_ids:
        d_tr_ = data[g_name][subj_id][:n_examples_tr, :, :].reshape((-1, nH)).T
        d_te_ = data[g_name][subj_id][n_examples_tr:, :, :].reshape((-1, nH)).T
        X_intercepts.append(np.mean(d_tr_, axis=1))
        X_train.append(d_tr_ - np.mean(X_intercepts[-1]))
        X_test.append(d_te_ - np.mean(X_intercepts[-1]))

srm = SRM(features=dim_srm)
srm.fit(X_train)
X_test_srm_ = np.array(srm.transform(X_test))
X_test_srm_ = X_test_srm_.reshape(
    (len(X_test_srm_), dim_srm, n_examples_te, nTR)
)

X_test_srm = {gn: None for gn in group_names}
X_test_srm['control'] = X_test_srm_[:n_subjs]
X_test_srm['patient'] = X_test_srm_[n_subjs:]


# compute t/s ISC
n_subj_pairs = int(comb(n_subjs, 2))
sisc = defaultdict(list)
tisc = defaultdict(list)

# with-condition
for g_name in group_names:
    for k in range(n_examples_te):
        sisc_g_k_diag = np.zeros((n_subj_pairs, dim_srm))
        tisc_g_k_diag = np.zeros((n_subj_pairs, n_param))
        # tisc_g_k_diag = np.zeros((n_subj_pairs, nTR))
        for (i_comb, (i_s, j_s)) in enumerate(combinations(range(n_subjs), 2)):
            # compute sptial isc
            sisc_g_k_ij = np.corrcoef(
                X_test_srm[g_name][i_s, :, k, n_param:],
                X_test_srm[g_name][j_s, :, k, n_param:]
            )[dim_srm:, :dim_srm]
            # compute time isc
            tisc_g_k_ij = np.corrcoef(
                X_test_srm[g_name][i_s, :, k, n_param:].T,
                X_test_srm[g_name][j_s, :, k, n_param:].T
            )[n_param:, :n_param]
            # collect all subj pairs
            sisc_g_k_diag[i_comb] = np.diag(sisc_g_k_ij)
            tisc_g_k_diag[i_comb] = np.diag(tisc_g_k_ij)
        # collect the k-th example
        sisc[g_name, g_name].append(sisc_g_k_diag)
        tisc[g_name, g_name].append(tisc_g_k_diag)


# between the two condition
# with in a condition
for k in range(n_examples_te):
    sisc_g_k_diag = np.zeros((n_subj_pairs, dim_srm))
    tisc_g_k_diag = np.zeros((n_subj_pairs, n_param))
    for (i_comb, (i_s, j_s)) in enumerate(combinations(range(n_subjs), 2)):
        # compute sptial isc
        sisc_g_k_ij = np.corrcoef(
            X_test_srm['patient'][i_s, :, k, n_param:],
            X_test_srm['control'][j_s, :, k, n_param:]
        )[dim_srm:, :dim_srm] + np.corrcoef(
            X_test_srm['patient'][j_s, :, k, n_param:],
            X_test_srm['control'][i_s, :, k, n_param:]
        )[dim_srm:, :dim_srm]
        # compute time isc
        tisc_g_k_ij = np.corrcoef(
            X_test_srm['patient'][i_s, :, k, n_param:].T,
            X_test_srm['control'][j_s, :, k, n_param:].T
        )[n_param:, :n_param] + np.corrcoef(
            X_test_srm['patient'][j_s, :, k, n_param:].T,
            X_test_srm['control'][i_s, :, k, n_param:].T
        )[n_param:, :n_param]
        # collect all subj pairs
        sisc_g_k_diag[i_comb] = np.diag(sisc_g_k_ij) / 2
        tisc_g_k_diag[i_comb] = np.diag(tisc_g_k_ij) / 2
    # collect the k-th example
    sisc['control', 'patient'].append(sisc_g_k_diag)
    tisc['control', 'patient'].append(tisc_g_k_diag)


'''compute stats'''

sisc_ddict = {}
tisc_ddict = {}
for i, g_name_i in enumerate(group_names):
    for j, g_name_j in enumerate(group_names):
        if i <= j:
            msg = f'{g_name_i}-{g_name_j}: {np.shape(sisc[g_name_i,g_name_j])}'
            print(msg)
            sisc_ddict[f'{g_name_i}-{g_name_j}'] = np.ravel(
                sisc[g_name_i, g_name_j])
            tisc_ddict[f'{g_name_i}-{g_name_j}'] = np.ravel(
                tisc[g_name_i, g_name_j])


def compute_isc_stats(iscs, average_over_subjs=False):
    mu, se = {}, {}
    # for i, g_name_i in enumerate(group_names):
    i, g_name_i = 0, 'control'
    for j, g_name_j in enumerate(group_names):
        if i <= j:
            if average_over_subjs:
                isc_ = np.mean(iscs[g_name_i, g_name_j], axis=1)
            else:
                # isc_ = np.mean(iscs[g_name_i][g_name_j], axis=2)
                isc_ = iscs[g_name_i, g_name_j]
            mu_, se_ = compute_stats(isc_)
            mu[f'{g_name_i}-{g_name_j}'] = mu_
            se[f'{g_name_i}-{g_name_j}'] = se_
    return mu, se


'''Temporal ISC'''
n_se = 1
mu_tisc, se_tisc = compute_isc_stats(sisc)
sort_ids = np.argsort(np.mean(mu_tisc['control-control'], axis=0))[::-1]

f, ax = plt.subplots(1, 1, figsize=(7, 4))
for i, key in enumerate(mu_tisc.keys()):
    mu_, se_ = compute_stats(mu_tisc[key])
    ax.errorbar(x=range(len(mu_)), y=mu_[sort_ids], yerr=se_[sort_ids]*n_se,
                label=f'{key}')
ax.legend()
ax.set_xlabel('Components (ordered by ISC value)')
ax.set_ylabel('Temporal ISC')
sns.despine()


mu_sub_ij_tisc = {k: np.mean(mu_tisc[k], axis=1) for k in mu_tisc.keys()}
df = pd.DataFrame(mu_sub_ij_tisc)
# df['ids'] = np.arange(dim_srm)
dabest_data = dabest.load(
    data=df, idx=list(mu_sub_ij_tisc.keys()),
    # paired=True, id_col='ids',
)
dabest_data.mean_diff.plot(
    swarm_label='Temporal ISC', fig_size=(7, 4),
    # swarm_ylim=[.55, 1], contrast_ylim=[-.2, .2]
)
# dabest_data.mean_diff.statistical_tests


'''Spatial ISC'''
mu_sisc, se_sisc = compute_isc_stats(tisc)

f, ax = plt.subplots(1, 1, figsize=(7, 4))
for i, key in enumerate(mu_sisc.keys()):
    print(key)
    np.shape(mu_sisc[key])
    mu_, se_ = compute_stats(mu_sisc[key])
    ax.errorbar(x=range(len(mu_)), y=mu_, yerr=se_*n_se, label=f'{key}')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Spatial ISC')
sns.despine()

mu_sub_ij_sisc = {k: np.mean(mu_sisc[k], axis=1) for k in mu_sisc.keys()}

df = pd.DataFrame(mu_sub_ij_sisc)
# df['ids'] = np.arange(n_subj)
dabest_data = dabest.load(
    data=df, idx=list(mu_sub_ij_sisc.keys()),
    # paired=True, id_col='ids',
)
dabest_data.mean_diff.plot(
    swarm_label='Spatial ISC', fig_size=(7, 4),
    # swarm_ylim=[.3, 1], contrast_ylim=[-.25, .25]
)
# dabest_data.mean_diff.statistical_tests
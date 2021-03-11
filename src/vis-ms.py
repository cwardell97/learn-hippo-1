import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from scipy.stats import pearsonr
from sklearn import metrics
from task import SequenceLearning
from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, get_test_data_dir, \
    pickle_load_dict, get_test_data_fname, pickle_save_dict, load_env_metadata
from analysis import compute_acc, compute_dk, compute_stats, remove_none, \
    compute_cell_memory_similarity, create_sim_dict, compute_mistake, \
    batch_compute_true_dk, process_cache, get_trial_cond_ids, \
    compute_n_trials_to_skip, compute_cell_memory_similarity_stats, \
    sep_by_qsource, get_qsource, trim_data, compute_roc, get_hist_info
from analysis.task import get_oq_keys
from vis import plot_pred_acc_rcl, get_ylim_bonds
from matplotlib.ticker import FormatStrFormatter
import pickle

''' set vars '''
warnings.filterwarnings("ignore")
sns.set(style='white', palette='colorblind', context='poster')
gr_pal = sns.color_palette('colorblind')[2:4]
log_root = '/Users/carsonwardell/Desktop/Thesis/log/'
# log_root = '/tigress/cwardell/logs/learn-hippocampus/log'
all_conds = TZ_COND_DICT.values()


# exp_name = 'vary-schema-level'
# def_prob_range = np.arange(.25, 1, .1)
# for def_prob in def_prob_range:

# the name of the experiemnt
exp_name = 'Mental-Sims-v2.4_ev-p1_r1--ms-p10_r10'
# exp_name = 'familiarity-signal'
subj_ids = np.arange(15)
penalty_random = 0
def_prob = .25
n_def_tps = 0
# n_def_tps = 8
# loading params
pad_len_load = -1
p_rm_ob_enc_load = 0
p_rm_ob_rcl_load = 0
attach_cond = 0
supervised_epoch = 600
epoch_load = 1000
learning_rate = 8e-4
n_branch = 4
n_param = 16
enc_size = 16
n_event_remember = 2
comp_val = .8
leak_val = 0
# test param
penalty_train = 4
penalty_test = 4
enc_size_test = 16
# enc_size_test = 8
pad_len_test = 0
p_test = 0
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test
slience_recall_time = None
similarity_max_test = .9
similarity_min_test = 0
n_examples_test = 256
subj_id = 1








'''init'''
p = P(
    exp_name=exp_name, sup_epoch=supervised_epoch,
    n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
    enc_size=enc_size, n_event_remember=n_event_remember,
    def_prob=def_prob, n_def_tps=n_def_tps,
    penalty=penalty_train, penalty_random=penalty_random,
    p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
)
# create logging dirs
test_params = [penalty_test, pad_len_test, slience_recall_time]
log_path, log_subpath = build_log_path(
    subj_id, p, log_root=log_root, mkdir=False)

(log_subpath)
env = load_env_metadata(log_subpath)
def_path = np.array(env['def_path'])
def_tps = env['def_tps']
log_subpath['data']
print(log_subpath['data'])

# init env
p.update_enc_size(enc_size_test)
task = SequenceLearning(
    n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
    p_rm_ob_enc=p_rm_ob_enc_test, p_rm_ob_rcl=p_rm_ob_rcl_test,
    def_path=def_path, def_prob=def_prob, def_tps=def_tps,
    similarity_cap_lag=p.n_event_remember,
    similarity_max=similarity_max_test,
    similarity_min=similarity_min_test
)

test_data_dir, test_data_subdir = get_test_data_dir(
    log_subpath, epoch_load, test_params)
test_data_fname = get_test_data_fname(n_examples_test)
if enc_size_test != enc_size:
    test_data_dir = os.path.join(
        test_data_dir, f'enc_size_test-{enc_size_test}')
    test_data_subdir = os.path.join(
        test_data_subdir, f'enc_size_test-{enc_size_test}')
fpath = os.path.join(test_data_dir, test_data_fname)
# skip if no data
if not os.path.exists(fpath):
    print('DNE')
    continue


# calc time stuff
T_part = n_param + pad_len_test
T_total = T_part * task.n_parts

# make fig dir
fig_dir = os.path.join(log_subpath['figs'], test_data_subdir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

test_data_dict = pickle_load_dict(fpath)
[dist_a_, Y_, log_cache_, log_cond_] = test_data_dict['results']
[X_raw, Y_raw] = test_data_dict['XY']


# hardcode for practice
fpath = '/Users/carsonwardell/Desktop/Thesis/log/Mental-Sims-v2.4_ev-p1_r1--ms-p10_r10/p-16_b-4_pad-random/tp-0.25/p_rm_ob_rcl-0.00_enc-0.00/lp-4/enc-cum_size-16/nmem-2/rp-LCA_metric-cosine/h-194_hdec-128/lr-0.0008-eta-0.1/sup_epoch-600/subj-1/data/epoch-1000/penalty-4/delay--1/srt-None/n256.pkl'
data_dict = pickle.load(open('/Users/carsonwardell/Desktop/Thesis/n256.pkl', "rb"))
[dist_a_, Y_, log_cache, log_cond_] = data_dict['results']

log_cache)


# unpack cache
[C, H, M, CM, DA, V], [inpt] = process_cache(log_cache, T_total, p)

'''group level input gate by condition -- from vis-data
to do: remove condition grouping, only plot input gate'''
n_se = 1
f, ax = plt.subplots(1, 1, figsize=(
    5 * (pad_len_test / n_param + 1), 4))
for i, cn in enumerate(all_conds):
    p_dict = lca_param_dicts[0]
    p_dict_ = remove_none(p_dict[cn]['mu'])
    mu_, er_ = compute_stats(p_dict_, n_se=n_se, axis=0)
    ax.errorbar(
        x=np.arange(T_part) - pad_len_test, y=mu_[T_part:], yerr=er_[T_part:], label=f'{cn}'
    )
ax.legend()
ax.set_ylim([-.05, .7])
ax.set_ylabel(lca_param_names[0])
ax.set_xlabel('Time (part 2)')
ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
if pad_len_test > 0:
    ax.axvline(0, color='grey', linestyle='--')
sns.despine()
f.tight_layout()
fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-ig.png'
f.savefig(fname, dpi=120, bbox_to_anchor='tight')

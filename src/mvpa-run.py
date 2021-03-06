import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from task import SequenceLearning
from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.utils import chunk
from utils.io import build_log_path, get_test_data_dir, \
    pickle_load_dict, get_test_data_fname, pickle_save_dict, load_env_metadata
from analysis import batch_compute_true_dk, trim_data, \
    process_cache, get_trial_cond_ids, compute_n_trials_to_skip
from analysis.task import get_oq_keys
from analysis.neural import build_yob, build_cv_ids
from vis import imshow_decoding_heatmap
warnings.filterwarnings("ignore")
# plt.switch_backend('agg')
sns.set(style='white', palette='colorblind', context='poster')
all_conds = TZ_COND_DICT.values()
log_root = '../log/'
# log_root = '/tigress/qlu/logs/learn-hippocampus/log'


def compute_matches(proba_, target_):
    proba_ag = np.argmax(proba_, axis=3)
    proba_ag_tp = np.transpose(proba_ag, (0, 2, 1))
    assert np.shape(proba_ag_tp) == np.shape(target_),\
        f'{np.shape(proba_ag_tp)}!={np.shape(target_)}'
    matches = target_ == proba_ag_tp - 1
    match_rate = np.sum(matches) / matches.size
    return matches, match_rate


exp_name = 'vary-training-penalty'
def_prob = .25

# exp_name = 'vary-schema-level'
# def_prob_range = np.arange(.25, 1, .1)
# for def_prob in def_prob_range:
# print(def_prob)

supervised_epoch = 600
epoch_load = 1000
learning_rate = 7e-4

n_branch = 4
n_param = 16
enc_size = 16
# n_def_tps = 8
n_def_tps = 0
comp_val = .8
penalty_random = 0
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = 0

# testing params
enc_size_test = 16
pad_len_test = 0
p_test = 0
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test
slience_recall_time = None

similarity_max_test = .9
similarity_min_test = 0
n_examples_test = 256
subj_ids = np.arange(15)

penalty_test = 2
penalty_train = 2

n_subjs = len(subj_ids)
DM_qsources = ['EM only', 'both']

if not os.path.isdir(f'../figs/{exp_name}'):
    os.makedirs(f'../figs/{exp_name}')


def prealloc_stats():
    return {cond: {'mu': [None] * n_subjs, 'er': [None] * n_subjs}
            for cond in all_conds}


print(f'penalty_train={penalty_train}, penalty_test={penalty_test}')
enc_acc_g = [None] * n_subjs
schematic_enc_err_rate_g = [None] * n_subjs
df_grcl = [None] * n_subjs
df_genc = [None] * n_subjs
prop_pfenc_g = [None] * n_subjs
match_rate_g = np.zeros((n_subjs, 2))
classifier_g = [None] * n_subjs

for i_s, subj_id in enumerate(subj_ids):
    np.random.seed(subj_id)
    torch.manual_seed(subj_id)

    '''init'''
    p = P(
        exp_name=exp_name, sup_epoch=supervised_epoch,
        n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
        enc_size=enc_size, def_prob=def_prob, n_def_tps=n_def_tps,
        penalty=penalty_train, penalty_random=penalty_random,
        p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
        lr=learning_rate,
    )
    # create logging dirs
    test_params = [penalty_test, pad_len_test, slience_recall_time]
    log_path, log_subpath = build_log_path(
        subj_id, p, log_root=log_root, mkdir=False)
    env = load_env_metadata(log_subpath)
    def_path = np.array(env['def_path'])
    def_tps = env['def_tps']
    print(log_subpath['data'])
    p.update_enc_size(enc_size_test)

    # init env
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
    fpath = os.path.join(test_data_dir, test_data_fname)
    if not os.path.exists(fpath):
        print('DNE')
        print(fpath)
        continue

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
    n_examples_skip = compute_n_trials_to_skip(log_cond_, p)
    n_examples = n_examples_test - n_examples_skip
    data_to_trim = [dist_a_, Y_, log_cond_, log_cache_,  X_raw]
    [dist_a, Y, log_cond, log_cache, X_raw] = trim_data(
        n_examples_skip, data_to_trim)
    # process the data
    n_trials = np.shape(X_raw)[0]
    trial_id = np.arange(n_trials)
    cond_ids = get_trial_cond_ids(log_cond)

    activity_, ctrl_param_ = process_cache(log_cache, T_total, p)
    [C, H, M, CM, DA, V] = activity_
    [inpt] = ctrl_param_

    comp = np.full(np.shape(inpt), comp_val)
    leak = np.full(np.shape(inpt), 0)

    # onehot to int
    actions = np.argmax(dist_a, axis=-1)
    targets = np.argmax(Y, axis=-1)

    # compute performance
    corrects = targets == actions
    dks = actions == p.dk_id
    mistakes = np.logical_and(targets != actions, ~dks)

    # split data wrt p1 and p2
    CM_p1, CM_p2 = CM[:, :T_part, :], CM[:, T_part:, :]
    X_raw_p1 = np.array(X_raw)[:, :T_part, :]
    X_raw_p2 = np.array(X_raw)[:, T_part:, :]
    #
    corrects_p2 = corrects[:, T_part:]
    mistakes_p1 = mistakes[:, :T_part]
    mistakes_p2 = mistakes[:, T_part:]
    dks_p2 = dks[:, T_part:]
    inpt_p2 = inpt[:, T_part:]
    targets_p1, targets_p2 = targets[:, :T_part], targets[:, T_part:]
    actions_p1, actions_p2 = actions[:, :T_part], actions[:, T_part:]

    # pre-extract p2 data for the DM condition
    corrects_dmp2 = corrects_p2[cond_ids['DM']]
    mistakes_dmp2 = mistakes_p2[cond_ids['DM']]
    mistakes_dmp1 = mistakes_p1[cond_ids['DM']]
    dks_dmp2 = dks_p2[cond_ids['DM']]
    CM_dmp2 = CM_p2[cond_ids['DM']]

    targets_dmp2 = targets_p2[cond_ids['DM'], :]
    actions_dmp2 = actions_p2[cond_ids['DM']]
    targets_dmp1 = targets_p1[cond_ids['DM'], :]
    actions_dmp1 = actions_p1[cond_ids['DM']]

    # get observation key and values for p1 p2
    o_keys = np.zeros((n_trials, T_total))
    o_vals = np.zeros((n_trials, T_total))
    for i in trial_id:
        o_keys[i], _, o_vals[i] = get_oq_keys(X_raw[i], task)
    o_keys_p1, o_keys_p2 = o_keys[:, :T_part], o_keys[:, T_part:]
    o_vals_p1, o_vals_p2 = o_vals[:, :T_part], o_vals[:, T_part:]
    o_keys_dmp1 = o_keys_p1[cond_ids['DM']]
    o_keys_dmp2 = o_keys_p2[cond_ids['DM']]
    o_vals_dmp1 = o_vals_p1[cond_ids['DM']]
    o_vals_dmp2 = o_vals_p2[cond_ids['DM']]

    is_def_tp = np.array(def_tps).astype(np.bool)
    def_path_int = np.argmax(def_path, axis=1)

    '''decoding data-prep
    '''
    # build y
    Yob_p1 = build_yob(o_keys_p1, o_vals_p1)
    Yob_p2 = build_yob(o_keys_p2, o_vals_p2)

    # precompute mistakes-related variables
    n_mistakes_per_trial = np.sum(mistakes_dmp2, axis=1)
    has_mistake = n_mistakes_per_trial > 0

    # split trials w/ vs. w/o mistakes
    actions_dmp1hm = actions_dmp1[has_mistake, :]
    targets_dmp1hm = targets_dmp1[has_mistake, :]
    actions_dmp2hm = actions_dmp2[has_mistake, :]
    targets_dmp2hm = targets_dmp2[has_mistake, :]
    mistakes_dmp2hm = mistakes_dmp2[has_mistake, :]
    CM_dmp2hm = CM_dmp2[has_mistake, :, :]
    o_keys_dmp1hm = o_keys_dmp1[has_mistake, :]
    o_keys_dmp2hm = o_keys_dmp2[has_mistake, :]

    actions_dmp1nm = actions_dmp1[~has_mistake, :]
    targets_dmp1nm = targets_dmp1[~has_mistake, :]
    actions_dmp2nm = actions_dmp2[~has_mistake, :]
    targets_dmp2nm = targets_dmp2[~has_mistake, :]
    corrects_dmp2nm = corrects_dmp2[~has_mistake, :]
    CM_dmp2nm = CM_dmp2[~has_mistake, :, :]
    o_keys_dmp1nm = o_keys_dmp1[~has_mistake, :]
    o_keys_dmp2nm = o_keys_dmp2[~has_mistake, :]

    o_keys_dmhm = np.hstack([o_keys_dmp1hm, o_keys_dmp2hm])
    actions_dmhm = np.hstack([actions_dmp1hm, actions_dmp2hm])
    targets_dmhm = np.hstack([targets_dmp1hm, targets_dmp2hm])
    o_keys_dmnm = np.hstack([o_keys_dmp1nm, o_keys_dmp2nm])
    actions_dmnm = np.hstack([actions_dmp1nm, actions_dmp2nm])
    targets_dmnm = np.hstack([targets_dmp1nm, targets_dmp2nm])

    actions_dmnm[actions_dmnm == n_branch] = -1
    actions_dmhm[actions_dmhm == n_branch] = -1
    actions_dmhm += 1
    actions_dmnm += 1
    targets_dmhm += 1
    targets_dmnm += 1

    '''decoding'''
    # data prep
    CM = np.hstack([CM_p1, CM_p2])
    Yob = np.hstack([Yob_p1, Yob_p2])
    CM_rs = np.reshape(CM, (n_trials * T_total, p.net.n_hidden))
    Yob_rs = np.reshape(Yob, (n_trials * T_total, T_part))

    # for the RM condition
    # assert no feature is unobserved by the end of part 1
    assert np.sum(Yob[cond_ids['RM']][:, T_part - 1, :] == -1) == 0
    # fill the part 2 to be observed
    for t in np.arange(T_part, T_total):
        Yob[cond_ids['RM'], t, :] = Yob[cond_ids['RM'], T_part - 1, :]
    # for the DM condition
    # estimated recall time
    rt_est = np.argmax(inpt_p2, axis=1)
    dm_ids = np.where(cond_ids['DM'])[0]
    # for the ith DM trial
    for i in range(np.sum(cond_ids['DM'])):
        rti = rt_est[i]
        ii = dm_ids[i]
        # fill the part 2, after recall, to be observed
        for t in np.arange(T_part + rti, T_total):
            Yob[ii, t, :] = Yob[ii, T_part - 1, :]

    # build trial id matrix
    trial_id_mat = np.tile(trial_id, (T_total, 1)).T
    trial_id_unroll = np.reshape(trial_id_mat, (n_trials * T_total, ))

    # start decoding
    n_folds = 10
    rc_alphas = np.logspace(-2, 5, num=4)
    parameters = {'C': rc_alphas}
    cvsplits = chunk(list(range(n_trials)), n_folds)
    Yob_proba = np.zeros((n_trials, T_part, T_total, n_branch + 1))
    best_C = np.zeros(n_folds)
    for fid, testset_ids_i in enumerate(cvsplits):
        temask = np.logical_and(
            trial_id_unroll >= testset_ids_i[0],
            trial_id_unroll <= testset_ids_i[-1])
        # for the n/t-th classifier
        for n in range(T_part):
            # make inner cv ids
            sub_n_trials = len(Yob_rs[~temask, n])
            sub_cvids = build_cv_ids(sub_n_trials, n_folds // 2)
            cvgs = GridSearchCV(
                LogisticRegression(penalty='l2'), parameters,
                cv=PredefinedSplit(sub_cvids), return_train_score=True
            )
            cvgs.fit(CM_rs[~temask], Yob_rs[~temask, n])
            cvgs.best_estimator_.fit(CM_rs[~temask], Yob_rs[~temask, n])

            obsed_class = [i in (np.unique(Yob_rs[~temask, n]) + 1)
                           for i in np.arange(n_branch + 1)]

            # probabilistic estimates for the i-th
            for ii in testset_ids_i:
                Yob_proba[ii, n][:, obsed_class] = cvgs.best_estimator_.predict_proba(
                    CM[ii])

        best_C[fid] = cvgs.best_estimator_.C
        final_models = [
            LogisticRegression(penalty='l2', C=np.mean(best_C))
            for _ in range(T_part)
        ]

        for n in range(T_part):
            final_models[n].fit(CM_rs, Yob_rs[:, n])
        classifier_g[i_s] = final_models

    # LogisticRegression(penalty='l2')
    matches, match_rate = compute_matches(Yob_proba, Yob)
    match_rate_p1 = np.sum(
        matches[:, :T_part, :]) / matches[:, :T_part, :].size
    match_rate_p2 = np.sum(
        matches[:, T_part:, :]) / matches[:, T_part:, :].size
    match_rate_g[i_s] = [match_rate_p1, match_rate_p2]
    print(f'mvpa accuracy = {[match_rate_p1, match_rate_p2]}')

    '''stats for encoding acc'''
    Yob_proba_enc = Yob_proba[cond_ids['DM'], :, T_part - 1, :]
    # get feature values at encoding
    feat_val_enc = np.argmax(Yob_proba_enc, axis=2)
    feat_val_true = targets_dmp2 + 1
    enc_acc_mat_dm = feat_val_true == feat_val_enc
    trial_enc_perfectly = np.all(feat_val_true == feat_val_enc, axis=1)

    enc_acc_g[i_s] = np.sum(enc_acc_mat_dm) / enc_acc_mat_dm.size
    prop_pfenc_g[i_s] = np.sum(
        trial_enc_perfectly) / trial_enc_perfectly.size

    # compute % schema consistent encoding error
    enc_err_locs = np.where(enc_acc_mat_dm == False)

    n_schematic_enc_err = 0
    schematic_enc_err_rate = 0
    n_enc_errs = len(enc_err_locs[0])
    if n_enc_errs > 0:
        for loc_i, loc_j in zip(enc_err_locs[0], enc_err_locs[1]):
            if is_def_tp[loc_j]:
                n_schematic_enc_err += 1
                schematic_enc_err_rate = n_schematic_enc_err / n_enc_errs
    schematic_enc_err_rate_g[i_s] = schematic_enc_err_rate

    '''stats'''
    n_trials_dm = np.sum(cond_ids['DM'])
    Yob_proba_dmp2 = Yob_proba[cond_ids['DM'], :, T_part:, :]
    Yob_maxp_dmp2 = np.argmax(Yob_proba_dmp2, axis=-1)
    # compute a query-before-obs bool
    qbe4o = np.zeros(np.shape(o_keys_dmp2))
    for t in range(T_part):
        qbe4o[:, t] = o_keys_dmp2[:, t] > t

    # init a df_rcl
    df_rcl = pd.DataFrame(
        columns=['trial_id', 'time', 'schema_consistent', 'has_enc_err',
                 'outcome', 'max_response', 'max_response_schematic',
                 'act_response', 'act_response_schematic'],
    )
    df_enc = pd.DataFrame(
        columns=['trial_id', 'time', 'schema_consistent', 'has_enc_err',
                 'max_response'],
    )
    # i, t = 0, 0
    for i in range(n_trials_dm):
        for t in range(T_part):

            p_max_it = Yob_maxp_dmp2[i, t, t]
            is_schematic = def_tps[t]
            act_response = actions_dmp2[i, t]
            has_enc_err = enc_acc_mat_dm[i, t]

            if is_schematic:
                schema_consistent = targets_dmp2[i, t] == def_path_int[t]
                if p_max_it == def_path_int[t] + 1:
                    max_response_schematic = True
                else:
                    max_response_schematic = False
                if act_response == def_path_int[t]:
                    act_response_schematic = True
                else:
                    act_response_schematic = False
            else:
                schema_consistent = np.nan
                max_response_schematic = np.nan
                act_response_schematic = np.nan

            # get the t-th encoded feature for the i-th trial
            # classify it into {other-S, other-NS, studied, dk}
            if feat_val_enc[i, t] == 0:
                # np.sum(feat_val_enc==0)
                enc_type = 'dk'
            elif feat_val_enc[i, t] == feat_val_true[i, t]:
                enc_type = 'studied'
            else:
                if feat_val_enc[i, t] == def_path_int[t] + 1:
                    enc_type = 'other-S'
                else:
                    enc_type = 'other'

            # for part 2, memory based prediction phase
            if dks_dmp2[i, t]:
                outcome = 'dk'
            elif corrects_dmp2[i, t]:
                outcome = 'correct'
            elif mistakes_dmp2[i, t]:
                if act_response_schematic:
                    outcome = 'mistake-S'
                else:
                    outcome = 'mistake'
            else:
                raise ValueError('must be dk or correct or mistake')

            if p_max_it == 0:
                max_response = 'dk'
            elif p_max_it == 1 + targets_dmp2[i, t]:
                max_response = 'studied'
            elif max_response_schematic:
                max_response = 'other-S'
            else:
                max_response = 'other'

            # only count qbe4o trial
            df_enc = df_enc.append({
                'trial_id': i,
                'time': t,
                'schema_consistent': schema_consistent,
                'has_enc_err': has_enc_err,
                'max_response': enc_type
            }, ignore_index=True)

            if qbe4o[i, t]:
                df_rcl = df_rcl.append({
                    'trial_id': i,
                    'time': t,
                    'schema_consistent': schema_consistent,
                    'has_enc_err': has_enc_err,
                    'outcome': outcome,
                    'max_response': max_response,
                    'max_response_schematic': max_response_schematic,
                    'act_response': act_response,
                    'act_response_schematic': act_response_schematic
                }, ignore_index=True)

    df_grcl[i_s] = df_rcl
    df_genc[i_s] = df_enc

    '''plot'''

    # get decoding heatmaps
    Yob_proba_dm = Yob_proba[cond_ids['DM']]
    Yob_proba_hm = Yob_proba_dm[has_mistake, :]
    Yob_proba_nm = Yob_proba_dm[~has_mistake, :]

    # for the i-th mistakes trial, plot the j-th mistake
    fig_dir = os.path.join(log_subpath['figs'], test_data_subdir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    td_dir_path = os.path.join(fig_dir, 'trial_data')
    if not os.path.exists(td_dir_path):
        os.makedirs(td_dir_path)
    # i, j = 0, 0
    for i in range(np.shape(mistakes_dmp2hm)[0]):
        # when/what feature were mistaken
        mistake_feature_i = np.where(mistakes_dmp2hm[i, :])[0]
        for j in range(len(mistake_feature_i)):
            decoded_feat_mat = Yob_proba_hm[i, mistake_feature_i[j]]
            feat_otimes = np.where(
                o_keys_dmhm[i] == mistake_feature_i[j])[0]
            feat_qtimes = mistake_feature_i[j] + np.array([0, T_part])
            targets_dmnm_i = targets_dmhm[i, :]
            actions_dmnm_i = actions_dmhm[i, :]

            f, axes = imshow_decoding_heatmap(
                decoded_feat_mat, feat_otimes, feat_qtimes,
                targets_dmnm_i, actions_dmnm_i, n_param, n_branch
            )
            fig_path = os.path.join(td_dir_path, f'mistake-{i}-{j}')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''corrects'''
    # i, j = 0, 0
    for i in range(np.shape(corrects_dmp2nm)[0]):
        # when/what feature were mistaken
        correct_feature_i = np.where(corrects_dmp2nm[i, :])[0]
        for j in range(len(correct_feature_i)):
            decoded_feat_mat = Yob_proba_nm[i, correct_feature_i[j]]
            feat_otimes = np.where(
                o_keys_dmnm[i] == correct_feature_i[j])[0]
            feat_qtimes = correct_feature_i[j] + np.array([0, T_part])
            targets_dmnm_i = targets_dmnm[i, :]
            actions_dmnm_i = actions_dmnm[i, :]
            f, axes = imshow_decoding_heatmap(
                decoded_feat_mat, feat_otimes, feat_qtimes,
                targets_dmnm_i, actions_dmnm_i, n_param, n_branch
            )
            fig_path = os.path.join(td_dir_path, f'correct-{i}-{j}')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

'''compute average encoding accuracy across subjects'''
mvpa_data_dict = {
    'enc_acc_g': enc_acc_g, 'prop_pfenc_g': prop_pfenc_g,
    'schematic_enc_err_rate_g': schematic_enc_err_rate_g,
    'df_grcl': df_grcl, 'df_genc': df_genc, 'match_rate_g': match_rate_g,
    'classifier_g': classifier_g
}
mvpa_data_dict_fname = f'mvpa-{exp_name}-p{penalty_train}-{penalty_test}-%.2f.pkl' % def_prob
pickle_save_dict(mvpa_data_dict, os.path.join(
    'data', mvpa_data_dict_fname))

import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from itertools import product
from scipy.stats import pearsonr
from sklearn import metrics
from task import SequenceLearning
from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, get_test_data_dir, \
    pickle_load_dict, get_test_data_fname, pickle_save_dict, load_env_metadata
from analysis import compute_acc, compute_dk, compute_stats, \
    compute_cell_memory_similarity, create_sim_dict, compute_mistake, \
    compute_event_similarity, batch_compute_true_dk, \
    process_cache, get_trial_cond_ids, compute_n_trials_to_skip,\
    compute_cell_memory_similarity_stats, sep_by_qsource, prop_true, \
    get_qsource, trim_data, compute_roc, get_hist_info, remove_none

from vis import plot_pred_acc_full, plot_pred_acc_rcl, get_ylim_bonds
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition.pca import PCA

warnings.filterwarnings("ignore")
# plt.switch_backend('agg')
sns.set(style='white', palette='colorblind', context='poster')

all_conds = TZ_COND_DICT.values()

log_root = '../log/'
# exp_name = 'penalty-fixed-discrete-simple_'
# exp_name = 'penalty-random-discrete'
# exp_name = 'penalty-random-discrete-lowsim'
exp_name = '0220-v1-widesim-comp.8'
# exp_name = 'penalty-random-discrete-schema'

supervised_epoch = 600
epoch_load = 1000
learning_rate = 7e-4

n_branch = 4
n_param = 16
enc_size = 16
n_event_remember = 2
# def_prob = .9
def_prob = .25
# n_def_tps = n_param // 2
n_def_tps = 0

n_hidden = 194
n_hidden_dec = 128
eta = .1

penalty_random = 1
# testing param, ortho to the training directory
penalty_discrete = 0
penalty_onehot = 0
normalize_return = 1

# loading params
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
# slience_recall_time = range(n_param)

# similarity_max_test = .4
# similarity_min_test = 0
similarity_max_test = .9
similarity_min_test = 0
n_examples_test = 256

# subj_ids = [2, 3, 4, 5]
subj_ids = np.arange(10)
# subj_ids = np.arange(2)
# subj_ids = [1]

# penaltys_train = [0]
penaltys_train = [0, 4]
# penaltys_train = [4]


n_subjs = len(subj_ids)
DM_qsources = ['EM only', 'both']


def prealloc_stats():
    return {cond: {'mu': [None] * n_subjs, 'er': [None] * n_subjs}
            for cond in all_conds}


for penalty_train in penaltys_train:
    penaltys_test_ = np.arange(0, penalty_train + 1, 4)
    # penaltys_test_ = [penalty_train]
    # penaltys_test_ = [penalty_train]
    for penalty_test in penaltys_test_:
        # penalty_train, penalty_test = 0, 0
        print(f'penalty_train={penalty_train}, penalty_test={penalty_test}')

        acc_dict = prealloc_stats()
        mis_dict = prealloc_stats()
        dk_dict = prealloc_stats()
        inpt_dict = prealloc_stats()
        leak_dict = prealloc_stats()
        comp_dict = prealloc_stats()
        inpt_dict_bq = {qs: [None] * n_subjs for qs in DM_qsources}
        leak_dict_bq = {qs: [None] * n_subjs for qs in DM_qsources}
        comp_dict_bq = {qs: [None] * n_subjs for qs in DM_qsources}
        ma_raw_list = [None] * n_subjs
        ma_list = [None] * n_subjs
        ma_cos_list = [None] * n_subjs
        tma_dm_p2_dict_bq = {qs: [None] * n_subjs for qs in DM_qsources}
        q_source_list = [None] * n_subjs
        ms_lure_list = [None] * n_subjs
        ms_targ_list = [None] * n_subjs
        tpr_list = [None] * n_subjs
        fpr_list = [None] * n_subjs
        auc_list = [None] * n_subjs
        cmpt_bar_list = [None] * n_subjs
        def_tps_list = [None] * n_subjs

        for i_s, subj_id in enumerate(subj_ids):
            np.random.seed(subj_id)
            torch.manual_seed(subj_id)

            '''init'''
            p = P(
                exp_name=exp_name, sup_epoch=supervised_epoch,
                n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
                enc_size=enc_size, n_event_remember=n_event_remember,
                def_prob=def_prob, n_def_tps=n_def_tps,
                penalty=penalty_train, penalty_random=penalty_random,
                penalty_onehot=penalty_onehot, penalty_discrete=penalty_discrete,
                normalize_return=normalize_return,
                p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
                n_hidden=n_hidden, n_hidden_dec=n_hidden_dec,
                lr=learning_rate, eta=eta,
            )
            # create logging dirs
            test_params = [penalty_test, pad_len_test, slience_recall_time]
            log_path, log_subpath = build_log_path(
                subj_id, p, log_root=log_root, mkdir=False)
            env = load_env_metadata(log_subpath)
            def_path = np.array(env['def_path'])
            # def_tps = env['def_tps']
            # def_tps_list[i_s] = def_tps

            p.update_enc_size(enc_size_test)

            # init env
            task = SequenceLearning(
                n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
                p_rm_ob_enc=p_rm_ob_enc_test, p_rm_ob_rcl=p_rm_ob_rcl_test,
                # def_path=def_path, def_prob=def_prob, def_tps=def_tps,
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
            data_to_trim = [dist_a_, Y_, log_cond_,
                            log_cache_, true_dk_wm_, true_dk_em_]
            [dist_a, Y, log_cond, log_cache, true_dk_wm, true_dk_em] = trim_data(
                n_examples_skip, data_to_trim)
            # process the data
            cond_ids = get_trial_cond_ids(log_cond)
            activity_, ctrl_param_ = process_cache(log_cache, T_total, p)
            [C, H, M, CM, DA, V] = activity_
            [inpt, leak, comp] = ctrl_param_

            # onehot to int
            actions = np.argmax(dist_a, axis=-1)
            targets = np.argmax(Y, axis=-1)

            # compute performance
            corrects = targets == actions
            dks = actions == p.dk_id
            mistakes = np.logical_and(targets != actions, ~dks)

            '''plotting params'''
            alpha = .5
            n_se = 3
            # colors
            gr_pal = sns.color_palette('colorblind')[2:4]
            # make dir to save figs
            fig_dir = os.path.join(log_subpath['figs'], test_data_subdir)
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            '''Schematicity influence'''
            schema_consistency = np.array([
                np.sum(np.argmax(def_path, axis=1) == targets_i[T_part:])
                for targets_i in targets
            ])
            # plt.hist(schema_consistency)

            cond_ = 'DM'
            dvs = [corrects, dks, mistakes]
            dv_names = ['corrects', 'dks', 'mistakes']
            f, axes = plt.subplots(1, 3, figsize=(12, 4))
            for i, dv_i in enumerate(dvs):
                dv = np.mean(dv_i, axis=1)
                r_val, p_val = pearsonr(
                    schema_consistency[cond_ids[cond_]], dv[cond_ids[cond_]]
                )
                sns.regplot(
                    schema_consistency[cond_ids[cond_]], dv[cond_ids[cond_]],
                    scatter_kws={'s': 40, 'alpha': .5},
                    x_jitter=.1, y_jitter=.01,
                    ax=axes[i],
                )
                axes[i].set_title('r = %.2f, p = %.2f' % (r_val, p_val))
                axes[i].set_ylabel(dv_names[i])
                axes[i].set_xlabel('Schematicity')
            sns.despine()
            f.tight_layout()
            fig_path = os.path.join(
                fig_dir, f'performance-{cond_}-schema-effect.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''plot behavioral performance - vertically aligned'''
            # input_dict = {'Y': Y, 'dist_a': dist_a, 'cond_ids': cond_ids}
            # pickle_save_dict(input_dict, 'temp/enc8.pkl')

            # f, axes = plt.subplots(3, 1, figsize=(7, 9))

            for i, cn in enumerate(all_conds):
                # f, ax = plt.subplots(1, 1, figsize=(7, 3.5))
                Y_ = Y[cond_ids[cn], :]
                dist_a_ = dist_a[cond_ids[cn], :]
                # compute performance for this condition
                acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
                dk_mu = compute_dk(dist_a_)
                mis_mu, mis_er = compute_mistake(Y_, dist_a_, return_er=True)

                # cache data for all cond-subj
                acc_dict[cn]['mu'][i_s] = acc_mu
                acc_dict[cn]['er'][i_s] = acc_er
                mis_dict[cn]['mu'][i_s] = mis_mu
                mis_dict[cn]['er'][i_s] = mis_er
                dk_dict[cn]['mu'][i_s] = dk_mu

            '''P(error | schema cons) vs. P(error | schema in-cons)'''
            # f, axes = plt.subplots(1, 3, figsize=(17, 5))
            # for i, cn in enumerate(all_conds):
            #     mis_c = mis_dict[cn]['mu'][i_s][T_part:][np.array(
            #         def_tps).astype(np.bool)]
            #     mis_ic = mis_dict[cn]['mu'][i_s][T_part:][~np.array(
            #         def_tps).astype(np.bool)]
            #     mis_er_c = mis_dict[cn]['er'][i_s][T_part:][np.array(
            #         def_tps).astype(np.bool)]
            #     mis_er_ic = mis_dict[cn]['er'][i_s][T_part:][np.array(
            #         def_tps).astype(np.bool)]
            #
            #     heights = [np.mean(mis_c), np.mean(mis_ic)]
            #     yerrs = [np.std(mis_er_c), np.std(mis_er_ic)]
            #     xticklabels = ['schematic', 'nonschematic']
            #     xticks = range(len(heights))
            #
            #     # f, ax = plt.subplots(1, 1, figsize=(6, 5))
            #     axes[i].bar(
            #         x=xticks, height=heights, yerr=yerrs,
            #         color=sns.color_palette('colorblind')[3]
            #     )
            #     axes[i].axhline(0, color='grey', linestyle='--')
            #     axes[i].set_title(cn)
            #     axes[i].set_xlabel('Transition type')
            #     axes[i].set_xticks(xticks)
            #     axes[i].set_ylim([-.05, .5])
            #     axes[i].set_xticklabels(xticklabels)
            #     axes[i].set_ylabel('P(error)')
            #     f.tight_layout()
            #     sns.despine()
            # fig_path = os.path.join(fig_dir, f'error-schema-effect.png')
            # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''plot behavioral performance - each cond separately, 2nd part'''

            f, axes = plt.subplots(1, 3, figsize=(12, 4))
            for i, cn in enumerate(['RM', 'DM', 'NM']):
                Y_ = Y[cond_ids[cn], :]
                dist_a_ = dist_a[cond_ids[cn], :]
                # compute performance for this condition
                acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
                dk_mu = compute_dk(dist_a_)

                if i == 0:
                    add_legend = True
                    show_ylabel = True
                    legend_loc = (.33, .7)

                else:
                    add_legend = False
                    show_ylabel = False

                # plot
                plot_pred_acc_rcl(
                    acc_mu[T_part:], acc_er[T_part:],
                    acc_mu[T_part:] + dk_mu[T_part:],
                    p, f, axes[i],
                    title=f'{cn}',
                    add_legend=add_legend, legend_loc=legend_loc,
                    show_ylabel=show_ylabel
                )
                # axes[i].set_ylabel()
                axes[i].set_ylim([-.05, 1.05])
            fig_path = os.path.join(fig_dir, f'tz-acc-horizontal.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''compare LCA params across conditions'''
            lca_param_names = ['input gate', 'competition']
            lca_param_dicts = [inpt_dict, comp_dict]
            lca_param_records = [inpt, comp]
            for i, cn in enumerate(all_conds):
                for p_dict, p_record in zip(lca_param_dicts, lca_param_records):
                    p_dict[cn]['mu'][i_s], p_dict[cn]['er'][i_s] = compute_stats(
                        p_record[cond_ids[cn]])

            f, axes = plt.subplots(2, 1, figsize=(7, 6))
            for i, cn in enumerate(all_conds):
                for j, p_dict in enumerate(lca_param_dicts):
                    axes[j].errorbar(
                        x=range(T_part),
                        y=p_dict[cn]['mu'][i_s][T_part:],
                        yerr=p_dict[cn]['er'][i_s][T_part:],
                        label=f'{cn}'
                    )
            for i, ax in enumerate(axes):
                ax.legend()
                ax.set_ylabel(lca_param_names[i])
                ax.set_xlabel('Time, recall phase')
                ax.set_xticks(np.arange(0, p.env.n_param, 5))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if pad_len_test > 0:
                for ax in axes:
                    ax.axvline(pad_len_test, color='grey', linestyle='--')
            sns.despine()
            f.tight_layout()

            fig_path = os.path.join(fig_dir, f'tz-lca-param.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''comp'''
            # i_s = 1
            # n_se = 3
            # mean_cmpt_by_cond = [
            #     np.mean(p_dict[cn]['mu'][i_s][T_part:])
            #     for cn in all_conds
            # ]
            # se_cmpt_by_cond = [
            #     n_se * np.std(p_dict[cn]['mu'][i_s][T_part:]) / np.sqrt(T_part)
            #     for cn in all_conds
            # ]
            #
            # f, ax = plt.subplots(1, 1, figsize=(5, 4))
            # xticks = range(len(all_conds))
            # ax.bar(
            #     x=xticks, height=mean_cmpt_by_cond, yerr=se_cmpt_by_cond,
            #     color=sns.color_palette('colorblind')
            # )
            # ax.set_xticks(list(xticks))
            # ax.set_xticklabels(list(all_conds))
            # ax.set_ylim([0, 1])
            # ax.set_ylabel('Competition')
            # sns.despine()
            # fig_path = os.path.join(fig_dir, f'tz-lca-cmpt-bar.png')
            # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''compare LCA params across conditions'''
            # for j, p_dict in enumerate(lca_param_dicts):
            #     f, ax = plt.subplots(1, 1, figsize=(7, 3.5))
            #     for i, cn in enumerate(['RM', 'DM']):
            #         ax.errorbar(
            #             x=range(T_part),
            #             y=p_dict[cn]['mu'][i_s][T_part:],
            #             yerr=p_dict[cn]['er'][i_s][T_part:],
            #             label=f'{cn}'
            #         )
            #     ax.legend()
            #     ax.set_ylabel(lca_param_names[j])
            #     ax.set_xlabel('Time, recall phase')
            #     ax.set_xticks(np.arange(0, p.env.n_param, 5))
            #     ax.set_ylim([-.05, .7])
            #     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #     if pad_len_test > 0:
            #         ax.axvline(pad_len_test, color='grey', linestyle='--')
            #     sns.despine()
            #     f.tight_layout()
            #     fig_path = os.path.join(
            #         fig_dir, f'tz-lca-param-{lca_param_names[j]}.png')
            #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            # for j, p_dict in enumerate(lca_param_dicts):
            #     f, ax = plt.subplots(1, 1, figsize=(7, 3.5))
            #     ax.errorbar(
            #         x=range(T_part),
            #         y=p_dict['DM']['mu'][i_s][T_part:],
            #         yerr=p_dict['DM']['er'][i_s][T_part:],
            #     )
            #     # ax.legend()
            #     ax.set_ylabel(lca_param_names[j])
            #     ax.set_xlabel('Time, recall phase')
            #     ax.set_xticks(np.arange(0, p.env.n_param, 5))
            #     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #     if pad_len_test > 0:
            #         ax.axvline(pad_len_test, color='grey', linestyle='--')
            #     sns.despine()
            #     f.tight_layout()
            #     fig_path = os.path.join(
            #         fig_dir, f'tz-lca-param-{lca_param_names[j]}.png')
            #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''compute cell-memory similarity / memory activation '''
            # compute similarity between cell state vs. memories
            sim_cos, sim_lca = compute_cell_memory_similarity(
                C, V, inpt, leak, comp)
            sim_cos_dict = create_sim_dict(
                sim_cos, cond_ids, n_targ=p.n_segments)
            sim_lca_dict = create_sim_dict(
                sim_lca, cond_ids, n_targ=p.n_segments)
            sim_cos_stats = compute_cell_memory_similarity_stats(
                sim_cos_dict, cond_ids)
            sim_lca_stats = compute_cell_memory_similarity_stats(
                sim_lca_dict, cond_ids)
            ma_list[i_s] = sim_lca_stats
            ma_raw_list[i_s] = sim_lca_dict
            ma_cos_list[i_s] = sim_cos_stats

            avg_ma = {cond: {m_type: None for m_type in memory_types}
                      for cond in all_conds}
            for cond in all_conds:
                for m_type in memory_types:
                    if sim_lca_dict[cond][m_type] is not None:
                        # print(np.shape(sim_lca_dict[cond][m_type]))
                        avg_ma[cond][m_type] = np.mean(
                            sim_lca_dict[cond][m_type], axis=-1
                        )

            '''mem_act | schema cons vs. mem_act | schema in-cons'''

            # cn = 'DM'
            # ma_dm_p2 = avg_ma[cn]['targ'][:, T_part:]
            # ma_dm_p2_c = ma_dm_p2[:, np.array(def_tps).astype(np.bool)]
            # ma_dm_p2_ic = ma_dm_p2[:, ~np.array(def_tps).astype(np.bool)]
            # ma_dm_p2_c = np.mean(ma_dm_p2_c, axis=1)
            # ma_dm_p2_ic = np.mean(ma_dm_p2_ic, axis=1)
            # heights = [np.mean(ma_dm_p2_c), np.mean(ma_dm_p2_ic)]
            # xticks = range(len(heights))
            # xticklabels = ['schematic', 'nonschematic']
            # f, ax = plt.subplots(1, 1, figsize=(6, 5))
            # ax.bar(
            #     x=xticks, height=heights,
            #     color=sns.color_palette('colorblind')[2]
            # )
            # ax.axhline(0, color='grey', linestyle='--')
            # ax.set_title(cn)
            # ax.set_xlabel('Transition type')
            # ax.set_xticks(xticks)
            # # ax.set_ylim([-.05, .5])
            # ax.set_xticklabels(xticklabels)
            # ax.set_ylabel('Memory activation')
            # f.tight_layout()
            # sns.despine()

            # '''plot target/lure activation for all conditions'''
            # sim_stats_plt = {'LCA': sim_lca_stats, 'cosine': sim_cos_stats}
            # ylim_bonds = {'LCA': None, 'cosine': None}
            # for ker_name, sim_stats_plt_ in sim_stats_plt.items():
            #     # print(ker_name, sim_stats_plt_)
            #     tsf = (T_part + pad_len_test) / T_part
            #     f, axes = plt.subplots(3, 1, figsize=(7 * tsf, 9))
            #     for i, c_name in enumerate(cond_ids.keys()):
            #         for m_type in memory_types:
            #             if m_type == 'targ' and c_name == 'NM':
            #                 continue
            #             color_ = gr_pal[0] if m_type == 'targ' else gr_pal[1]
            #             axes[i].errorbar(
            #                 x=range(T_part),
            #                 y=sim_stats_plt_[c_name][m_type]['mu'][T_part:],
            #                 yerr=sim_stats_plt_[c_name][m_type]['er'][T_part:],
            #                 label=f'{m_type}', color=color_
            #             )
            #             axes[i].set_title(c_name)
            #             axes[i].set_ylabel('Memory activation')
            #     axes[0].legend()
            #     axes[-1].set_xlabel('Time, recall phase')
            #     # make all ylims the same
            #     ylim_bonds[ker_name] = get_ylim_bonds(axes)
            #     # ylim_bonds[ker_name] = [-.05, .6]
            #     for i, ax in enumerate(axes):
            #         ax.set_ylim(ylim_bonds[ker_name])
            #         ax.set_xticks(np.arange(0, p.env.n_param, 5))
            #         ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            #         ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #
            #     if pad_len_test > 0:
            #         for ax in axes:
            #             ax.axvline(pad_len_test, color='grey', linestyle='--')
            #     f.tight_layout()
            #     sns.despine()
            #     fig_path = os.path.join(fig_dir, f'tz-memact-{ker_name}.png')
            #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''plot target/lure activation for all conditions - horizontal'''
            # sim_stats_plt = {'LCA': sim_lca_stats, 'cosine': sim_cos_stats}
            ylim_bonds = {'LCA': None, 'cosine': None}
            ker_name, sim_stats_plt_ = 'LCA', sim_lca_stats
            # print(ker_name, sim_stats_plt_)
            tsf = (T_part + pad_len_test) / T_part
            f, axes = plt.subplots(1, 3, figsize=(12, 4))
            for i, c_name in enumerate(cond_ids.keys()):
                for m_type in memory_types:
                    if m_type == 'targ' and c_name == 'NM':
                        continue
                    color_ = gr_pal[0] if m_type == 'targ' else gr_pal[1]
                    axes[i].errorbar(
                        x=range(T_part),
                        y=sim_stats_plt_[c_name][m_type]['mu'][T_part:],
                        yerr=sim_stats_plt_[c_name][m_type]['er'][T_part:],
                        label=f'{m_type}', color=color_
                    )
                    axes[i].set_title(c_name)
                    axes[i].set_xlabel('Time')
            axes[0].set_ylabel('Memory activation')
            axes[0].legend()

            # make all ylims the same
            ylim_bonds[ker_name] = get_ylim_bonds(axes)
            ylim_bonds[ker_name] = [-.05, .6]
            for i, ax in enumerate(axes):
                ax.set_ylim(ylim_bonds[ker_name])
                ax.set_xticks([0, p.env.n_param - 1])
                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            if pad_len_test > 0:
                for ax in axes:
                    ax.axvline(pad_len_test, color='grey', linestyle='--')
            f.tight_layout()
            sns.despine()
            fig_path = os.path.join(fig_dir, f'tz-memact-{ker_name}-hori.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            # '''plot target/lure activation for all conditions'''
            # sim_stats_plt = {'LCA': sim_lca_stats, 'cosine': sim_cos_stats}
            # for ker_name, sim_stats_plt_ in sim_stats_plt.items():
            #     # print(ker_name, sim_stats_plt_)
            #     tsf = (T_part + pad_len_test) / T_part
            #     for i, c_name in enumerate(cond_ids.keys()):
            #         f, ax = plt.subplots(1, 1, figsize=(7 * tsf, 3.5))
            #         for m_type in memory_types:
            #             if m_type == 'targ' and c_name == 'NM':
            #                 continue
            #             color_ = gr_pal[0] if m_type == 'targ' else gr_pal[1]
            #             ax.errorbar(
            #                 x=range(T_part),
            #                 y=sim_stats_plt_[c_name][m_type]['mu'][T_part:],
            #                 yerr=sim_stats_plt_[c_name][m_type]['er'][T_part:],
            #                 label=f'{m_type}', color=color_
            #             )
            #         ax.set_title(c_name)
            #         ax.set_ylabel('Memory activation')
            #         ax.set_xlabel('Time, recall phase')
            #         ax.legend()
            #
            #         # ax.set_ylim([-.05, .625])
            #         ax.set_ylim(ylim_bonds[ker_name])
            #         ax.set_xticks(np.arange(0, p.env.n_param, 5))
            #         ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            #         ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #
            #         if pad_len_test > 0:
            #             ax.axvline(pad_len_test, color='grey', linestyle='--')
            #         f.tight_layout()
            #         sns.despine()
            #         fig_path = os.path.join(
            #             fig_dir, f'tz-memact-{ker_name}-{c_name}.png')
            #         f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''correlate lure recall vs. behavior '''
            for cond_ in ['NM', 'DM']:
                lure_act_dmp2 = sim_lca_dict[cond_]['lure'][:, T_part:, :]
                corrects_dmp2 = corrects[cond_ids[cond_]][:, T_part:]
                mistakes_dmp2 = mistakes[cond_ids[cond_]][:, T_part:]
                dks_dmp2 = dks[cond_ids[cond_]][:, T_part:]

                n_corrects_dmp2 = np.mean(corrects_dmp2, axis=1)
                n_mistakes_dmp2 = np.mean(mistakes_dmp2, axis=1)
                n_dks_dmp2 = np.mean(dks_dmp2, axis=1)
                sum_lure_act_dmp2 = np.mean(
                    np.mean(lure_act_dmp2, axis=-1), axis=1)

                dvs = [n_corrects_dmp2, n_mistakes_dmp2, n_dks_dmp2]
                ylabels = ['% correct', '% mistake', '% uncertain']
                colors = [sns.color_palette()[0], sns.color_palette()
                          [3], 'grey']

                f, axes = plt.subplots(
                    1, len(dvs), figsize=(5 * len(dvs), 4.5))
                for i, ax in enumerate(axes):
                    r_val, p_val = pearsonr(sum_lure_act_dmp2, dvs[i])
                    sns.regplot(
                        sum_lure_act_dmp2, dvs[i], color=colors[i],
                        ax=axes[i]
                    )
                    ax.set_xlabel('Lure activation')
                    ax.set_ylabel(ylabels[i])
                    tt_ = r'$r \approx %.2f, p \approx %.3f$' % (r_val, p_val)
                    ax.set_title(tt_)
                sns.despine()
                f.tight_layout()
                fig_path = os.path.join(fig_dir, f'reg-lure-behav-{cond_}.png')
                f.savefig(fig_path, dpi=150, bbox_to_anchor='tight')

            '''use lure recall to predict what error the model make'''

            # # np.argmax(np.mean(lure_act_dmp2, axis=1), axis=-1)
            # np.shape(Y)
            #
            # # get the memory ids for all memories
            # # index 0 - ongoing event; index 1 -> previous event; etc.
            # mem_ids = [np.where(cond_ids[cond_])[0] - i
            #            for i in range(n_event_remember)]
            #
            # # [i for i in range(n_event_remember)]
            # mem_ids[0]
            # # np.shape(mem_ids)
            # len(X_raw)
            # X_raw[0]
            # # extract observed features from prev events
            # i = 1
            # X_i = np.array(X_raw)[mem_ids[i]]
            #
            # X_i_ok = X_i[:, :, :task.k_dim]
            # X_i_ov = X_i[:, :, task.k_dim:task.k_dim+task.v_dim]
            # event_order = np.argmax(X_i_ok, axis=2)
            #
            # n_fv_observed = np.zeros((len(X_i), T_part))
            # # loop over events
            # for j in range(len(X_i)):
            #     event_order_j_p2 = event_order[j][T_part:]
            #     for t in range(T_part):
            #         n_fv_observed[j, t] = np.sum(event_order_j_p2[:t] <= t)
            #
            # np.shape(X_i_ok)
            # np.shape(np.argmax(X_i_ok, axis=2))
            #
            # i = 0
            #
            # np.array(X_raw)[mem_ids[i]]

            '''compute q source, and check q source % for all conditions'''

            # pick a condition
            q_source = get_qsource(true_dk_em, true_dk_wm, cond_ids, p)
            all_q_source = list(q_source['DM'].keys())
            q_source_list[i_s] = q_source

            # split lca parameters by query source
            lca_param_dicts_bq = dict(
                zip(lca_param_names, [inpt_dict_bq, leak_dict_bq, comp_dict_bq]))
            for i_p, p_name in enumerate(lca_param_names):
                p_record_cond = lca_param_records[i_p][cond_ids['DM']]
                p_record_cond_qs = sep_by_qsource(
                    p_record_cond, q_source['DM'], n_se=n_se)
                for qs in DM_qsources:
                    lca_param_dicts_bq[p_name][qs][i_s] = p_record_cond_qs[qs][0]

            # split target actvation by query source
            tma_qs = sep_by_qsource(
                avg_ma['DM']['targ'], q_source['DM'], n_se=n_se)

            # plot distribution of  query source
            width = .85
            f, axes = plt.subplots(3, 1, figsize=(7, 10))
            for i, (cd_name, q_source_cd_p2) in enumerate(q_source.items()):
                # unpack data
                eo_cd_p2, wo_cd_p2, nt_cd_p2, bt_cd_p2 = q_source_cd_p2.values()
                axes[i].bar(range(n_param), prop_true(
                    eo_cd_p2), label='EM', width=width)
                axes[i].bar(range(n_param), prop_true(wo_cd_p2), label='WM', width=width,
                            bottom=prop_true(eo_cd_p2))
                axes[i].bar(range(n_param), prop_true(bt_cd_p2), label='both', width=width,
                            bottom=prop_true(eo_cd_p2) + prop_true(wo_cd_p2))
                axes[i].bar(range(n_param), prop_true(nt_cd_p2), label='neither', width=width,
                            bottom=prop_true(eo_cd_p2) + prop_true(wo_cd_p2) + prop_true(bt_cd_p2))
                axes[i].set_ylabel('Proportion (%)')
                axes[i].set_title(f'{cd_name}')
                axes[i].legend()
                axes[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
            axes[-1].set_xlabel('Time, recall phase')
            sns.despine()
            f.tight_layout()
            fig_path = os.path.join(fig_dir, f'tz-q-source.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''plot target memory activation profile, for all trials'''
            cond_name = 'DM'
            m_type = 'targ'
            ylab = 'Activation'

            f, axes = plt.subplots(2, 2, figsize=(9, 7))
            mu_, er_ = compute_stats(
                avg_ma[cond_name][m_type][:, T_part:], n_se=3
            )
            axes[0, 0].plot(
                avg_ma[cond_name][m_type][:, T_part:].T,
                alpha=.1, color=gr_pal[0]
            )
            axes[0, 0].errorbar(x=range(T_part), y=mu_,
                                yerr=er_, color='black')
            axes[0, 0].set_xlabel('Time, recall phase')
            axes[0, 0].set_ylabel(ylab)
            axes[0, 0].set_title(f'All trials')

            n_trials_ = 5

            trials_ = np.random.choice(
                range(len(avg_ma[cond_name][m_type])), n_trials_)
            axes[0, 1].plot(avg_ma[cond_name][m_type]
                            [:, T_part:][trials_, :].T)

            axes[0, 1].set_xlabel('Time, recall phase')
            axes[0, 1].set_ylabel(ylab)
            axes[0, 1].set_title(f'{n_trials_} example trials')
            axes[0, 1].set_ylim(axes[0, 0].get_ylim())

            sorted_targ_act_cond_p2 = np.sort(
                avg_ma[cond_name][m_type][:, T_part:], axis=1)[:, :: -1]
            mu_, er_ = compute_stats(sorted_targ_act_cond_p2, n_se=3)
            axes[1, 0].plot(sorted_targ_act_cond_p2.T,
                            alpha=.1, color=gr_pal[0])
            axes[1, 0].errorbar(x=range(T_part), y=mu_,
                                yerr=er_, color='black')
            axes[1, 0].set_ylabel(ylab)
            axes[1, 0].set_xlabel(f'Time (the sorting axis)')
            axes[1, 0].set_title(f'Sorted')

            recall_peak_times = np.argmax(
                avg_ma[cond_name][m_type][:, T_part:], axis=1)
            sns.violinplot(recall_peak_times, color=gr_pal[0], ax=axes[1, 1])
            axes[1, 1].set_xlim(axes[0, 1].get_xlim())
            axes[1, 1].set_title(f'Max distribution')
            axes[1, 1].set_xlabel('Time, recall phase')
            axes[1, 1].set_ylabel('Density')

            if pad_len_test > 0:
                axes[0, 0].axvline(pad_len_test, color='grey', linestyle='--')
                axes[0, 1].axvline(pad_len_test, color='grey', linestyle='--')
                axes[1, 1].axvline(pad_len_test, color='grey', linestyle='--')
            m_type_txt = 'target' if m_type == 'targ' else m_type
            f.suptitle(f'Memory activation profile, {m_type_txt}, {cond_name}',
                       y=.95, fontsize=18)

            sns.despine()
            f.tight_layout(rect=[0, 0, 1, 0.9])
            # f.subplots_adjust(top=0.9)
            fig_path = os.path.join(
                fig_dir, f'mem-act-profile-{cond_name}-{m_type}-lca.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''memory activation ~ time + dk'''
            cond_name = 'DM'
            m_type = 'targ'
            ylab = 'Activation'

            # f, axes = plt.subplots(2, 2, figsize=(9, 7))
            np.shape(avg_ma[cond_name][m_type][:, T_part:])
            np.shape(q_source['DM']['EM only'])

            # mu_, er_ = compute_stats(
            #     avg_ma[cond_name][m_type][:, T_part:], n_se=3
            # )

            # nthres = 50
            # t = 2
            # tma_min = np.min(avg_ma[cond_name][m_type][:, T_part:])
            # tma_max = np.max(avg_ma[cond_name][m_type][:, T_part:])
            # thres = np.linspace(tma_min, tma_max, nthres)
            # prob = np.zeros((T_part, nthres))
            #
            # for t in range(T_part):
            #     for i, thres_i in enumerate(thres):
            #         prob[t, i] = np.mean(
            #             avg_ma[cond_name][m_type][:, T_part+t] > thres_i)
            #
            # v_pal = sns.color_palette('Blues', n_colors=T_part)
            # sns.palplot(v_pal)
            #
            # f, ax = plt.subplots(1, 1, figsize=(5, 4))
            # for t in range(T_part):
            #     ax.plot(np.log(thres), prob[t, :], color=v_pal[t])
            # ax.set_ylabel('P(activation > v | t)')
            # ax.set_xlabel('log(v)')
            # f.tight_layout()
            # sns.despine()

            # use previous uncertainty to predict memory activation
            cond_name = 'DM'
            m_type = 'targ'
            dk_cond_p2 = dks[cond_ids[cond_name], n_param:]
            t_pick_max = 9
            t_picks = np.arange(2, t_pick_max)
            v_pal = sns.color_palette('viridis', n_colors=t_pick_max)
            f, ax = plt.subplots(1, 1, figsize=(8, 4))
            for t_pick_ in t_picks:
                # compute number of don't knows produced so far
                ndks_p2_b4recall = np.sum(dk_cond_p2[:, :t_pick_], axis=1)
                ndks_p2_b4recall = ndks_p2_b4recall / t_pick_
                nvs = np.unique(ndks_p2_b4recall)
                ma_mu = np.zeros(len(nvs),)
                ma_er = np.zeros(len(nvs),)
                for i, val in enumerate(np.unique(ndks_p2_b4recall)):
                    ma_ndk = avg_ma[cond_name][m_type][:,
                                                       T_part:][ndks_p2_b4recall == val, t_pick_]
                    ma_mu[i], ma_er[i] = compute_stats(
                        ma_ndk, n_se=1)
                ax.errorbar(x=nvs, y=ma_mu, yerr=ma_er, color=v_pal[t_pick_])

            ax.legend(t_picks, title='time', bbox_to_anchor=(1.3, 1.1))
            ax.set_title(f'Target activation, {cond_name}')
            # ax.set_xlabel('# don\'t knows')
            ax.set_xlabel('percent uncertain')
            ax.set_xlim([0, 1.05])
            ax.set_ylabel('average recall peak')
            sns.despine()
            f.tight_layout()
            fig_path = os.path.join(
                fig_dir, f'tz-{cond_name}-targact-by-propdk.png')
            # fig_path = os.path.join(fig_dir, f'tz-{cond_name}-targact-by-ndk.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            # use objective uncertainty metric to decompose LCA params in the EM condition
            cond_name = 'DM'
            inpt_cond_p2 = inpt[cond_ids[cond_name], T_part:]
            # leak_cond_p2 = leak[cond_ids[cond_name], T_part:]
            comp_cond_p2 = comp[cond_ids[cond_name], T_part:]
            q_source_cond_p2 = q_source[cond_name]
            all_lca_param_cond_p2 = {
                'input gate': inpt_cond_p2,
                # 'leak': leak_cond_p2,
                'competition': comp_cond_p2
            }

            f, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
            for i, (param_name, param_cond_p2) in enumerate(all_lca_param_cond_p2.items()):
                # print(i, param_name, param_cond_p2)
                param_cond_p2_stats = sep_by_qsource(
                    param_cond_p2[:, pad_len_test:], q_source_cond_p2, n_se=n_se)
                for key, [mu_, er_] in param_cond_p2_stats.items():
                    if not np.all(np.isnan(mu_)):
                        axes[i].errorbar(x=range(n_param), y=mu_,
                                         yerr=er_, label=key)
                axes[i].set_ylabel(f'{param_name}')
                axes[i].legend(fancybox=True)

            axes[-1].set_xlabel('Time, recall phase')
            axes[0].set_title(f'LCA params, {cond_name}')
            for ax in axes:
                ax.set_xticks(np.arange(0, p.env.n_param, 5))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            f.tight_layout()
            sns.despine()
            fig_path = os.path.join(fig_dir, f'tz-lca-param-{cond_name}.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            # use CURRENT uncertainty to predict memory activation
            cond_name = 'DM'
            targ_act_cond_p2_stats = sep_by_qsource(
                avg_ma[cond_name]['targ'][:, T_part + pad_len_test:],
                q_source[cond_name],
                n_se=n_se
            )

            for qs in DM_qsources:
                tma_dm_p2_dict_bq[qs][i_s] = targ_act_cond_p2_stats[qs][0]

            f, ax = plt.subplots(1, 1, figsize=(7, 4))
            for key, [mu_, er_] in targ_act_cond_p2_stats.items():
                if not np.all(np.isnan(mu_)):
                    ax.errorbar(x=range(n_param), y=mu_, yerr=er_, label=key)
            ax.set_ylabel(f'{param_name}')
            # ax.legend(fancybox=True)
            ax.set_title(f'Target memory activation, {cond_name}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Activation')
            # ax.set_ylim([-.05, .75])
            ax.set_xticks([0, p.env.n_param - 1])
            ax.legend(['not in WM', 'in WM'], fancybox=True)
            # ax.legend([])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            f.tight_layout()
            sns.despine()
            fig_path = os.path.join(fig_dir, f'tma-{cond_name}-by-qsource.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''
            predictions performance / dk / errors
            w.r.t prediction source (EM, WM)
            '''
            cond_name = 'DM'
            corrects_cond_p2 = corrects[cond_ids[cond_name], n_param:]
            mistakes_cond_p2 = mistakes[cond_ids[cond_name], n_param:]
            acc_cond_p2_stats = sep_by_qsource(
                corrects_cond_p2, q_source[cond_name], n_se=n_se)
            dk_cond_p2_stats = sep_by_qsource(
                dk_cond_p2, q_source[cond_name], n_se=n_se)
            mistakes_cond_p2_stats = sep_by_qsource(
                mistakes_cond_p2, q_source[cond_name], n_se=n_se)

            stats_to_plot = {
                'correct': acc_cond_p2_stats, 'uncertain': dk_cond_p2_stats,
                'error': mistakes_cond_p2_stats,
            }

            f, axes = plt.subplots(len(stats_to_plot), 1, figsize=(7, 10))
            # loop over all statistics
            for i, (stats_name, stat) in enumerate(stats_to_plot.items()):
                # loop over all q source
                for key, [mu_, er_] in stat.items():
                    # plot if sample > 0
                    if not np.all(np.isnan(mu_)):
                        axes[i].errorbar(
                            x=range(n_param), y=mu_, yerr=er_, label=key
                        )
                # for every panel/stats
                axes[i].set_ylabel(f'P({stats_name})')
                axes[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
                axes[i].set_ylim([-.05, 1.05])
            # for the entire panel
            axes[0].set_title(f'Performance, {cond_name}')
            axes[-1].legend(fancybox=True)
            axes[-1].set_xlabel('Time, recall phase')
            f.tight_layout()
            sns.despine()
            fig_path = os.path.join(
                fig_dir, f'tz-{cond_name}-stats-by-qsource.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''ma ~ correct in the EM only case'''
            cond_name = 'DM'
            m_type = 'targ'

            tma_crt_mu, tma_crt_er = np.zeros(n_param,), np.zeros(n_param,)
            tma_incrt_mu, tma_incrt_er = np.zeros(n_param,), np.zeros(n_param,)
            for t in range(n_param):
                sel_op = q_source[cond_name]['EM only'][:, t]
                tma_ = avg_ma[cond_name][m_type][sel_op,
                                                 T_part + t + pad_len_test]
                crt_ = corrects_cond_p2[q_source[cond_name]
                                        ['EM only'][:, t], t]
                tma_crt_mu[t], tma_crt_er[t] = compute_stats(tma_[crt_])
                tma_incrt_mu[t], tma_incrt_er[t] = compute_stats(tma_[~crt_])

            f, ax = plt.subplots(1, 1, figsize=(7, 4))
            ax.errorbar(x=range(n_param), y=tma_crt_mu,
                        yerr=tma_crt_er, label='correct')
            ax.errorbar(x=range(n_param), y=tma_incrt_mu,
                        yerr=tma_incrt_er, label='incorrect')
            ax.set_ylim([-.05, None])
            ax.legend()
            ax.set_title(f'Target memory activation, {cond_name}')
            ax.set_ylabel('Activation')
            ax.set_xlabel('Time, recall phase')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            sns.despine()
            f.tight_layout()
            fig_path = os.path.join(fig_dir, f'tma-{cond_name}-by-cic.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''ma ~ kdk in the EM only case'''

            tma_k_mu, tma_k_er = np.zeros(n_param,), np.zeros(n_param,)
            tma_dk_mu, tma_dk_er = np.zeros(n_param,), np.zeros(n_param,)
            for t in range(n_param):
                sel_op = q_source[cond_name]['EM only'][:, t]
                tma_ = avg_ma[cond_name][m_type][sel_op,
                                                 T_part + t + pad_len_test]
                dk_ = dk_cond_p2[q_source[cond_name]['EM only'][:, t], t]
                tma_k_mu[t], tma_k_er[t] = compute_stats(tma_[~dk_])
                tma_dk_mu[t], tma_dk_er[t] = compute_stats(tma_[dk_])

            f, ax = plt.subplots(1, 1, figsize=(7, 4))
            ax.errorbar(x=range(n_param), y=tma_k_mu,
                        yerr=tma_k_er, label='know')
            ax.errorbar(x=range(n_param), y=tma_dk_mu,
                        yerr=tma_dk_er, label='don\'t know')
            ax.set_ylim([-.05, None])
            ax.legend()
            ax.set_title(f'Target memory activation, {cond_name}')
            ax.set_ylabel('Activation')
            ax.set_xlabel('Time, recall phase')
            sns.despine()
            f.tight_layout()
            fig_path = os.path.join(fig_dir, f'tma-{cond_name}-by-kdk.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''analyze the EM-only condition'''

            for source_ in all_q_source:
                if np.all(np.isnan(acc_cond_p2_stats[source_][0])):
                    continue
                f, ax = plt.subplots(1, 1, figsize=(6, 3.5))
                plot_pred_acc_rcl(
                    acc_cond_p2_stats[source_][0], acc_cond_p2_stats[source_][1],
                    acc_cond_p2_stats[source_][0] +
                    dk_cond_p2_stats[source_][0],
                    p, f, ax,
                    title=f'Prediction performance, {source_}, {cond_name}',
                    add_legend=True,
                )
                # if slience_recall_time is not None:
                #     ax.axvline(slience_recall_time, color='red',
                #                linestyle='--', alpha=alpha)
                ax.set_xlabel('Time, recall phase')
                ax.set_ylabel('Accuracy')
                ax.set_ylim([0, 1.05])
                f.tight_layout()
                sns.despine()
                fig_path = os.path.join(
                    fig_dir, f'tz-pa-{cond_name}-{source_}.png')
                f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''compare the over time'''

            def get_max_score(mem_act_cond):
                n_trials_ = np.shape(mem_act_cond)[0]
                rt_ = np.argmax(
                    np.max(mem_act_cond[:, T_part:], axis=-1),
                    axis=-1
                ) + T_part
                ms_targ = np.array(
                    [np.max(mem_act_cond[i, rt_[i], :])
                     for i in range(n_trials_)]
                )
                return ms_targ

            ms_lure = get_max_score(sim_lca_dict['NM']['lure'])
            ms_targ = get_max_score(sim_lca_dict['DM']['targ'])

            [dist_l, dist_r], [hist_info_l, hist_info_r] = get_hist_info(
                ms_lure, ms_targ)
            tpr, fpr = compute_roc(dist_l, dist_r)
            auc = metrics.auc(fpr, tpr)

            # collect group data
            ms_lure_list[i_s] = ms_lure
            ms_targ_list[i_s] = ms_targ
            tpr_list[i_s] = tpr
            fpr_list[i_s] = fpr
            auc_list[i_s] = auc

            [dist_l_edges, dist_l_normed, dist_l_edges_mids, bin_width_l] = hist_info_l
            [dist_r_edges, dist_r_normed, dist_r_edges_mids, bin_width_r] = hist_info_r

            leg_ = ['NM', 'DM']
            f, axes = plt.subplots(
                1, 2, figsize=(10, 3.3), gridspec_kw={'width_ratios': [2, 1]}
            )
            axes[0].bar(dist_l_edges_mids, dist_l_normed, width=bin_width_l,
                        alpha=.5, color=gr_pal[1])
            axes[0].bar(dist_r_edges_mids, dist_r_normed, width=bin_width_r,
                        alpha=.5, color=gr_pal[0])
            axes[0].legend(leg_, frameon=True)
            axes[0].set_title('Max score distribution at recall')
            axes[0].set_xlabel('Recall strength')
            axes[0].set_ylabel('Probability')
            axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            axes[1].plot(fpr, tpr)
            axes[1].plot([0, 1], [0, 1], linestyle='--', color='grey')
            axes[1].set_title('ROC, AUC = %.2f' % (auc))
            axes[1].set_xlabel('FPR')
            axes[1].set_ylabel('TPR')
            f.tight_layout()
            sns.despine()
            fig_path = os.path.join(fig_dir, f'ms-dist-t-peak.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''compute inter-event similarity'''
            targ_raw = np.argmax(np.array(Y_raw), axis=-1)

            ambiguity = np.zeros((n_examples, n_event_remember - 1))
            for i in range(n_examples):
                cur_mem_ids = np.arange(i - n_event_remember + 1, i)
                for j, j_raw in enumerate(cur_mem_ids):
                    ambiguity[i, j] = compute_event_similarity(
                        targ_raw[i], targ_raw[j])

            # plot event similarity distribution
            confusion_mu = np.mean(ambiguity, axis=1)

            f, axes = plt.subplots(2, 1, figsize=(5, 6))
            sns.distplot(confusion_mu, kde=False, ax=axes[0])
            axes[0].set_ylabel('P')
            axes[0].set_xlim([0, 1])

            sns.distplot(np.ravel(ambiguity), kde=False, ax=axes[1])
            axes[1].set_xlabel('Parameter overlap')
            axes[1].set_ylabel('P')
            axes[1].set_xlim([0, 1])

            sns.despine()
            f.tight_layout()

            '''performance metrics ~ ambiguity'''
            corrects_by_cond, mistakes_by_cond, dks_by_cond = {}, {}, {}
            corrects_by_cond_mu, mistakes_by_cond_mu, dks_by_cond_mu = {}, {}, {}
            confusion_by_cond_mu = {}
            for cond_name, cond_ids_ in cond_ids.items():
                # print(cond_name, cond_ids_)
                # collect the regressor by condiiton
                confusion_by_cond_mu[cond_name] = confusion_mu[cond_ids_]
                # collect the performance metrics
                corrects_by_cond[cond_name] = corrects[cond_ids_, :]
                mistakes_by_cond[cond_name] = mistakes[cond_ids_, :]
                dks_by_cond[cond_name] = dks[cond_ids_, :]
                # compute average for the recall phase
                corrects_by_cond_mu[cond_name] = np.mean(
                    corrects_by_cond[cond_name][:, T_part:], axis=1)
                mistakes_by_cond_mu[cond_name] = np.mean(
                    mistakes_by_cond[cond_name][:, T_part:], axis=1)
                dks_by_cond_mu[cond_name] = np.mean(
                    dks_by_cond[cond_name][:, T_part:], axis=1)

            '''show regression model w/ ambiguity as the predictor
            average the performance during the 2nd part (across time)
            '''
            # predictor: inter-event similarity
            ind_var = confusion_by_cond_mu
            dep_vars = {
                'Corrects': corrects_by_cond_mu, 'Errors': mistakes_by_cond_mu,
                'Uncertain': dks_by_cond_mu
            }
            c_pal = sns.color_palette(n_colors=3)
            f, axes = plt.subplots(3, 3, figsize=(
                9, 8), sharex=True, sharey=True)
            for col_id, cond_name in enumerate(cond_ids.keys()):
                for row_id, info_name in enumerate(dep_vars.keys()):
                    sns.regplot(
                        ind_var[cond_name], dep_vars[info_name][cond_name],
                        # robust=True,
                        scatter_kws={'alpha': .5, 'marker': '.', 's': 15},
                        x_jitter=.025, y_jitter=.05,
                        color=c_pal[col_id],
                        ax=axes[row_id, col_id]
                    )
                    corr, pval = pearsonr(
                        ind_var[cond_name], dep_vars[info_name][cond_name]
                    )
                    str_ = 'r = %.2f, p = %.2f' % (corr, pval)
                    str_ = str_ + '*' if pval < .05 else str_
                    str_ = cond_name + '\n' + str_ if row_id == 0 else str_
                    axes[row_id, col_id].set_title(str_)
                    axes[row_id, 0].set_ylabel(info_name)
                    axes[row_id, col_id].set_ylim([-.05, 1.05])

                axes[-1, col_id].set_xlabel('Similarity')
            sns.despine()
            f.tight_layout()
            fig_path = os.path.join(fig_dir, f'ambiguity-by-cond.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            # '''t-RDM: raw similarity'''
            # data = C
            # trsm = {}
            # for cond_name in cond_ids.keys():
            #     if np.sum(cond_ids[cond_name]) == 0:
            #         continue
            #     else:
            #         data_cond_ = data[cond_ids[cond_name], :, :]
            #         trsm[cond_name] = compute_trsm(data_cond_)
            #
            # f, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=True)
            # for i, cond_name in enumerate(TZ_COND_DICT.values()):
            #     sns.heatmap(
            #         trsm[cond_name], cmap='viridis', square=True,
            #         xticklabels=5, yticklabels=5,
            #         ax=axes[i]
            #     )
            #     axes[i].axvline(T_part, color='red', linestyle='--')
            #     axes[i].axhline(T_part, color='red', linestyle='--')
            #     axes[i].set_title(f'TR-TR correlation, {cond_name}')
            #     axes[i].set_ylabel('Time')
            # axes[-1].set_xlabel('Time')
            # f.tight_layout()
            # fig_path = os.path.join(fig_dir, f'trdm-by-cond.png')
            # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''pca the deicison activity'''

            # n_pcs = 5
            # data = DA
            # cond_name = 'DM'
            #
            # # fit PCA
            # pca = PCA(n_pcs)
            # # np.shape(data)
            # # np.shape(data_cond)
            # data_cond = data[cond_ids[cond_name], :, :]
            # data_cond = data_cond[:, ts_predict, :]
            # targets_cond = targets[cond_ids[cond_name]]
            # mistakes_cond = mistakes_by_cond[cond_name]
            # dks_cond = dks[cond_ids[cond_name], :]
            #
            # # Loop over timepoints
            # pca_cum_var_exp = np.zeros((np.sum(ts_predict), n_pcs))
            # for t in range(np.sum(ts_predict)):
            #     data_pca = pca.fit_transform(data_cond[:, t, :])
            #     pca_cum_var_exp[t] = np.cumsum(pca.explained_variance_ratio_)
            #
            #     f, ax = plt.subplots(1, 1, figsize=(7, 5))
            #     # plot the data
            #     for y_val in range(p.y_dim):
            #         y_sel_op = y_val == targets_cond
            #         sel_op_ = np.logical_and(
            #             ~dks[cond_ids[cond_name], t], y_sel_op[:, t])
            #         ax.scatter(
            #             data_pca[sel_op_, 0], data_pca[sel_op_, 1],
            #             marker='o', alpha=alpha,
            #         )
            #     ax.scatter(
            #         data_pca[dks[cond_ids[cond_name], t], 0],
            #         data_pca[dks[cond_ids[cond_name], t], 1],
            #         marker='o', color='grey', alpha=alpha,
            #     )
            #     legend_list = [f'choice {k}' for k in range(
            #         task.y_dim)] + ['uncertain']
            #     if np.sum(mistakes_cond[:, t]) > 0:
            #         legend_list += ['error']
            #         ax.scatter(
            #             data_pca[mistakes_cond[:, t],
            #                      0], data_pca[mistakes_cond[:, t], 1],
            #             facecolors='none', edgecolors='red',
            #         )
            #     # add legend
            #     ax.legend(legend_list, fancybox=True, bbox_to_anchor=(1, .5),
            #               loc='center left')
            #     # mark the plot
            #     ax.set_xlabel('PC 1')
            #     ax.set_ylabel('PC 2')
            #     # ax.set_title(f'Pre-decision activity, time = {t}')
            #     ax.set_title(f'Decision activity')
            #     sns.despine(offset=10)
            #     f.tight_layout()

            # # plot cumulative variance explained curve
            # t = -1
            # pc_id = 1
            # f, ax = plt.subplots(1, 1, figsize=(5, 3))
            # ax.plot(pca_cum_var_exp[t])
            # ax.set_title('First %d PCs capture %d%% of variance' %
            #              (pc_id+1, pca_cum_var_exp[t, pc_id]*100))
            # ax.axvline(pc_id, color='grey', linestyle='--')
            # ax.axhline(pca_cum_var_exp[t, pc_id], color='grey', linestyle='--')
            # ax.set_ylim([None, 1.05])
            # ytickval_ = ax.get_yticks()
            # ax.set_yticklabels(['{:,.0%}'.format(x) for x in ytickval_])
            # ax.set_xticks(np.arange(n_pcs))
            # ax.set_xticklabels(np.arange(n_pcs)+1)
            # ax.set_ylabel('cum. var. exp.')
            # ax.set_xlabel('Number of components')
            # sns.despine(offset=5)
            # f.tight_layout()
            # #
            # # sns.heatmap(pca_cum_var_exp, cmap='viridis')

        # '''end of loop over subject'''

        gdata_dict = {
            'lca_param_dicts': lca_param_dicts,
            'auc_list': auc_list,
            'acc_dict': acc_dict,
            'dk_dict': dk_dict,
            'mis_dict': mis_dict,
        }
        fname = f'p{penalty_train}-{penalty_test}-data.pkl'
        gdata_outdir = 'temp/'
        pickle_save_dict(gdata_dict, os.path.join(gdata_outdir, fname))

        '''group level performance'''
        n_se = 1
        f, axes = plt.subplots(1, 3, figsize=(14, 4))
        for i, cn in enumerate(all_conds):
            if i == 0:
                add_legend = True
                legend_loc = (.285, .7)
            else:
                add_legend = False
            # plot
            vs_ = [v_ for v_ in acc_dict[cn]['mu'] if v_ is not None]
            acc_gmu_, acc_ger_ = compute_stats(vs_, n_se=n_se, axis=0)
            vs_ = [v_ for v_ in dk_dict[cn]['mu'] if v_ is not None]
            dk_gmu_ = np.mean(vs_, axis=0)
            plot_pred_acc_rcl(
                acc_gmu_[T_part:], acc_ger_[T_part:],
                acc_gmu_[T_part:] + dk_gmu_[T_part:],
                p, f, axes[i],
                title=f'{cn}',
                add_legend=add_legend, legend_loc=legend_loc,
            )
            axes[i].set_ylim([0, 1.05])
            axes[i].set_xlabel('Time, recall phase')
        fname = f'../figs/p{penalty_train}-{penalty_test}-acc.png'
        f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        ''' compute effective cmpt '''
        effective_cmpt = {cn: [] for cn in all_conds}
        for i, cn in enumerate(all_conds):
            inpt_ = lca_param_dicts[0]
            cmpt_ = lca_param_dicts[1]
            inpt_cn_ = remove_none(inpt_[cn]['mu'])
            cmpt_cn_ = remove_none(cmpt_[cn]['mu'])
            # effective_cmpt[cn] = np.array(
            #     inpt_cn_)[:, T_part:] * np.array(cmpt_cn_)[:, T_part:]
            effective_cmpt[cn] = np.array(
                inpt_cn_)[:, T_part:]
            # effective_cmpt[cn] = np.array(np.array(inpt_cn_)[:, T_part:] > .02).astype(
            #     np.float) * np.array(cmpt_cn_)[:, T_part:]

        with sns.axes_style("whitegrid"):
            mu_, se_ = compute_stats(np.sum(effective_cmpt['DM'], axis=1))
            f, ax = plt.subplots(1, 1, figsize=(3, 4))
            xticks = range(len(all_conds))
            ax.bar(
                x=0, height=mu_, yerr=se_ * n_se,
                color=sns.color_palette('colorblind')
            )
            ax.set_xticks([0])
            ax.set_xticklabels([''])
            ax.set_ylim([0, 1])
            sns.despine()
            f.tight_layout()
            fname = f'../figs/p{penalty_train}-{penalty_test}-lca-effcmpt.png'
            f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        '''group level LCA parameter by condition'''
        # lca_param_names = ['input gate', 'leak', 'competition']
        # lca_param_dicts = [inpt_dict, leak_dict, comp_dict]

        n_se = 1
        f, axes = plt.subplots(1, 2, figsize=(10, 4))
        for i, cn in enumerate(all_conds):
            for j, p_dict in enumerate(lca_param_dicts):
                p_dict_ = remove_none(p_dict[cn]['mu'])
                mu_, er_ = compute_stats(p_dict_, n_se=n_se, axis=0)
                axes[j].errorbar(
                    x=range(T_part), y=mu_[T_part:], yerr=er_[T_part:], label=f'{cn}'
                )
        axes[0].legend()
        for i, ax in enumerate(axes):
            ax.set_ylabel(lca_param_names[i])
            ax.set_xlabel('Time, recall phase')
            ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if pad_len_test > 0:
            for ax in axes:
                ax.axvline(pad_len_test, color='grey', linestyle='--')

        axes[0].set_ylim([-.05, .6])
        axes[1].set_ylim([.1, .8])

        sns.despine()
        f.tight_layout()
        fname = f'../figs/p{penalty_train}-{penalty_test}-lca.png'
        f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        '''cmpt bar plot'''
        cmpt_by_cond = {cn: None for cn in all_conds}
        cmpt_by_cond_mu = np.zeros(len(all_conds))
        cmpt_by_cond_se = np.zeros(len(all_conds))
        for i, cn in enumerate(all_conds):
            p_dict = lca_param_dicts[1]
            p_dict_ = remove_none(p_dict[cn]['mu'])
            p_dict_p2 = [p_dict_s[T_part:] for p_dict_s in p_dict_]
            cmpt_by_cond[cn] = np.array(p_dict_p2)
            cmpt_by_cond_mu[i] = np.mean(p_dict_p2)
            cmpt_by_cond_se[i] = np.std(
                p_dict_p2) / np.sqrt(np.shape(p_dict_p2)[0])

        with sns.axes_style("whitegrid"):
            f, ax = plt.subplots(1, 1, figsize=(5, 4))
            xticks = range(len(all_conds))
            ax.bar(
                x=xticks, height=cmpt_by_cond_mu, yerr=cmpt_by_cond_se * n_se,
                color=sns.color_palette('colorblind')
            )
            ax.set_xticks(xticks)
            ax.set_xticklabels(all_conds)
            ax.set_ylabel('Competition')
            ax.set_ylim([0, 1])
            sns.despine()
            f.tight_layout()

            fname = f'../figs/p{penalty_train}-{penalty_test}-lca-cmpt.png'
            f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        '''cmpt bar plot'''
        with sns.axes_style("whitegrid"):
            f, ax = plt.subplots(1, 1, figsize=(3, 4))
            xticks = range(len(all_conds))
            ax.bar(
                x=0, height=np.mean(cmpt_by_cond_mu), yerr=np.mean(cmpt_by_cond_se) * n_se,
                color=sns.color_palette('colorblind')
            )
            ax.set_xticks([0])
            ax.set_xticklabels([''])
            # ax.set_ylabel('Competition')
            ax.set_ylim([0, 1])
            sns.despine()
            f.tight_layout()

            fname = f'../figs/p{penalty_train}-{penalty_test}-lca-cmpt-mu.png'
            f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        '''group level LCA parameter by q source'''
        # sns.set(style='whitegrid', palette='colorblind', context='poster')
        n_se = 1
        f, axes = plt.subplots(2, 1, figsize=(7, 6))
        for i_p, p_name in enumerate(lca_param_names):
            for qs in ['EM only', 'both']:
                lca_param_dicts_bq_ = remove_none(
                    lca_param_dicts_bq[p_name][qs]
                )
                mu_, er_ = compute_stats(
                    lca_param_dicts_bq_, n_se=n_se, axis=0
                )
                axes[i_p].errorbar(
                    x=range(T_part), y=mu_, yerr=er_, label=qs
                )
        for i, ax in enumerate(axes):
            ax.legend()
            ax.set_ylabel(lca_param_names[i])
            ax.set_xlabel('Time, recall phase')
            ax.set_xticks(np.arange(0, p.env.n_param, 5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        sns.despine()
        f.tight_layout()

        '''group level memory activation by condition'''
        # sns.set(style='white', palette='colorblind', context='talk')
        n_se = 1
        ma_list_dict = {'lca': ma_list, 'cosine': ma_cos_list}
        for metric_name, ma_list_ in ma_list_dict.items():
            # ma_list_ = ma_list
            # ma_list_ = ma_cos_list
            # f, axes = plt.subplots(3, 1, figsize=(7, 9))
            f, axes = plt.subplots(1, 3, figsize=(14, 4))
            for i, c_name in enumerate(cond_ids.keys()):
                for m_type in memory_types:
                    if m_type == 'targ' and c_name == 'NM':
                        continue
                    color_ = gr_pal[0] if m_type == 'targ' else gr_pal[1]

                    # for the current cn - mt combination, average across people
                    y_list = []
                    for i_s, subj_id in enumerate(subj_ids):
                        if ma_list_[i_s] is not None:
                            ma_list_i_s = ma_list_[i_s]
                            y_list.append(
                                ma_list_i_s[c_name][m_type]['mu'][T_part:]
                            )
                    mu_, er_ = compute_stats(y_list, n_se=1, axis=0)
                    axes[i].errorbar(
                        x=range(T_part), y=mu_, yerr=er_,
                        label=f'{m_type}', color=color_
                    )
                axes[i].set_title(c_name)
                axes[i].set_xlabel('Time, recall phase')
            axes[0].set_ylabel('Memory activation')
            # make all ylims the same
            ylim_l, ylim_r = get_ylim_bonds(axes)
            for i, ax in enumerate(axes):
                ax.legend()
                ax.set_xlabel('Time, recall phase')
                ax.set_ylim([np.max([-.05, ylim_l]), ylim_r])
                ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                if metric_name == 'lca':
                    ax.set_yticks([0, .5])
                    ax.set_ylim([-.05, .6])
                else:
                    ax.set_yticks([0, .5, 1])
                    ax.set_ylim([-.05, 1.05])

            if pad_len_test > 0:
                for ax in axes:
                    ax.axvline(pad_len_test, color='grey', linestyle='--')
            f.tight_layout()
            sns.despine()
            fname = f'../figs/p{penalty_train}-{penalty_test}-{metric_name}-rs.png'
            f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        '''ma raw'''
        # ma_raw_list = remove_none(ma_raw_list)
        #
        # # cond_ = 'DM'
        # f, axes = plt.subplots(1, 3, figsize=(14, 4))
        # for i, cond_ in enumerate(cond_ids.keys()):
        #     if cond_ is not 'NM':
        #         targ_acts = np.array([ma_raw_i[cond_]['targ']
        #                               for ma_raw_i in ma_raw_list])
        #         targ_acts_mu = np.mean(np.mean(targ_acts, axis=0), axis=0)
        #         targ_acts_se = np.std(np.mean(targ_acts, axis=1),
        #                               axis=0) / np.sqrt(len(ma_raw_list))
        #
        #     lure_acts = np.array([ma_raw_i[cond_]['lure']
        #                           for ma_raw_i in ma_raw_list])
        #     lure_acts_mu = np.mean(np.mean(lure_acts, axis=0), axis=0)
        #     lure_acts_se = np.std(np.mean(lure_acts, axis=1),
        #                           axis=0) / np.sqrt(len(ma_raw_list))
        #
        #     alphas = [.5, 1]
        #
        #     for mid in range(2):
        #         if cond_ is not 'NM':
        #             axes[i].errorbar(
        #                 y=targ_acts_mu[T_part:, mid],
        #                 yerr=targ_acts_se[T_part:, mid],
        #                 x=range(T_part), color=gr_pal[0], alpha=alphas[mid]
        #             )
        #         axes[i].errorbar(
        #             y=lure_acts_mu[T_part:, mid],
        #             yerr=lure_acts_se[T_part:, mid],
        #             x=range(T_part), color=gr_pal[1], alpha=alphas[mid]
        #         )
        #     axes[i].set_ylim([-.05, .6])
        #     axes[i].set_xlabel('Time, recall phase')
        # axes[0].set_ylabel('Memory activation')
        # f.tight_layout()
        # sns.despine()

        '''target memory activation by q source'''
        n_se = 1
        f, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        for qs in DM_qsources:
            # remove none
            tma_dm_p2_dict_bq_ = remove_none(tma_dm_p2_dict_bq[qs])
            mu_, er_ = compute_stats(tma_dm_p2_dict_bq_, n_se=n_se, axis=0)
            ax.errorbar(
                x=range(T_part), y=mu_, yerr=er_, label=qs
            )
        ax.set_ylabel('Memory activation')
        ax.set_xlabel('Time, recall phase')
        ax.legend(['not in WM', 'in WM'])
        ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.set_yticks([0, .5])
        ax.set_ylim([-.05, .6])

        f.tight_layout()
        sns.despine()
        fname = f'../figs/p{penalty_train}-{penalty_test}-rs-dm-byq.png'
        f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        # ms_lure = get_max_score(sim_lca_dict['NM']['lure'])
        # ms_targ = get_max_score(sim_lca_dict['DM']['targ'])
        # def remove_none(input_list):
        #     return [i for i in input_list if i is not None]

        ms_lure_list = remove_none(ms_lure_list)
        ms_targ_list = remove_none(ms_targ_list)
        tpr_list = remove_none(tpr_list)
        fpr_list = remove_none(fpr_list)
        auc_list = remove_none(auc_list)
        auc_1se = np.std(auc_list) / np.sqrt(len(auc_list))
        # ms_targ_list = [i for i in ms_targ_list if i != None]
        [dist_l, dist_r], [hist_info_l, hist_info_r] = get_hist_info(
            np.concatenate(ms_lure_list),
            np.concatenate(ms_targ_list)
        )
        tpr_g, fpr_g = compute_roc(dist_l, dist_r)
        auc_g = metrics.auc(tpr_g, fpr_g)

        [dist_l_edges, dist_l_normed, dist_l_edges_mids, bin_width_l] = hist_info_l
        [dist_r_edges, dist_r_normed, dist_r_edges_mids, bin_width_r] = hist_info_r

        leg_ = ['NM', 'DM']
        f, axes = plt.subplots(
            1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]}
        )
        axes[0].bar(dist_l_edges_mids, dist_l_normed, width=bin_width_l,
                    alpha=.5, color=gr_pal[1])
        axes[0].bar(dist_r_edges_mids, dist_r_normed, width=bin_width_r,
                    alpha=.5, color=gr_pal[0])
        axes[0].legend(leg_, frameon=True)
        # axes[0].set_title('Max score distribution at recall')
        axes[0].set_xlabel('Max score')
        axes[0].set_ylabel('Probability')
        axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        axes[1].plot(fpr_g, tpr_g)
        # for tpr_list_i, fpr_list_i in zip(tpr_list, fpr_list):
        #     plt.plot(fpr_list_i, tpr_list_i, color=c_pal[0], alpha=.15)
        axes[1].plot([0, 1], [0, 1], linestyle='--', color='grey')
        axes[1].set_title('AUC = %.2f' % (np.mean(auc_list)))
        axes[1].set_xlabel('FPR')
        axes[1].set_ylabel('TPR')
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        f.tight_layout()
        sns.despine()
        fname = f'../figs/p{penalty_train}-{penalty_test}-roc.png'
        f.savefig(fname, dpi=120, bbox_to_anchor='tight')
        # fig_path = os.path.join(fig_dir, f'ms-dist-t-peak.png')
        # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

        f, ax = plt.subplots(1, 1, figsize=(5, 4))
        iv_ = (np.sum(effective_cmpt['DM'], axis=1) +
               np.sum(effective_cmpt['NM'], axis=1)) / 2
        # iv_ = np.mean(cmpt_by_cond['DM'], axis=1)

        r_val, p_val = pearsonr(iv_, np.array(auc_list))
        sns.regplot(iv_, auc_list, 'x', ax=ax)
        ax.set_xlabel('Effective competition')
        ax.set_ylabel('AUC')
        # ax.set_ylim([0.2, 1.1])
        # ax.set_xlim([0, .6])
        ax.annotate(r'$r \approx %.2f$, $p \approx %.2f$' % (r_val, p_val), xy=(
            0.05, 0.05), xycoords='axes fraction')
        sns.despine()
        f.tight_layout()
        fname = f'../figs/p{penalty_train}-{penalty_test}-reg-effcmpt-auc.png'
        f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        '''auc ~ center of mass of input gate'''
        n_se = 1
        cn = 'DM'
        p_dict_ = remove_none(lca_param_dicts[0][cn]['mu'])
        ig_p2 = np.array(p_dict_)[:, T_part:]
        ig_p2_norm = ig_p2 / np.sum(ig_p2, axis=1, keepdims=True)

        rt = ig_p2 * (np.arange(T_part) + 1)
        mrt = np.mean(rt, axis=1)
        r_val, p_val = pearsonr(mrt, np.array(auc_list))

        f, ax = plt.subplots(1, 1, figsize=(5, 4))
        sns.regplot(mrt, auc_list)
        ax.set_xlabel('Recall time')
        ax.set_ylabel('AUC')
        ax.annotate(r'$r \approx %.2f$, $p \approx %.2f$' % (r_val, p_val), xy=(
            0.05, 0.05), xycoords='axes fraction')
        sns.despine()
        f.tight_layout()
        fname = f'../figs/p{penalty_train}-{penalty_test}-reg-rt-auc.png'
        f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        # f, axes = plt.subplots(1, 2, figsize=(10, 4))
        # for i, cn in enumerate(all_conds):
        #     for j, p_dict in enumerate(lca_param_dicts):
        #         p_dict_ = remove_none(p_dict[cn]['mu'])
        #         mu_, er_ = compute_stats(p_dict_, n_se=n_se, axis=0)
        #         axes[j].errorbar(
        #             x=range(T_part), y=mu_[T_part:], yerr=er_[T_part:], label=f'{cn}'
        #         )
        # axes[0].legend()
        # for i, ax in enumerate(axes):
        #     ax.set_ylabel(lca_param_names[i])
        #     ax.set_xlabel('Time, recall phase')
        #     ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param-1))
        #     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # if pad_len_test > 0:
        #     for ax in axes:
        #         ax.axvline(pad_len_test, color='grey', linestyle='--')
        #
        # axes[0].set_ylim([-.05, .6])
        # axes[1].set_ylim([.1, .8])
        #
        # sns.despine()
        # f.tight_layout()

        '''p(error | schema consistent) vs. p(error | schema inconsistent)'''

        mis_dict_mu = {cn: None for cn in all_conds}
        missing_ids = {cn: None for cn in all_conds}
        for i, cn in enumerate(all_conds):
            # print(cn)
            mis_dict_mu[cn], missing_ids[cn] = remove_none(
                mis_dict[cn]['mu'], return_missing_idx=True
            )
            mis_dict_mu[cn] = np.array(mis_dict_mu[cn])[:, T_part:]


#         if len(missing_ids['DM']) > 0:
            # this is wrong, earlier poped ideas shift the indexing
#             for missing_subject_id in missing_ids['DM']:
#                 def_tps_list.pop(missing_subject_id)
#
#         sc_sel_op = np.array(def_tps_list).astype(np.bool)
#         f, axes = plt.subplots(1, 3, figsize=(17, 5))
#         for i, cn in enumerate(all_conds):
#             mis_gsc, mis_gsic = [], []
#             for i_s in range(len(mis_dict_mu[cn])):
#                 mis_gsc.append(mis_dict_mu[cn][i_s][sc_sel_op[i_s]])
#                 mis_gsic.append(mis_dict_mu[cn][i_s][~sc_sel_op[i_s]])
#             mis_gsc_mu, mis_gsc_se = compute_stats(np.mean(mis_gsc, axis=1))
#             mis_gsic_mu, mis_gsic_se = compute_stats(np.mean(mis_gsic, axis=1))
#
#             heights = [mis_gsc_mu, mis_gsic_mu]
#             yerrs = [mis_gsc_se, mis_gsic_se]
#             xticklabels = ['schematic', 'nonschematic']
#             xticks = range(len(heights))
#
#             # f, ax = plt.subplots(1, 1, figsize=(6, 5))
#             axes[i].bar(
#                 x=xticks, height=heights, yerr=yerrs,
#                 color=sns.color_palette('colorblind')[3]
#             )
#             axes[i].axhline(0, color='grey', linestyle='--')
#             axes[i].set_title(cn)
#             axes[i].set_xlabel('Transition type')
#             axes[i].set_xticks(xticks)
#             axes[i].set_ylim([-.05, .5])
#             axes[i].set_xticklabels(xticklabels)
#             axes[i].set_ylabel('P(error)')
#             f.tight_layout()
#             sns.despine()
#
#         fname = f'../figs/p{penalty_train}-{penalty_test}-error-schema-effect.png'
#         f.savefig(fname, dpi=120, bbox_to_anchor='tight')
# #
#         '''input gate | schema consistent vs. schema inconsistent'''
#         def split_data_byschema(inp_data):
#             d_gsc, d_gsic = [], []
#             for i_s in range(len(inp_data)):
#                 d_gsc.append(inp_data[i_s][sc_sel_op[i_s]])
#                 d_gsic.append(inp_data[i_s][~sc_sel_op[i_s]])
#             d_gsc_mu, d_gsc_se = compute_stats(np.mean(d_gsc, axis=1))
#             d_gsic_mu, d_gsic_se = compute_stats(np.mean(d_gsic, axis=1))
#             heights = [d_gsc_mu, d_gsic_mu]
#             yerrs = [d_gsc_se, d_gsic_se]
#             return heights, yerrs
#
#         n_se = 1
#         # f, axes = plt.subplots(1, 2, figsize=(10, 4))
#         f, axes = plt.subplots(1, 3, figsize=(17, 5))
#         for i, cn in enumerate(all_conds):
#             ig = lca_param_dicts[0][cn]['mu']
#             ig = np.array(remove_none(ig))[:, T_part:]
#             heights, yerrs = split_data_byschema(ig)
#             # f, ax = plt.subplots(1, 1, figsize=(6, 5))
#             axes[i].bar(
#                 x=xticks, height=heights, yerr=yerrs,
#                 color=sns.color_palette('colorblind')[0]
#             )
#             axes[i].axhline(0, color='grey', linestyle='--')
#             axes[i].set_title(cn)
#             axes[i].set_xlabel('Transition type')
#             axes[i].set_xticks(xticks)
#             axes[i].set_ylim([-.025, .2])
#             axes[i].set_xticklabels(xticklabels)
#             axes[i].set_ylabel('Input gate')
#             f.tight_layout()
#             sns.despine()
#
#         # f, axes = plt.subplots(1, 2, figsize=(10, 4))
#         f, axes = plt.subplots(1, 3, figsize=(17, 5))
#         for i, cn in enumerate(all_conds):
#             ig = lca_param_dicts[1][cn]['mu']
#             ig = np.array(remove_none(ig))[:, T_part:]
#             heights, yerrs = split_data_byschema(ig)
#             # f, ax = plt.subplots(1, 1, figsize=(6, 5))
#             axes[i].bar(
#                 x=xticks, height=heights, yerr=yerrs,
#                 color=sns.color_palette('colorblind')[0]
#             )
#             axes[i].axhline(0, color='grey', linestyle='--')
#             axes[i].set_title(cn)
#             axes[i].set_xlabel('Transition type')
#             axes[i].set_xticks(xticks)
#             axes[i].set_ylim([-.05, .9])
#             axes[i].set_xticklabels(xticklabels)
#             axes[i].set_ylabel('Competition')
#             f.tight_layout()
#             sns.despine()
#
#     # xticklabels = ['schematic', 'nonschematic']
#     # xticks = range(len(heights))
#
#
# # plt.plot(np.sum(np.array(def_tps_list),axis=0))

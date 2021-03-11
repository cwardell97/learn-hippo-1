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
from sklearn.decomposition.pca import PCA

warnings.filterwarnings("ignore")
sns.set(style='white', palette='colorblind', context='poster')
gr_pal = sns.color_palette('colorblind')[2:4]
log_root = '/Users/carsonwardell/Desktop/Thesis/log/'
# log_root = '/tigress/qlu/logs/learn-hippocampus/log'
all_conds = TZ_COND_DICT.values()


# exp_name = 'vary-schema-level'
# def_prob_range = np.arange(.25, 1, .1)
# for def_prob in def_prob_range:

# the name of the experiemnt
exp_name = 'vary-test-penalty'
# exp_name = 'familiarity-signal'
subj_ids = np.arange(15)
penalty_random = 0
def_prob = .25
n_def_tps = 0
# n_def_tps = 8
# loading params
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = 0
attach_cond = 0
supervised_epoch = 600
epoch_load = 1000
learning_rate = 7e-4
n_branch = 4
n_param = 16
enc_size = 16
n_event_remember = 2
comp_val = .8
leak_val = 0
# test param
penaltys_train = [4]
penaltys_test = np.array([0, 2, 4])
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

n_subjs = len(subj_ids)
DM_qsources = ['EM only', 'both']
memory_types = ['targ', 'lure']

if not os.path.isdir(f'../figs/{exp_name}'):
    os.makedirs(f'../figs/{exp_name}')


def prealloc_stats():
    return {cond: {'mu': [None] * n_subjs, 'er': [None] * n_subjs}
            for cond in all_conds}


for penalty_train in penaltys_train:
    penaltys_test_ = penaltys_test[penaltys_test <= penalty_train]
    if penalty_random == 1:
        penaltys_test_ = penaltys_test[penaltys_test <= penalty_train]
    else:
        penaltys_test_ = [penalty_train]
    for penalty_test in penaltys_test_:
        print(
            f'penalty_train={penalty_train}, penalty_test={penalty_test}')

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
        ig_dm_p2_dict_bq = {qs: [None] * n_subjs for qs in DM_qsources}
        q_source_list = [None] * n_subjs
        ms_lure_list = [None] * n_subjs
        ms_targ_list = [None] * n_subjs
        tpr_list = [None] * n_subjs
        fpr_list = [None] * n_subjs
        auc_list = [None] * n_subjs

        inpt_dmp2_g = [None] * n_subjs
        actions_dmp2_g = [None] * n_subjs
        targets_dmp2_g = [None] * n_subjs
        def_path_int_g = [None] * n_subjs
        def_tps_g = [None] * n_subjs

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
                p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
            )
            # create logging dirs
            test_params = [penalty_test, pad_len_test, slience_recall_time]
            log_path, log_subpath = build_log_path(
                subj_id, p, log_root=log_root, mkdir=False)
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

            # make fig dir
            fig_dir = os.path.join(log_subpath['figs'], test_data_subdir)
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            test_data_dict = pickle_load_dict(fpath)
            [dist_a_, Y_, log_cache_, log_cond_] = test_data_dict['results']
            [X_raw, Y_raw] = test_data_dict['XY']

            '''precompute some constants'''

            def_path_int = np.argmax(def_path, axis=1)
            is_def_tp = np.array(def_tps).astype(np.bool)
            # figure out max n-time-steps across for all trials
            T_part = n_param + pad_len_test
            T_total = T_part * task.n_parts
            ts_predict = np.array(
                [t % T_part >= pad_len_test for t in range(T_total)])

            '''organize results to analyzable form'''
            # skip examples untill EM is full
            n_examples_skip = compute_n_trials_to_skip(log_cond_, p)
            n_trials = n_examples_test - n_examples_skip
            trial_id = np.arange(n_trials)

            data_to_trim = [dist_a_, Y_, log_cond_, log_cache_, X_raw]
            [dist_a, Y, log_cond, log_cache, X_raw] = trim_data(
                n_examples_skip, data_to_trim)
            X_raw = np.array(X_raw)

            # process the data
            cond_ids = get_trial_cond_ids(log_cond)
            [C, H, M, CM, DA, V], [inpt] = process_cache(
                log_cache, T_total, p)
            # compute ground truth / objective uncertainty, delay phase removed
            true_dk_wm, true_dk_em = batch_compute_true_dk(X_raw, task)
            q_source = get_qsource(true_dk_em, true_dk_wm, cond_ids, p)

            # load lca params
            comp = np.full(np.shape(inpt), comp_val)
            leak = np.full(np.shape(inpt), leak_val)

            # compute performance
            actions = np.argmax(dist_a, axis=-1)
            targets = np.argmax(Y, axis=-1)
            corrects = targets == actions
            dks = actions == p.dk_id
            mistakes = np.logical_and(targets != actions, ~dks)

            # split data wrt p1 and p2
            CM_p1, CM_p2 = CM[:, :T_part, :], CM[:, T_part:, :]
            DA_p1, DA_p2 = DA[:, :T_part, :], DA[:, T_part:, :]
            X_raw_p1, X_raw_p2 = X_raw[:, :T_part, :], X_raw[:, T_part:, :]
            corrects_p2 = corrects[:, T_part:]
            mistakes_p1 = mistakes[:, :T_part]
            mistakes_p2 = mistakes[:, T_part:]
            dks_p1, dks_p2 = dks[:, :T_part], dks[:, T_part:]
            inpt_p2 = inpt[:, T_part:]
            targets_p1 = targets[:, :T_part]
            targets_p2 = targets[:, T_part:]
            actions_p1 = actions[:, :T_part]
            actions_p2 = actions[:, T_part:]

            # pre-extract p2 data for the DM condition
            corrects_dmp2 = corrects_p2[cond_ids['DM']]
            mistakes_dmp2 = mistakes_p2[cond_ids['DM']]
            mistakes_dmp1 = mistakes_p1[cond_ids['DM']]
            dks_dmp2 = dks_p2[cond_ids['DM']]
            CM_dmp2 = CM_p2[cond_ids['DM']]
            DA_dmp2 = DA_p2[cond_ids['DM']]

            inpt_dmp2 = inpt_p2[cond_ids['DM']]
            targets_dmp1 = targets_p1[cond_ids['DM'], :]
            targets_dmp2 = targets_p2[cond_ids['DM'], :]
            actions_dmp1 = actions_p1[cond_ids['DM']]
            actions_dmp2 = actions_p2[cond_ids['DM']]

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

            # save info to do schema analysis
            inpt_dmp2_g[i_s] = inpt_dmp2
            actions_dmp2_g[i_s] = actions_dmp2
            targets_dmp2_g[i_s] = targets_dmp2
            def_path_int_g[i_s] = def_path_int
            def_tps_g[i_s] = def_tps

            # save enc data, probably don't need to do it here
            # input_dict = {'Y': Y, 'dist_a': dist_a, 'cond_ids': cond_ids}
            # pickle_save_dict(input_dict, f'data/enc{enc_size_test}.pkl')

            # compute performance stats
            for i, cn in enumerate(all_conds):
                Y_ = Y[cond_ids[cn], :]
                dist_a_ = dist_a[cond_ids[cn], :]
                # compute performance for this condition
                acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
                dk_mu = compute_dk(dist_a_)
                mis_mu, mis_er = compute_mistake(
                    Y_, dist_a_, return_er=True)

                # cache data for all cond-subj
                acc_dict[cn]['mu'][i_s] = acc_mu
                acc_dict[cn]['er'][i_s] = acc_er
                mis_dict[cn]['mu'][i_s] = mis_mu
                mis_dict[cn]['er'][i_s] = mis_er
                dk_dict[cn]['mu'][i_s] = dk_mu

            '''plot behavioral performance'''
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

            '''compare norm
            '''
            data_ = DA_p2
            data_name = 'DA_p2'

            dk_norms = np.linalg.norm(data_[dks_p2], axis=1)
            ndk_norms = np.linalg.norm(data_[~dks_p2], axis=1)
            dk_norm_mu, dk_norm_se = compute_stats(dk_norms)
            ndk_norm_mu, ndk_norm_se = compute_stats(ndk_norms)

            f, ax = plt.subplots(1, 1, figsize=(4.5, 4))
            xticklabels = ['Uncertain', 'Certain']
            xticks = range(len(xticklabels))
            ax.bar(x=xticks, height=[dk_norm_mu, ndk_norm_mu],
                   yerr=np.array([dk_norm_se, ndk_norm_se]) * 3)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_ylabel('Activity norm')
            f.tight_layout()
            sns.despine()
            fname = fname = f'../figs/{exp_name}/state-norm-mu.png'
            f.savefig(fname, dpi=100, bbox_to_anchor='tight')

            '''compute cell-memory similarity / memory activation '''
            lca_param_names = ['input gate', 'competition']
            lca_param_dicts = [inpt_dict, comp_dict]
            lca_param_records = [inpt, comp]
            for i, cn in enumerate(all_conds):
                for p_dict, p_record in zip(lca_param_dicts, lca_param_records):
                    p_dict[cn]['mu'][i_s], p_dict[cn]['er'][i_s] = compute_stats(
                        p_record[cond_ids[cn]])

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
                        avg_ma[cond][m_type] = np.mean(
                            sim_lca_dict[cond][m_type], axis=-1)

            '''plot target/lure activation for all conditions - horizontal'''
            ylim_bonds = {'LCA': None, 'cosine': None}
            ker_name, sim_stats_plt_ = 'LCA', sim_lca_stats
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
            ylim_bonds[ker_name] = (
                np.max((ylim_bonds[ker_name][0], -.05)
                       ), np.round((ylim_bonds[ker_name][1] + .1), decimals=1)
            )
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
            fig_path = os.path.join(
                fig_dir, f'tz-memact-{ker_name}-hori.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''use CURRENT uncertainty to predict memory activation'''
            n_se = 1
            cond_name = 'DM'
            targ_act_cond_p2_stats = sep_by_qsource(
                avg_ma[cond_name]['targ'][:, T_part + pad_len_test:],
                q_source[cond_name],
                n_se=n_se
            )
            ig_cond_p2_stats = sep_by_qsource(
                lca_param_records[0][cond_ids[cond_name]
                                     ][:, T_part + pad_len_test:],
                q_source[cond_name],
                n_se=n_se
            )

            for qs in DM_qsources:
                tma_dm_p2_dict_bq[qs][i_s] = targ_act_cond_p2_stats[qs][0]
                ig_dm_p2_dict_bq[qs][i_s] = ig_cond_p2_stats[qs][0]

            f, ax = plt.subplots(1, 1, figsize=(6, 4))
            for key, [mu_, er_] in targ_act_cond_p2_stats.items():
                if not np.all(np.isnan(mu_)):
                    ax.errorbar(x=range(n_param), y=mu_,
                                yerr=er_, label=key)
            ax.set_title(f'Target memory activation, {cond_name}')
            ax.set_xlabel('Time (part 2)')
            ax.set_ylabel('Activation')
            ax.set_ylim([-.05, None])
            ax.set_xticks([0, p.env.n_param - 1])
            ax.legend(['not already observed',
                       'already observed'], fancybox=True)
            # ax.legend([])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            f.tight_layout()
            sns.despine()
            fig_path = os.path.join(
                fig_dir, f'tma-{cond_name}-by-qsource.png')
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

            [dist_l_edges, dist_l_normed, dist_l_edges_mids,
                bin_width_l] = hist_info_l
            [dist_r_edges, dist_r_normed, dist_r_edges_mids,
                bin_width_r] = hist_info_r

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

            '''pca the deicison activity'''
            n_pcs = 8
            data = DA
            cond_name = 'DM'

            # fit PCA
            pca = PCA(n_pcs)
            data_cond = data[cond_ids[cond_name], :, :]
            data_cond = data_cond[:, ts_predict, :]
            targets_cond = targets[cond_ids[cond_name]]
            dks_cond = dks[cond_ids[cond_name], :]

            # Loop over timepoints
            pca_cum_var_exp = np.zeros((np.sum(ts_predict), n_pcs))
            for t in range(np.sum(ts_predict)):
                data_pca = pca.fit_transform(data_cond[:, t, :])
                pca_cum_var_exp[t] = np.cumsum(
                    pca.explained_variance_ratio_)

                f, ax = plt.subplots(1, 1, figsize=(8, 5))
                alpha = .5
                # plot the data
                for y_val in range(p.y_dim):
                    y_sel_op = y_val == targets_cond
                    sel_op_ = np.logical_and(
                        ~dks[cond_ids[cond_name], t], y_sel_op[:, t])
                    ax.scatter(
                        data_pca[sel_op_, 0], data_pca[sel_op_, 1],
                        marker='o', alpha=alpha,
                    )
                ax.scatter(
                    data_pca[dks[cond_ids[cond_name], t], 0],
                    data_pca[dks[cond_ids[cond_name], t], 1],
                    marker='o', color='grey', alpha=alpha,
                )
                legend_list = [f'choice {k}' for k in range(
                    task.y_dim)] + ['uncertain']

                ax.legend(legend_list, fancybox=True, bbox_to_anchor=(1.2, .5),
                          loc='center left')
                # mark the plot
                ax.set_xlabel('PC 1')
                ax.set_ylabel('PC 2')

                ax.set_title(
                    f'Decision activity \npart {int(np.ceil(t/T_part))}, t={t%T_part}')
                sns.despine(offset=10)
                f.tight_layout()
                fig_path = os.path.join(fig_dir, f'pca-t{t}.png')
                f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

        '''end of loop over subject'''
        gdata_dict = {
            'lca_param_dicts': lca_param_dicts,
            'auc_list': auc_list,
            'acc_dict': acc_dict,
            'dk_dict': dk_dict,
            'mis_dict': mis_dict,
            'lca_ma_list': ma_list,
            'cosine_ma_list': ma_cos_list,
            'inpt_dmp2_g': inpt_dmp2_g,
            'actions_dmp2_g': actions_dmp2_g,
            'targets_dmp2_g': targets_dmp2_g,
            'def_path_int_g': def_path_int_g,
            'def_tps_g': def_tps_g

        }
        fname = '%s-dp%.2f-p%d-%d.pkl' % (
            exp_name, def_prob, penalty_train, penalty_test)
        gdata_outdir = 'data/'
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
            axes[i].set_xlabel('Time (part 2)')
        fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-acc.png'
        f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        '''group level input gate by condition'''
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

        n_se = 1
        f, ax = plt.subplots(1, 1, figsize=(
            5 * (pad_len_test / n_param + 1), 4))
        for i, cn in enumerate(['RM', 'DM']):
            p_dict = lca_param_dicts[0]
            p_dict_ = remove_none(p_dict[cn]['mu'])
            mu_, er_ = compute_stats(p_dict_, n_se=n_se, axis=0)
            ax.errorbar(
                x=np.arange(T_part) - pad_len_test, y=mu_[T_part:], yerr=er_[T_part:], label=f'{cn}'
            )

        ax.legend()
        ax.set_ylim([-.05, .9])
        ax.set_ylabel(lca_param_names[0])
        ax.set_xlabel('Time (part 2)')
        ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if pad_len_test > 0:
            ax.axvline(0, color='grey', linestyle='--')
        sns.despine()
        f.tight_layout()
        fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-ig-nonm.png'
        f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        '''group level memory activation by condition'''
        n_se = 1
        ma_list_dict = {'lca': ma_list, 'cosine': ma_cos_list}
        for metric_name, ma_list_ in ma_list_dict.items():
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
                axes[i].set_xlabel('Time (part 2)')
            axes[0].set_ylabel('Memory activation')
            # make all ylims the same
            ylim_l, ylim_r = get_ylim_bonds(axes)
            for i, ax in enumerate(axes):
                ax.legend()
                ax.set_xlabel('Time (part 2)')
                ax.set_ylim([np.max([-.05, ylim_l]), ylim_r])
                ax.set_xticks(
                    np.arange(0, p.env.n_param, p.env.n_param - 1))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                if metric_name == 'lca':
                    # ax.set_yticks([0, .2, .4])
                    # ax.set_ylim([-.01, .45])
                    ax.set_yticks([0, .3, .6])
                    ax.set_ylim([-.01, .8])
                else:
                    ax.set_yticks([0, .5, 1])
                    ax.set_ylim([-.05, 1.05])

            if pad_len_test > 0:
                for ax in axes:
                    ax.axvline(pad_len_test, color='grey', linestyle='--')
            f.tight_layout()
            sns.despine()

            fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-{metric_name}-rs.png'
            f.savefig(fname, dpi=120, bbox_to_anchor='tight')

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
        ax.set_xlabel('Time (part 2)')
        ax.legend(['not already observed', 'already observed'])
        ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_yticks([0, .2, .4])
        ax.set_ylim([-.01, .45])
        f.tight_layout()
        sns.despine()
        fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-rs-dm-byq.png'
        f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        # make a no legend version
        f, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        for qs in DM_qsources:
            # remove none
            tma_dm_p2_dict_bq_ = remove_none(tma_dm_p2_dict_bq[qs])
            mu_, er_ = compute_stats(tma_dm_p2_dict_bq_, n_se=n_se, axis=0)
            ax.errorbar(
                x=range(T_part), y=mu_, yerr=er_, label=qs
            )
        ax.set_ylabel('Memory activation')
        ax.set_xlabel('Time (part 2)')
        # ax.legend(['not recently observed', 'recently observed'])
        ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_yticks([0, .2, .4])
        ax.set_ylim([-.01, .45])
        f.tight_layout()
        sns.despine()
        fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-rs-dm-byq-noleg.png'
        f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        '''target memory activation by q source'''
        n_se = 1
        f, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        for qs in DM_qsources:
            # remove none
            ig_dm_p2_dict_bq_ = remove_none(ig_dm_p2_dict_bq[qs])
            mu_, er_ = compute_stats(ig_dm_p2_dict_bq_, n_se=n_se, axis=0)
            ax.errorbar(
                x=range(T_part), y=mu_, yerr=er_, label=qs
            )
        ax.set_ylabel('Input gate')
        ax.set_xlabel('Time (part 2)')
        # ax.legend(['not already observed', 'already observed'])
        ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_yticks([0, .3, .6])
        ax.set_ylim([-.01, .7])
        f.tight_layout()
        sns.despine()
        fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-ig-dm-byq.png'
        f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        f, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        ax.plot(0, 0)
        ax.plot(0, 1)
        ax.legend(['not already observed', 'already observed'])
        sns.despine()
        fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-observed-not-legend.png'
        f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        # '''auc'''
        # ms_lure_list = remove_none(ms_lure_list)
        # ms_targ_list = remove_none(ms_targ_list)
        # tpr_list = remove_none(tpr_list)
        # fpr_list = remove_none(fpr_list)
        # auc_list = remove_none(auc_list)
        # auc_1se = np.std(auc_list) / np.sqrt(len(auc_list))
        # # ms_targ_list = [i for i in ms_targ_list if i != None]
        # [dist_l, dist_r], [hist_info_l, hist_info_r] = get_hist_info(
        #     np.concatenate(ms_lure_list),
        #     np.concatenate(ms_targ_list)
        # )
        # tpr_g, fpr_g = compute_roc(dist_l, dist_r)
        # auc_g = metrics.auc(tpr_g, fpr_g)
        #
        # [dist_l_edges, dist_l_normed, dist_l_edges_mids, bin_width_l] = hist_info_l
        # [dist_r_edges, dist_r_normed, dist_r_edges_mids, bin_width_r] = hist_info_r

        # leg_ = ['NM', 'DM']
        # f, axes = plt.subplots(
        #     1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]}
        # )
        # axes[0].bar(dist_l_edges_mids, dist_l_normed, width=bin_width_l,
        #             alpha=.5, color=gr_pal[1])
        # axes[0].bar(dist_r_edges_mids, dist_r_normed, width=bin_width_r,
        #             alpha=.5, color=gr_pal[0])
        # axes[0].legend(leg_, frameon=True)
        # axes[0].set_xlabel('Max score')
        # axes[0].set_ylabel('Probability')
        # axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #
        # axes[1].plot(fpr_g, tpr_g)
        # axes[1].plot([0, 1], [0, 1], linestyle='--', color='grey')
        # axes[1].set_title('AUC = %.2f' % (np.mean(auc_list)))
        # axes[1].set_xlabel('FPR')
        # axes[1].set_ylabel('TPR')
        # axes[1].set_xticks([0, 1])
        # axes[1].set_yticks([0, 1])
        # f.tight_layout()
        # sns.despine()
        # fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-roc.png'
        # f.savefig(fname, dpi=120, bbox_to_anchor='tight')

        # '''auc ~ center of mass of input gate'''
        # n_se = 1
        # cn = 'DM'
        # p_dict_ = remove_none(lca_param_dicts[0][cn]['mu'])
        # ig_p2 = np.array(p_dict_)[:, T_part:]
        # ig_p2_norm = ig_p2
        # rt = np.dot(ig_p2_norm, (np.arange(T_part) + 1))
        # r_val, p_val = pearsonr(rt, np.array(auc_list))
        #
        # f, ax = plt.subplots(1, 1, figsize=(5, 4))
        # sns.regplot(rt, auc_list)
        # ax.set_xlabel('Recall time')
        # ax.set_ylabel('AUC')
        # ax.annotate(r'$r \approx %.2f$, $p \approx %.2f$' % (r_val, p_val), xy=(
        #     0.05, 0.05), xycoords='axes fraction')
        # sns.despine()
        # f.tight_layout()
        # fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-reg-rt-auc.png'
        # f.savefig(fname, dpi=120, bbox_to_anchor='tight')

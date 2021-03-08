





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

import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

from models import LCALSTM as Agent
from task import SequenceLearning
from exp_ms import run_ms
from analysis import compute_stats, compute_behav_metrics, compute_acc, compute_dk, compute_stats_max
from vis import plot_pred_acc_full
from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, save_ckpt, save_all_params, pickle_save_dict, get_test_data_dir, get_test_data_fname, load_env_metadata, pickle_load_dict, load_ckpt

plt.switch_backend('agg')
sns.set(style='white', palette='colorblind', context='talk')

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='test', type=str)
parser.add_argument('--subj_id', default=99, type=int)
parser.add_argument('--n_param', default=16, type=int)
parser.add_argument('--n_branch', default=4, type=int)
parser.add_argument('--pad_len', default=-1, type=int)
parser.add_argument('--def_prob', default=None, type=float)
parser.add_argument('--n_def_tps', default=0, type=int)
parser.add_argument('--enc_size', default=16, type=int)
parser.add_argument('--penalty', default=5, type=int)
parser.add_argument('--penalty_random', default=1, type=int)
parser.add_argument('--penalty_discrete', default=1, type=int)
parser.add_argument('--penalty_onehot', default=0, type=int)
parser.add_argument('--normalize_return', default=1, type=int) #set to zero --> raw reward/penalty
parser.add_argument('--attach_cond', default=0, type=int)
parser.add_argument('--p_rm_ob_enc', default=0.3, type=float)
parser.add_argument('--p_rm_ob_rcl', default=0, type=float)
parser.add_argument('--similarity_max', default=.9, type=float)
parser.add_argument('--similarity_min', default=0, type=float)
parser.add_argument('--n_hidden', default=194, type=int)
parser.add_argument('--n_hidden_dec', default=128, type=int)
parser.add_argument('--lr', default=7e-4, type=float)
parser.add_argument('--eta', default=0.1, type=float)
parser.add_argument('--cmpt', default=0.8, type=float)
parser.add_argument('--n_event_remember', default=2, type=int)
parser.add_argument('--sup_epoch', default=1, type=int)
parser.add_argument('--n_epoch', default=5, type=int)
parser.add_argument('--n_examples', default=256, type=int) #256
parser.add_argument('--log_root', default='../log/', type=str)
args = parser.parse_args()
print(args)

# process args
exp_name = args.exp_name
subj_id = args.subj_id
n_param = args.n_param
n_branch = args.n_branch
pad_len = args.pad_len
def_prob = args.def_prob
n_def_tps = args.n_def_tps
enc_size = args.enc_size
penalty = args.penalty
penalty_random = args.penalty_random
penalty_discrete = args.penalty_discrete
penalty_onehot = args.penalty_onehot
normalize_return = args.normalize_return
attach_cond = args.attach_cond
p_rm_ob_enc = args.p_rm_ob_enc
p_rm_ob_rcl = args.p_rm_ob_rcl
similarity_max = args.similarity_max
similarity_min = args.similarity_min
n_hidden = args.n_hidden
n_hidden_dec = args.n_hidden_dec
learning_rate = args.lr
cmpt = args.cmpt
eta = args.eta
n_event_remember = args.n_event_remember
n_examples = args.n_examples
n_epoch = args.n_epoch
supervised_epoch = args.sup_epoch
log_root = args.log_root
# specify epoch to load
epoch_load = 1000

'''init'''
seed_val = subj_id
np.random.seed(seed_val)
torch.manual_seed(seed_val)

p = P(
    exp_name=exp_name, subj_id=subj_id,
    sup_epoch=supervised_epoch,
    n_param=n_param, n_branch=n_branch, pad_len=pad_len,
    def_prob=def_prob, n_def_tps=n_def_tps,
    enc_size=enc_size, n_event_remember=n_event_remember,
    penalty=penalty, penalty_random=penalty_random,
    penalty_discrete=penalty_discrete, penalty_onehot=penalty_onehot,
    normalize_return=normalize_return, attach_cond=attach_cond,
    p_rm_ob_enc=p_rm_ob_enc, p_rm_ob_rcl=p_rm_ob_rcl,
    n_hidden=n_hidden, n_hidden_dec=n_hidden_dec,
    lr=learning_rate, eta=eta, cmpt=cmpt,
)

# init env
task = SequenceLearning(
    n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=p.env.pad_len,
    p_rm_ob_enc=p.env.p_rm_ob_enc, p_rm_ob_rcl=p.env.p_rm_ob_rcl,
    def_path=p.env.def_path, def_prob=p.env.def_prob, def_tps=p.env.def_tps,
    similarity_cap_lag=p.n_event_remember,
    similarity_max=similarity_max, similarity_min=similarity_min,
)
x_dim = task.x_dim
if attach_cond != 0:
    x_dim += 1
# init agent
agent = Agent(
    input_dim=x_dim, output_dim=p.a_dim,
    rnn_hidden_dim=p.net.n_hidden, dec_hidden_dim=p.net.n_hidden_dec,
    dict_len=p.net.dict_len, cmpt=p.net.cmpt
)

# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)

'''
# create path for pretrained model storage
pad_len_test = 0
p_rm_ob = 0
n_examples_test = 256
fix_cond = None
slience_recall_time = None
scramble = False

test_params = [penalty, pad_len, slience_recall_time]
test_data_dir, _ = get_test_data_dir(
    log_subpath, epoch_load, test_params)
test_data_fname = get_test_data_fname(
    n_examples_test, fix_cond, scramble)

if enc_size != enc_size:
    test_data_dir = os.path.join(
        test_data_dir, f'enc_size_test-{enc_size_test}'
    )
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
fpath = os.path.join(test_data_dir, test_data_fname)
'''


'''
# hardcode pretrained model filepath (local)
tpath = '/Users/carsonwardell/Desktop/Thesis/log/training-models-local/p-16_b-4_pad-random/tp-0.25/p_rm_ob_rcl-0.00_enc-0.30/lp-4/enc-cum_size-16/nmem-2/rp-LCA_metric-cosine/h-194_hdec-128/lr-0.0007-eta-0.1/sup_epoch-600/subj-1/data/epoch-1000/penalty-2/delay-0/srt-None/n256.pkl'
train_logsubpath = {'ckpts': '/Users/carsonwardell/Desktop/Thesis/log/training-models-local/p-16_b-4_pad-random/tp-0.25/p_rm_ob_rcl-0.00_enc-0.30/lp-4/enc-cum_size-16/nmem-2/rp-LCA_metric-cosine/h-194_hdec-128/lr-0.0007-eta-0.1/sup_epoch-600/subj-1/ckpts', 'data': '/Users/carsonwardell/Desktop/Thesis/log/training-models-local/p-16_b-4_pad-random/tp-0.25/p_rm_ob_rcl-0.00_enc-0.30/lp-4/enc-cum_size-16/nmem-2/rp-LCA_metric-cosine/h-194_hdec-128/lr-0.0007-eta-0.1/sup_epoch-600/subj-1/data', 'figs': '/Users/carsonwardell/Desktop/Thesis/log/training-models-local/p-16_b-4_pad-random/tp-0.25/p_rm_ob_rcl-0.00_enc-0.30/lp-4/enc-cum_size-16/nmem-2/rp-LCA_metric-cosine/h-194_hdec-128/lr-0.0007-eta-0.1/sup_epoch-600/subj-1/figs'}

'''
# hardcode pretrained model filepath (cluster)
tpath = '/tigress/cwardell/logs/learn-hippocampus/log/training-models/p-16_b-4_pad-random/tp-0.25/p_rm_ob_rcl-0.00_enc-0.30/lp-4/enc-cum_size-16/nmem-2/rp-LCA_metric-cosine/h-194_hdec-128/lr-0.0007-eta-0.1/sup_epoch-600/subj-1/data/epoch-1000/penalty-2/delay-0/srt-None/n256.pkl'
train_logsubpath = {'ckpts': '/tigress/cwardell/logs/learn-hippocampus/log/training-models/p-16_b-4_pad-random/tp-0.25/p_rm_ob_rcl-0.00_enc-0.30/lp-4/enc-cum_size-16/nmem-2/rp-LCA_metric-cosine/h-194_hdec-128/lr-0.0007-eta-0.1/sup_epoch-600/subj-1/ckpts', 'data': '/tigress/cwardell/logs/learn-hippocampus/log/training-models/p-16_b-4_pad-random/tp-0.25/p_rm_ob_rcl-0.00_enc-0.30/lp-4/enc-cum_size-16/nmem-2/rp-LCA_metric-cosine/h-194_hdec-128/lr-0.0007-eta-0.1/sup_epoch-600/subj-1/data', 'figs': '/tigress/cwardell/logs/learn-hippocampus/log/training-models/p-16_b-4_pad-random/tp-0.25/p_rm_ob_rcl-0.00_enc-0.30/lp-4/enc-cum_size-16/nmem-2/rp-LCA_metric-cosine/h-194_hdec-128/lr-0.0007-eta-0.1/sup_epoch-600/subj-1/figs'}





# load model
agent, optimizer = load_ckpt(
    epoch_load, train_logsubpath['ckpts'], agent)
optimizer= torch.optim.Adam(agent.parameters(), lr=p.net.lr)

# init scheduler REMOVE comment
scheduler_rl = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=1 / 2, patience=30, threshold=1e-3, min_lr=1e-8,
    verbose=True)

# if data dir does not exsits ... skip
if agent is None:
    print('Agent DNE')

# freeze memory controlling layer
for param in agent.parameters():
    param.requires_grad_ = False
agent.hpc.requires_grad_ = True

# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)
# save experiment params initial weights
save_all_params(log_subpath['data'], p)
save_ckpt(0, log_subpath['ckpts'], agent, optimizer)

'''task definition'''
log_freq = 200
Log_loss_critic = np.zeros(n_epoch,)
Log_loss_actor = np.zeros(n_epoch,)
Log_loss_sup = np.zeros(n_epoch,)
Log_return = np.zeros(n_epoch,)
Log_pi_ent = np.zeros(n_epoch,)
Log_acc = np.zeros((n_epoch, task.n_parts))
Log_mis = np.zeros((n_epoch, task.n_parts))
Log_dk = np.zeros((n_epoch, task.n_parts))
Log_cond = np.zeros((n_epoch, n_examples))
log_cache = np.zeros((n_epoch, n_param))
# simulation lengths
av_sims_lengs = np.zeros(n_epoch)
all_sims_lengs = np.zeros((n_epoch, n_examples))
av_epoch_reward = np.zeros(n_epoch)
av_epoch_ep_reward = np.zeros(n_epoch)
av_epoch_ms_reward = np.zeros(n_epoch)
# sim composition
av_mem1_matches_e = np.zeros(n_epoch)
av_mem2_matches_e = np.zeros(n_epoch)
av_no_matches_e = np.zeros(n_epoch)
av_step_num_e = np.zeros(n_epoch)
av_both_matches_e = np.zeros(n_epoch)


k = 2
epoch_id = 0
for epoch_id in np.arange(epoch_id, n_epoch):
    time0 = time.time()
    print("epoch: ", epoch_id)

    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    [results, metrics, sims_data,
    reward_data, sim_origins] = run_ms(
        agent, optimizer,
        task, p, n_examples, tpath,
        fix_penalty=penalty, get_cache=True,
        learning=True, get_data=True, seed_num=2,
        mem_num=2, counter_fact=False
    )

    # unpack output
    [dist_a, targ_a, log_cache[epoch_id], Log_cond[epoch_id]] = results
    print("t-s-l, log_cache shape:", np.shape(log_cache))
    [Log_loss_sup[epoch_id], Log_loss_actor[epoch_id], Log_loss_critic[epoch_id],
    Log_return[epoch_id], Log_pi_ent[epoch_id]] = metrics
    [av_sims_data, all_sims_data] = sims_data
    [av_epoch_reward[epoch_id]]= reward_data
    [av_mem1_matches,av_mem2_matches,
    av_no_matches, av_both_matches] = sim_origins

    # assign to logs
    av_sims_lengs[epoch_id] = av_sims_data
    all_sims_lengs[epoch_id] = all_sims_data
    #av_epoch_ep_reward[epoch_id] = av_ep_reward
    #av_epoch_ms_reward[epoch_id] = av_reward - av_ep_reward

    # save sim lengths
    #print("av_mem1_matches: ", av_mem1_matches)
    #print("av_mem2_matches: ", av_mem2_matches)
    #print("av_no_matches: ", av_no_matches)
    av_mem1_matches_e[epoch_id] = av_mem1_matches
    av_mem2_matches_e[epoch_id] = av_mem2_matches
    av_no_matches_e[epoch_id] = av_no_matches
    av_both_matches_e[epoch_id] = av_both_matches

    #update lr scheduler
    neg_pol_score = (n_param - av_sims_lengs[epoch_id])/n_param
    scheduler_rl.step(neg_pol_score)



    # save weights
    if np.mod(epoch_id + 1, log_freq) == 0:
        save_ckpt(epoch_id + 1, log_subpath['ckpts'], agent, optimizer)

    # kludge
    slience_recall_time = None
    # save the data
    test_params = [penalty, pad_len, slience_recall_time]
    test_data_dir, _ = get_test_data_dir(
        log_subpath, epoch_load, test_params)
    test_data_fname = get_test_data_fname(
        n_examples, None, False)
    test_data_dict = {'results': results, 'metrics': metrics,
    'av_sims_data': av_sims_data, 'all_sims_data': all_sims_data}
    fpath = os.path.join(test_data_dir, test_data_fname)
    pickle_save_dict(test_data_dict, fpath)



'''plot learning curves'''
f, ax = plt.subplots(figsize=(10, 9)) #, sharex=True)
ax.plot(av_sims_lengs, label = 'sim_lengths')
ax.set_ylabel('sim length', color = 'blue')
ax.axhline(0, color='grey', linestyle='--')
ax.set_xlabel('epoch')
ax2 = ax.twinx()

ax2.plot(av_epoch_reward, color = 'red', label = 'total reward')
#ax2.plot(av_epoch_ep_reward, color = 'green', label = 'e.p. reward')
#ax2.plot(av_epoch_ms_reward, color = 'orange', label = 'm.s. reward')
ax2.set_ylabel("average reward")
ax2.legend()


f2, axes2 = plt.subplots(figsize=(10, 9)) #, sharex=True)
axes2.plot(range(n_examples), all_sims_lengs[1,:])
axes2.set_ylabel('sim length')
axes2.axhline(0, color='grey', linestyle='--')
axes2.set_xlabel('trial')

# sim composition
f3, axes3 = plt.subplots(figsize=(20, 10)) #, sharex=True)
axes3.plot(av_mem1_matches_e, label = 'origin: memory 1')
axes3.plot(av_mem2_matches_e, label = 'origin: memory 2')
axes3.plot(av_both_matches_e, label = 'origin: both memories')
axes3.plot(av_no_matches_e, label = 'origin: NA')

print("size mem1 matches:", av_mem1_matches_e)
print("size mem2 matches:", av_mem2_matches_e)
print("size both matches:", av_both_matches_e)
print("size no matches:", av_no_matches_e)
print("first entry:", av_mem1_matches_e[0])



axes3.set_ylabel('% of total instances per epoch')
axes3.set_xlabel('epoch')
axes3.legend()

f4, axes4 = plt.subplots(figsize=(10, 9)) #, sharex=True)
axes4.plot(range(n_examples), all_sims_lengs[n_epoch-1,:])
axes4.set_ylabel('sim length')
axes4.axhline(0, color='grey', linestyle='--')
axes4.set_xlabel('trial')

n_se = 1
f5, ax = plt.subplots(2, 1, figsize=(5, 4))

mu_, er_ = compute_stats(log_cache, n_se=n_se, axis=0)
ax[0].errorbar(
    x=np.arange(n_param), y=mu_, yerr=er_
    )
#ax[0].legend()
ax[0].set_ylim([-.05, .7])
ax[0].set_ylabel('input gate value')
ax[0].set_xlabel('Time')
ax[0].set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
#ax[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
#ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].axvline(0, color='grey', linestyle='--')

mu_, er_ = compute_stats_max(log_cache, n_se=n_se, axis=1)
ax[1].errorbar(
    x=np.arange(n_epoch), y=mu_, yerr=er_
    )
#ax[1].set_ylim([-.05, .7])
ax[1].set_ylabel('av max input gate value')
ax[1].set_xlabel('epochs')

sns.despine()
f5.tight_layout()
f.tight_layout()

# create fig paths and save
fig1_path = os.path.join(log_subpath['figs'], 'tz-lc.png')
fig2_path = os.path.join(log_subpath['figs'], 'first_epoch_sims.png')
fig4_path = os.path.join(log_subpath['figs'], 'last_epoch_sims.png')
fig3_path = os.path.join(log_subpath['figs'], 'sim_composition.png')
fig5_path = os.path.join(log_subpath['figs'], 'inpt_gate.png')

f.savefig(fig1_path, dpi=100, bbox_to_anchor='tight')
f2.savefig(fig2_path, dpi=100, bbox_to_anchor='tight')
f3.savefig(fig3_path, dpi=100, bbox_to_anchor='tight')
f4.savefig(fig4_path, dpi=100, bbox_to_anchor='tight')
f5.savefig(fig5_path, dpi=100, bbox_to_anchor='tight')






'''plot performance --> just plot don't knows
# prep data
cond_ids = {}
for cond_name_ in list(TZ_COND_DICT.values()):
    cond_id_ = TZ_COND_DICT.inverse[cond_name_]
    cond_ids[cond_name_] = Log_cond[-1, :] == cond_id_
    targ_a_ = targ_a[cond_ids[cond_name_], :]
    dist_a_ = dist_a[cond_ids[cond_name_], :]
    # compute performance for this condition
    acc_mu, acc_er = compute_acc(targ_a_, dist_a_, return_er=True)
    dk_mu = compute_dk(dist_a_)
    f, ax = plt.subplots(1, 1, figsize=(7, 4))
    plot_pred_acc_full(
        acc_mu, acc_er, acc_mu + dk_mu,
        [n_param], p,
        f, ax,
        title=f'Performance on the TZ task: {cond_name_}',
    )
    fig_path = os.path.join(log_subpath['figs'], f'tz-acc-{cond_name_}.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
'''

'''eval the model
pad_len_test = 0
p_rm_ob = 0
n_examples_test = 256
fix_cond = None
slience_recall_time = None
scramble = False
epoch_load = epoch_id + 1

np.random.seed(seed_val)
torch.manual_seed(seed_val)
task = SequenceLearning(
    n_param=p.env.n_param, n_branch=p.env.n_branch,
    def_path=p.env.def_path, def_prob=p.env.def_prob, def_tps=p.env.def_tps,
    pad_len=pad_len_test, p_rm_ob_enc=p_rm_ob, p_rm_ob_rcl=p_rm_ob,
    similarity_max=similarity_max, similarity_min=similarity_min,
    similarity_cap_lag=p.n_event_remember,
)

for fix_penalty in np.arange(0, penalty + 1, 2):
    [results, metrics, sims_data] = run_ms(
    agent, optimizer,
    task, p, n_examples_test, tpath,
    fix_penalty=penalty, slience_recall_time=slience_recall_time,
    learning=False, get_data=True,
    )

# save the data
test_params = [fix_penalty, pad_len_test, slience_recall_time]
test_data_dir, _ = get_test_data_dir(
    log_subpath, epoch_load, test_params)
test_data_fname = get_test_data_fname(
    n_examples_test, fix_cond, scramble)
test_data_dict = {'results': results, 'metrics': metrics,
'sims_data': sims_data}
fpath = os.path.join(test_data_dir, test_data_fname)
pickle_save_dict(test_data_dict, fpath)
'''

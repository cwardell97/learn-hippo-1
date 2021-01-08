import os
import torch
import numpy as np
import argparse

from itertools import product
from models import LCALSTM as Agent
from task import SequenceLearning
from exp_ms import run_ms
from utils.params import P
from utils.io import build_log_path, load_ckpt, pickle_save_dict, \
    get_test_data_dir, get_test_data_fname, load_env_metadata, pickle_load_dict

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
parser.add_argument('--penalty', default=4, type=int)
parser.add_argument('--penalty_random', default=1, type=int)
parser.add_argument('--penalty_discrete', default=1, type=int)
parser.add_argument('--penalty_onehot', default=0, type=int)
parser.add_argument('--normalize_return', default=1, type=int)
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
parser.add_argument('--n_epoch', default=2, type=int)
parser.add_argument('--n_examples', default=256, type=int)
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



'''loop over conditions for testing'''

slience_recall_times = [range(n_param), None]

subj_ids = np.arange(1)

penaltys_train = [4]
penaltys_test = np.array([2])

all_conds = ['DM']

for slience_recall_time in slience_recall_times:
    for subj_id, penalty_train, fix_cond in product(subj_ids, penaltys_train, all_conds):
        print(
            f'\nsubj : {subj_id}, penalty : {penalty_train}, cond : {fix_cond}')
        print(f'slience_recall_time : {slience_recall_time}')

        penaltys_test_ = penaltys_test[penaltys_test <= penalty_train]
        for fix_penalty in penaltys_test_:
            print(f'penalty_test : {fix_penalty}')

            p = P(
                exp_name=exp_name, sup_epoch=supervised_epoch,
                n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
                def_prob=def_prob, n_def_tps=n_def_tps,
                enc_size=enc_size, n_event_remember=n_event_remember,
                penalty=penalty_train, penalty_random=penalty_random,
                attach_cond=attach_cond,
                p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
            )
            # create logging dirs
            log_path, log_subpath = build_log_path(
                subj_id, p, log_root=log_root, mkdir=False, verbose=False
            )

            # create fpath
            test_params = [fix_penalty, pad_len_test, slience_recall_time]
            test_data_dir, _ = get_test_data_dir(
                log_subpath, epoch_load, test_params)
            test_data_fname = get_test_data_fname(
                n_examples_test, fix_cond, scramble)

            if enc_size_test != enc_size:
                test_data_dir = os.path.join(
                    test_data_dir, f'enc_size_test-{enc_size_test}'
                )
                if not os.path.exists(test_data_dir):
                    os.makedirs(test_data_dir)
            fpath = os.path.join(test_data_dir, test_data_fname)

            # init env
            env_data = load_env_metadata(log_subpath)
            def_path = env_data['def_path']
            p.env.def_path = def_path
            p.update_enc_size(enc_size_test)



            task = SequenceLearning(
                n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
                p_rm_ob_enc=p_rm_ob_enc_test, p_rm_ob_rcl=p_rm_ob_rcl_test,
                similarity_max=similarity_max_test, similarity_min=similarity_min_test,
                similarity_cap_lag=p.n_event_remember,
            )
            x_dim = task.x_dim
            if attach_cond != 0:
                x_dim += 1
            # load the agent back
            agent = Agent(
                input_dim=x_dim, output_dim=p.a_dim,
                rnn_hidden_dim=p.net.n_hidden, dec_hidden_dim=p.net.n_hidden_dec,
                dict_len=p.net.dict_len
            )

            agent, optimizer = load_ckpt(
                epoch_load, log_subpath['ckpts'], agent)

            # if data dir does not exsits ... skip
            if agent is None:
                print('Agent DNE')
                continue

            # agent.parameters

            # freeze memory controlling layer
            for param in agent.parameters():
                param.requires_grad = False
            agent.hpc.requires_grad_ = True


            # training objective
            np.random.seed(seed)
            torch.manual_seed(seed)
            [results, metrics, XY] = run_ms(
                agent, optimizer,
                task, p, n_examples_test, fpath,
                fix_penalty=fix_penalty, slience_recall_time=slience_recall_time,
                learning=False, get_data=True,
            )

            # save the data
            test_params = [fix_penalty, pad_len_test, slience_recall_time]
            test_data_dir, _ = get_test_data_dir(
                log_subpath, epoch_load, test_params)
            test_data_fname = get_test_data_fname(
                n_examples_test, fix_cond, scramble)
            test_data_dict = {
                'results': results, 'metrics': metrics, 'XY': XY
            }
            if enc_size_test != enc_size:
                test_data_dir = os.path.join(
                    test_data_dir, f'enc_size_test-{enc_size_test}'
                )
                if not os.path.exists(test_data_dir):
                    os.makedirs(test_data_dir)
            fpath = os.path.join(test_data_dir, test_data_fname)
            pickle_save_dict(test_data_dict, fpath)







'''
playing with data:
print(training_data)

X = np.array(training_data[0])
Y = np.array(training_data[1])

X.shape
Y.shape


import random as rd

i = rd.randint(0,(n_examples_test-1))
print(i)
X_i = X[i,:,:]
Y_i = Y[i,:,:]

Y_i.shape[1]
X_i.shape[0]
import matplotlib.pyplot as plt
plt.imshow(X_i)
plt.imshow(Y_i)

j = rd.randint(0,(X_i.shape[0]-1))
X_i_t0 = X_i[j,:]
'''

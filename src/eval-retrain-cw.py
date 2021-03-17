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

#plt.switch_backend('agg')
#sns.set(style='white', palette='colorblind', context='talk')

''' remove parser capability for now
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
'''

log_root = '/Users/carsonwardell/Desktop/Thesis/log/'

exp_name = 'vary-test-penalty(trained)'
def_prob = None
n_def_tps = 0

seed = 0
supervised_epoch = 600
epoch_load = 1000

n_branch = 4
n_param = 16
enc_size = 16
# enc_size_test = 8
enc_size_test = enc_size
n_event_remember = 2

penalty_random = 1
# testing param, ortho to the training directory
attach_cond = 0
# loading params
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = .3

# testing params
pad_len_test = 0
p_test = 0
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test
n_examples_test = 256

similarity_max_test = .9
similarity_min_test = 0

'''loop over conditions for testing'''

slience_recall_times = [range(n_param), None]

subj_ids = np.arange(1)

penaltys_train = [4]
penaltys_test = np.array([2])

all_conds = ['DM']
scramble = False

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
                subj_id, p, log_root=log_root, mkdir=False, verbose=False)

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
            print(fpath)
            print(test_data_fname)
            print(test_data_dir)
            print(log_subpath)
            print(fpath.split("sup_epoch-600",1)[1])


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








#playing with data:
print(fpath)
import pickle
data = pickle.load(open('/Users/carsonwardell/Desktop/Thesis/n256.pkl', "rb"))

sims_data = data.pop('sims_data')

sims_data_1 = sims_data[0]
sims_data_2 = sims_data[1]
sims_data_1[0]
np.shape(sims_data_2)
plt.plot(sims_data_2)












A= 3
T= 16
t= 14
a_t = 1

from utils.utils import to_sqnp

time_oh_vec = np.identity(T)[t%T]
time_q_vec = np.identity(T)[t%T]
# special case if dk
if a_t == A:
    obs_val_oh_vec =  np.zeros(A)
else:
    obs_val_oh_vec = np.identity(A)[a_t]

retrn1 = np.concatenate([time_oh_vec,obs_val_oh_vec, time_oh_vec])
retrn1.shape
np.shape(to_sqnp(retrn1))


boop = np.reshape(retrn1, (np.shape(retrn1)[0],1))
boop = np.reshape(obs_val_oh_vec, (np.shape(obs_val_oh_vec)[0],1))
boop = np.reshape(time_oh_vec, (np.shape(time_oh_vec)[0],1))
plt.imshow(boop.T)
boop.shape
print(np.shape(time_oh_vec))
print(np.shape(obs_val_oh_vec))

np.shape(np.concatenate([time_oh_vec,obs_val_oh_vec]))

t=2
boop = []
for i in range(t,5):
    boop.append(0)

boop

np.shape(retrn1)
trans_retrn = np.atleast_2d(retrn).T.shape
np.shape(trans_retrn)
log_rets = np.empty(trans_retrn.shape)
np.atleast_2d(retrn).T.shape
log_rets.shape
log_rets = np.vstack([log_rets, retrn])
log_rets = np.vstack([log_rets, retrn1])
log_rets.shape[1]
plt.imshow(log_rets)

training_data = pickle_load_dict(fpath).pop('XY')
X_train = np.array(training_data[0])
Y_train = np.array(training_data[1])

X = np.array(training_data[0])
Y = np.array(training_data[1])

X.shape
Y.shape

import random as rd

i = rd.randint(0,(n_examples_test-1))
print("i:",i)
X_i = X_train[i,:,:]
Y_i = Y_train[i,:,:]

Y_i.shape#[1]
X_i.shape#[0]
import matplotlib.pyplot as plt
plt.imshow(X_i)
plt.imshow(Y_i)

Y_i[13].shape[0]

torch.from_numpy(Y_i)
torch.argmax(torch.from_numpy(Y_i))

j = rd.randint(0,(X_i.shape[0]-1))
X_i_t0 = X_i[j,:]


probs, rewards, values, ents = [], [], [], []
a = torch.empty(1)
a.add_(1)
print(a)

rewards.add(a)

print(rewards)

deep = np.random.choice(15,2, replace=False)


k = 2
n_examples = 256

data_sample_ind=np.random.choice(n_examples,k,replace = False)
X_dict = {}
Y_dict = {}
i=0
for k in data_sample_ind:
    X_dict["X_{0}".format(i)] = "Hello{0}".format(k)
    Y_dict["Y_{0}".format(i)] = "Hello{0}".format(k)
    i+=1
    print(i)
X_dict

X_dict["X_{0}".format(0)]

k=2
for i in range(k):
    print(i)

seed_num = 2
range(1,seed_num)
j = np.random.choice(seed_num)
print(j)

for t in range(16):
    if t<(seed_num):
        print(t)

for sn in range(seed_num):
    print(sn)




def io_convert(a_t, t, T, A):
    '''converts model output to one-hot vector input format
    '''
    time_oh_vec = np.identity(T)[t%T]
    time_q_vec = np.identity(T)[t%T]
    # special case if dk
    if a_t == A:
        obs_val_oh_vec =  np.zeros(A)
    else:
        obs_val_oh_vec = np.identity(A)[a_t]
    return np.concatenate([time_oh_vec,obs_val_oh_vec, time_oh_vec])

A = np.array([0., 0., 0., 1.]).shape[0]
print(A)
T = 16
t = 1
a_t = 3

io_convert(a_t, t, T, A)
# play with results
dpath = '/Users/carsonwardell/Desktop/Thesis/log/subj-1(working-2:22)'

mod_data = pickle_load_dict(dpath)

seed_num = 2
X_dict = [0,1]
print(len(X_dict))
j = np.random.choice(len(X_dict))
print("j", j)

T_part = 16
for t in range(T_part):
    print(t


deep = np.zeros(40)

deep_short = deep[:20,]

np.shape(deep_short)

# compare two X_array_list

outp = np.array([1,1,0,0])
fakemem = np.array([1,0,0,0])
fakemem2 = np.array([1,0,0,0])

if not np.all(outp == fakemem) and not np.all(outp == fakemem2):
    print("other")

mem_1_comp

print(np.sum([[3],[1],[0]]))



rewards = [torch.tensor(-5.)]
R = 0
gamma=0
returns = []
for r in rewards[::-1]:
    R = r + gamma * R
    returns.insert(0, R)
returns = torch.tensor(returns)
# normalize w.r.t to the statistics of this trajectory

print(returns)

deep = np.zeros((5,16))
print(np.shape(np.amax(deep, axis = 1)))

mem_num = 2
for mn in range(mem_num):
    print(mn)

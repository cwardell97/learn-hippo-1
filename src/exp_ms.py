import torch
import numpy as np
import torch.nn.functional as F
import pdb
import random as rd

from torch.autograd import Variable
from analysis import entropy
from utils.utils import to_sqnp
from utils.constants import TZ_COND_DICT, P_TZ_CONDS
from task.utils import scramble_array, scramble_array_list
from models import compute_returns, compute_a2c_loss, get_reward_ms, get_reward
from utils.io import pickle_load_dict
#from torchsummary import summary

''' check that supervised and cond_i have all been removed '''

def run_ms(
        agent, optimizer, task, p, n_examples, fpath,
        fix_penalty=None, slience_recall_time=None,
        learning=True, get_cache=True, get_data=True,
        counter_fact=False, seed_num=2, mem_num=2
):

    # load training data
    training_data = pickle_load_dict(fpath).pop('XY')
    X_train = np.array(training_data[0])
    Y_train = np.array(training_data[1])

    # logger
    log_return, log_pi_ent = 0, 0
    log_loss_sup, log_loss_actor, log_loss_critic = 0, 0, 0
    log_cond = np.zeros(n_examples,)
    log_dist_a = [[] for _ in range(n_examples)]
    log_targ_a = [[] for _ in range(n_examples)]
    log_cache = [None] * n_examples
    log_X = []
    log_sim_lengths = np.zeros(n_examples)
    av_a_t = np.zeros(n_examples)
    av_reward = np.zeros(n_examples)

    # note that first and second half of x is redudant, only need to show half
    for i in range(n_examples):
        log_a_t = []
        #pdb.set_trace()
        # pick a condition
        cond_i = 'DM'
        # cond_i = pick_condition(p, rm_only=supervised, fix_cond=fix_cond)
        # get the example for this trial
        #X_i, Y_i = X[i], Y[i]
        # load a single example (maybe just go through iteratively)
        #j = rd.randint(0,(n_examples_test-1))

        # sample k X & Ys randomly without replacement
        data_sample_ind=np.random.choice(n_examples,mem_num,replace=False)
        X_dict = {}
        Y_dict = {}
        l = 0
        for k in data_sample_ind:
            X_dict["X_{0}".format(l)] = X_train[k,:,:]
            Y_dict["Y_{0}".format(l)] = Y_train[k,:,:]
            l+=1

        # get time info
        T_total = np.shape(X_dict["X_{0}".format(1)])[0]
        T_part, pad_len, event_ends, event_bonds = task.get_time_param(T_total)
        enc_times = get_enc_times(p.net.enc_size, task.n_param, pad_len)


        # attach cond flag
        cond_flag = torch.zeros(T_total, 1)
        cond_indicator = -1 if cond_i == 'NM' else 1
        # if attach_cond == 1 then normal, if -1 then reversed
        cond_flag[-T_part:] = cond_indicator * p.env.attach_cond
        if p.env.attach_cond != 0:
            X_i = torch.cat((X_i, cond_flag), 1)

        # prealloc
        loss_sup = 0
        probs, rewards, values, ents = [], [], [], []
        log_cache_i = [None] * T_total

        # init model wm and em
        penalty_val_p1, penalty_rep_p1 = sample_penalty(p, fix_penalty, True)
        #penalty_val_p2, penalty_rep_p2 = sample_penalty(p, fix_penalty) REMOVE

        hc_t = agent.get_init_states()
        agent.retrieval_off()
        agent.encoding_off()

        '''sample simulation seeds'''
        # init seed dicts
        seed_dictX = {}
        seed_dictY = {}
        pair_nums = np.random.choice(T_part,seed_num,replace=False)

        if not counter_fact:
            # rand X,Y from X_dict/Y_dict for seeds
            j = np.random.choice(seed_num)
            X_j = X_dict["X_{0}".format(j)]
            Y_j = Y_dict["Y_{0}".format(j)]
            # select k f/v pairs from within X
            seed_dict = {}

            for sn in range(seed_num):
                # load X,Y for specific events
                seed_dictX["seed_X{0}".format(sn)] = X_j[pair_nums[sn]]
                seed_dictY["seed_Y{0}".format(sn)] = Y_j[pair_nums[sn]]

        # count_fact --> sample seeds from both events (CHANGE FOR memnum>2)
        else:
            for sn in range(seed_num):
                # even num draw from mem1
                if (sn%2) == 0:
                    seed_dictX["seed_X{0}".format(sn)] = X_dict["X_{0}".format(0)][pair_nums[sn]]
                    seed_dictY["seed_Y{0}".format(sn)] = Y_dict["Y_{0}".format(0)][pair_nums[sn]]
                # odd from mem2
                else:
                    seed_dictX["seed_X{0}".format(sn)] = X_dict["X_{0}".format(1)][pair_nums[sn]]
                    seed_dictY["seed_Y{0}".format(sn)] = Y_dict["Y_{0}".format(1)][pair_nums[sn]]


        ''' Load em with 'mem_num' events'''
        for mn in range(mem_num):
            #print("mem_num: ", mn)
            # load X,Y for specific events
            X_mn = X_dict["X_{0}".format(mn)]
            Y_mn = Y_dict["Y_{0}".format(mn)]
            for t in range(T_part):
                t_relative = t % T_part
                #in_2nd_part = t >= T_part REMOVE

                #if not in_2nd_part: REMOVE
                penalty_val, penalty_rep = penalty_val_p1, penalty_rep_p1

                #else: REMOVE
                #    penalty_val, penalty_rep = penalty_val_p2, penalty_rep_p2

                # testing condition
                if slience_recall_time is not None:
                    slience_recall(t_relative, in_2nd_part,
                                   slience_recall_time, agent)
                # whether to encode

                set_encoding_flag(t, enc_times, cond_i, agent)

                torch_X_mn = torch.from_numpy(X_mn[t])
                # forward
                x_it = append_prev_info(torch_X_mn, [penalty_rep])
                pi_a_t, v_t, hc_t, cache_t = agent.forward(
                    x_it.view(1, 1, -1), hc_t)
                # after delay period, compute loss
                a_t, p_a_t = agent.pick_action(pi_a_t)
                # get reward

                r_t = get_reward(a_t, Y_mn[t], penalty_val)
                # cache the results for later RL loss computation REMOVE
                rewards.append(r_t)
                values.append(v_t)
                probs.append(p_a_t)
                ents.append(entropy(pi_a_t))


                #update WM/EM bsaed on the condition (flushing WM / retrieve during part 2)
                hc_t = cond_manipulation(
                    cond_i, t, event_ends[0], hc_t, agent)

                # cache results for later analysis REMOVE
                if get_cache:
                    log_cache_i[t] = cache_t
                # for behavioral stuff, only record prediction time steps DELETE
                '''if t % T_part >= pad_len:
                    log_dist_a[i].append(to_sqnp(pi_a_t))
                    log_targ_a[i].append(to_sqnp(Y_i[t]))'''


        '''seed simulation, then predict'''
        for t in range(T_part):
            global X_i_t
            # init X_i_t @t=0
            if t==0:
                X_i_t = np.zeros(seed_dictX["seed_X{0}".format(0)].shape)
            if t<(seed_num-1):
                # do event prediction while t<k, ie during seeding expect last
                t_relative = t % T_part
                #in_2nd_part = t >= T_part REMOVE

                #if not in_2nd_part: REMOVE
                penalty_val, penalty_rep = penalty_val_p1, penalty_rep_p1

                #else: REMOVE
                #    penalty_val, penalty_rep = penalty_val_p2, penalty_rep_p2

                # testing condition
                if slience_recall_time is not None:
                    slience_recall(t_relative, in_2nd_part,
                                   slience_recall_time, agent)

                # whether to encode
                set_encoding_flag(t, enc_times, cond_i, agent)

                # forward (CHANGE: might need torch conversion)
                torch_x_i_t = torch.from_numpy(seed_dictX["seed_X{0}".format(t)])
                x_it = append_prev_info(torch_x_i_t.type(torch.FloatTensor),
                [penalty_rep]
                )

                pi_a_t, v_t, hc_t, cache_t = agent.forward(
                    x_it.view(1, 1, -1), hc_t)
                # after delay period, compute loss
                a_t, p_a_t = agent.pick_action(pi_a_t)
                # get reward
                r_t = get_reward(a_t, seed_dictY["seed_Y{0}".format(t)],
                penalty_val
                )

                # cache the results for later RL loss computation REMOVE
                rewards.append(r_t)
                probs.append(p_a_t)
                ents.append(entropy(pi_a_t))
                log_a_t.append(a_t)

            elif (seed_num-1)==t:

                # add in case for t=k, for first seed, but also sims_data
                # whether to encode
                set_encoding_flag(t, enc_times, cond_i, agent)

                # forward (CHANGE: might need torch conversion)
                torch_x_i_t = torch.from_numpy(seed_dictX["seed_X{0}".format(t)])
                x_it = append_prev_info(torch_x_i_t.type(torch.FloatTensor),
                [penalty_rep]
                )

                pi_a_t, v_t, hc_t, cache_t = agent.forward(
                    x_it.view(1, 1, -1), hc_t)
                # after delay period, compute loss
                a_t, p_a_t = agent.pick_action(pi_a_t)
                # get reward
                r_t = get_reward_ms(a_t, seed_dictY["seed_Y{0}".format(0)],
                penalty_val
                )

                # cache the results for later RL loss computation REMOVE
                rewards.append(r_t)
                probs.append(p_a_t)
                ents.append(entropy(pi_a_t))
                log_a_t.append(a_t)
                # convert model prediction to input for next timesteps
                #print("n_param: ", p.env.n_param)

                X_i_t = io_convert(a_t, t, p.env.n_param,
                seed_dictY["seed_Y{0}".format(0)].shape[0]
                )
                # if don't know, break
                if seed_dictY["seed_Y{0}".format(0)].shape[0] == a_t:
                    break

            else:
                # now just predict for rest, once k<t
                #start prediction task immediately after the last seed (@t=k)
                penalty_val, penalty_rep = penalty_val_p1, penalty_rep_p1

                # testing condition
                if slience_recall_time is not None:
                    slience_recall(t_relative, in_2nd_part,
                                   slience_recall_time, agent)
                # whether to encode
                set_encoding_flag(t, enc_times, cond_i, agent)

                torch_x_i_t = torch.from_numpy(X_i_t)
                # forward
                x_it = append_prev_info(torch_x_i_t.type(torch.FloatTensor), [penalty_rep])
                pi_a_t, v_t, hc_t, cache_t = agent.forward(
                    x_it.view(1, 1, -1), hc_t)
                # after delay period, compute loss
                a_t, p_a_t = agent.pick_action(pi_a_t)
                # get reward, use first Y_seed for DK length arg
                r_t = get_reward_ms(a_t, seed_dictY["seed_Y{0}".format(0)], penalty_val)

                torch.set_printoptions(profile="full")


                # convert model output to onehotinput for t+1 (action, time, total timesteps, total vals)
                # print("yshape: ", seed_dictY["seed_Y{0}".format(0)].shape)DELETE
                X_i_t = io_convert(a_t, t, p.env.n_param,
                seed_dictY["seed_Y{0}".format(0)].shape[0]
                )

                # cache the results for later RL loss computation
                rewards.append(r_t)
                values.append(v_t)
                probs.append(p_a_t)
                ents.append(entropy(pi_a_t))
                log_a_t.append(a_t)
                # compute supervised loss (Don't understand, I can remove, right?)
                #yhat_t = torch.squeeze(pi_a_t)[:-1]
                #loss_sup += F.mse_loss(yhat_t, torch.from_numpy(Y_i[t]))

                # if not supervised:
                # update WM/EM bsaed on the condition
                hc_t = cond_manipulation(
                    cond_i, t, event_ends[0], hc_t, agent)

                # cache results for later analysis
                if get_cache:
                    log_cache_i[t] = cache_t
                # for behavioral stuff, only record prediction time steps
                '''if t % T_part >= pad_len:
                    log_dist_a[i].append(to_sqnp(pi_a_t))
                    log_targ_a[i].append(to_sqnp(torch.from_numpy(Y_i[t])))'''

                # if don't know, break, except if its the first two!
                if seed_dictY["seed_Y{0}".format(0)].shape[0] == a_t:
                    #for j in range(t,T_total):
                    #    log_dist_a[i].append(0)
                    #    log_targ_a[i].append(0)
                    print("breaking on: ", t)
                    break
        # log sim length after t loop
        log_sim_lengths[i] = t





        # compute RL loss (just merge these together from two tasks)
        returns = compute_returns(rewards, normalize=p.env.normalize_return)
        #print("rewards:", rewards)
        #print("values:", values)
        #print("probs:", probs)
        loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
        pi_ent = torch.stack(ents).sum()
        # if learning and not supervised
    #    print("example num", i)


        '''for prm in agent.parameters():
            #print(prm)
            if prm.requires_grad==True:
                print("name:", prm.name)
                print("data:", prm.data)'''

        print("agent hpc req grad: ", agent.hpc.requires_grad_ )
        if learning:
            loss = loss_actor + loss_critic - pi_ent * p.net.eta
            print("loss", loss)
            optimizer.zero_grad()
            #loss = Variable(loss, requires_grad = True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1)
            optimizer.step()

        # after every event sequence, log stuff
        log_loss_sup += loss_sup / n_examples
        log_pi_ent += pi_ent.item() / n_examples
        log_return += torch.stack(rewards).sum().item() / n_examples
        log_loss_actor += loss_actor.item() / n_examples
        log_loss_critic += loss_critic.item() / n_examples
        log_cond[i] = TZ_COND_DICT.inverse[cond_i]
        if get_cache:
            log_cache[i] = log_cache_i

        av_reward[i] = np.mean(rewards)
        av_a_t[i] = np.mean(log_a_t)

    # return cache
    print("av_reward: ", np.mean(av_reward))
    print("av_a_t: ", np.mean(av_a_t))
    log_dist_a = np.array(log_dist_a)
    log_targ_a = np.array(log_targ_a)
    results = [log_dist_a, log_targ_a, log_cache, log_cond]
    metrics = [log_loss_sup, log_loss_actor, log_loss_critic,
               log_return, log_pi_ent]
    out = [results, metrics]
    print("log_dist_a shape: ", np.shape(log_dist_a))
    if get_data:
        #X_array_list = log_X
        #sim_lenths = log_sim_lengths
        av_sims_data = np.mean(log_sim_lengths)
        all_sims_data = log_sim_lengths
        out.append(av_sims_data)
        out.append(all_sims_data)
        out.append(np.mean(av_reward))
    return out


def append_prev_info(x_it_, scalar_list):
    x_it_ = x_it_.type(torch.FloatTensor)
    for s in scalar_list:
        s = s.type(torch.FloatTensor)
        #print("stype: ", s.type())
        #print("x_it_ type: ", x_it_.type())
        x_it_ = torch.cat(
            [x_it_, s.view(tensor_length(s))]
        )
        x_it_.type(torch.FloatTensor)
    return x_it_


def tensor_length(tensor):
    if tensor.dim() == 0:
        length = 1
    elif tensor.dim() > 1:
        raise ValueError('length for high dim tensor is undefined')
    else:
        length = len(tensor)
    return length


def get_enc_times(enc_size, n_param, pad_len):
    n_segments = n_param // enc_size
    enc_times_ = [enc_size * (k + 1) for k in range(n_segments)]
    enc_times = [pad_len + et - 1 for et in enc_times_]
    return enc_times


def pick_condition(p, rm_only=True, fix_cond=None):
    all_tz_conditions = list(TZ_COND_DICT.values())
    if fix_cond is not None:
        return fix_cond
    else:
        if rm_only:
            tz_cond = 'RM'
        else:
            tz_cond = np.random.choice(all_tz_conditions, p=P_TZ_CONDS)
        return tz_cond


def set_encoding_flag(t, enc_times, cond, agent):
    if t in enc_times and cond != 'NM':
        agent.encoding_on()
    else:
        agent.encoding_off()


def slience_recall(
        t_relative, in_2nd_part, slience_recall_time,
        agent
):
    if in_2nd_part:
        if t_relative in slience_recall_time:
            agent.retrieval_off()
        else:
            agent.retrieval_on()


def cond_manipulation(tz_cond, t, event_bond, hc_t, agent, n_lures=1):
    '''condition specific manipulation
    such as flushing, insert lure, etc.
    '''
    if t == event_bond:
        agent.retrieval_on()
        # flush WM unless RM
        if tz_cond != 'RM':
            hc_t = agent.get_init_states()
    return hc_t


def sample_penalty(p, fix_penalty, get_mean=False):
    if get_mean:
        penalty_val = p.env.penalty / 2
    else:
        # if penalty level is fixed, usually used during test
        if fix_penalty is not None:
            penalty_val = fix_penalty
        else:
            # otherwise sample a penalty level
            if p.env.penalty_random:
                if p.env.penalty_discrete:
                    penalty_val = np.random.choice(p.env.penalty_range)
                else:
                    penalty_val = np.random.uniform(0, p.env.penalty)
            else:
                # or train with a fixed penalty level
                penalty_val = p.env.penalty
    # form the input representation of the current penalty signal
    if p.env.penalty_onehot:
        penalty_rep = one_hot_penalty(penalty_val, p)
    else:
        penalty_rep = penalty_val
    return torch.tensor(penalty_val), torch.tensor(penalty_rep)


def one_hot_penalty(penalty_int, p):
    assert penalty_int in p.env.penalty_range, \
        print(f'invalid penalty_int = {penalty_int}')
    one_hot_dim = len(p.env.penalty_range)
    penalty_id = p.env.penalty_range.index(penalty_int)
    return np.eye(one_hot_dim)[penalty_id, :]


def time_scramble(X_i, Y_i, task, scramble_obs_only=True):
    if scramble_obs_only:
        # option 1: scramble observations
        X_i[:, :task.k_dim + task.v_dim] = scramble_array(
            X_i[:, :task.k_dim + task.v_dim])
    else:
        # option 2: scramble observations + queries
        [X_i, Y_i] = scramble_array_list([X_i, Y_i])
    return X_i, Y_i


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

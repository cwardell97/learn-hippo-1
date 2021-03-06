import torch
import numpy as np
import torch.nn.functional as F
import pdb
import random as rd
from scipy.stats import sem

from torch.autograd import Variable
from analysis import entropy
from utils.utils import to_sqnp
from utils.constants import TZ_COND_DICT, P_TZ_CONDS
from task.utils import scramble_array, scramble_array_list
from models import compute_returns, compute_a2c_loss, get_reward_ms
from utils.io import pickle_load_dict, unpack_inps_t
#from torchsummary import summary

''' check that supervised and cond_i have all been removed '''

def run_ms(
        agent, optimizer, task, p, n_examples, fpath,
        fix_penalty=None, slience_recall_time=None,
        learning=True, get_cache=True, get_data=True,
        counter_fact=True, em=True, seed_num=2,
        mem_num=2, cond='DM'
):

    # load training data
    training_data = pickle_load_dict(fpath).pop('XY')
    X_train = np.array(training_data[0])
    Y_train = np.array(training_data[1])
    print("X_train.shape", np.shape(X_train))
    # get time info
    T_total = np.shape(X_train)[1]
    T_part, pad_len, event_ends, event_bonds = task.get_time_param(T_total)
    enc_times = get_enc_times(p.net.enc_size, task.n_param, pad_len)

    # logger
    log_return, log_pi_ent = 0, 0
    log_loss_sup, log_loss_actor, log_loss_critic = 0, 0, 0
    log_cond = np.zeros(n_examples,)
    log_dist_a = [[] for _ in range(n_examples)]
    log_targ_a = [[] for _ in range(n_examples)]
    log_cache = np.zeros((n_examples,T_part))
    log_cache_sem = np.zeros(T_part)
    log_X = []
    # sim lengths
    log_sim_lengths = np.zeros(n_examples)
    # reward stuff
    av_reward = np.zeros(n_examples)
    # sim origins
    mem1_matches_ratio = np.zeros(n_examples)
    mem2_matches_ratio = np.zeros(n_examples)
    no_matches_ratio = np.zeros(n_examples)
    step_num_ratio = np.ones(n_examples)
    both_match_ratio = np.zeros(n_examples)

    # note that first and second half of x is redudent, only need to show half
    for i in range(n_examples):
        #print(i, "in ", n_examples)
        # init logs
        log_a_t = []
        #ep_rewards = []
        mem1_matches = []
        mem2_matches = []
        no_matches = []
        step_num = []
        both_matches = []


        # pick a condition
        cond_i = cond

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


        # prealloc
        loss_sup = 0
        probs, rewards, values, ents = [], [], [], []
        T_part = 16
        log_cache_i = np.empty(T_part)
        log_cache_i[:] = np.NaN


        # set penalty val
        penalty_val_p1, penalty_rep_p1 = [torch.tensor(fix_penalty),
                                          torch.tensor(fix_penalty)]
        # init model wm and em
        hc_t = agent.get_init_states()
        agent.retrieval_on()
        agent.encoding_off()

        '''sample simulation seeds'''
        seed_dictX, seed_dictY = get_seed_dicts(T_part,seed_num,
                                                X_dict, Y_dict, counter_fact)

        # save length of output for a timestep
        out_leng = np.add(p.env.n_param,
        seed_dictY["seed_Y{0}".format(0)].shape[0])


        ''' Load em with 'mem_num' events'''
        for mn in range(mem_num):
            #print("mem_num: ", mn)
            # load X,Y for specific events
            X_mn = X_dict["X_{0}".format(mn)]
            Y_mn = Y_dict["Y_{0}".format(mn)]
            #print("X_mn dims: ", np.shape(X_mn))
            for t in range(T_part):
                t_relative = t % T_part
                #in_2nd_part = t >= T_part REMOVE
                set_encoding_ms(t, enc_times, cond, em, agent)
                #if not in_2nd_part: REMOVE
                penalty_val, penalty_rep = penalty_val_p1, penalty_rep_p1

                #else: REMOVE
                #    penalty_val, penalty_rep = penalty_val_p2, penalty_rep_p2

                # testing condition REMOVE
                #if slience_recall_time is not None:
                #    slience_recall(t_relative, in_2nd_part,
                #                   slience_recall_time, agent)

                torch_X_mn = torch.from_numpy(X_mn[t])
                # forward
                x_it = append_prev_info(torch_X_mn, [penalty_rep])
                pi_a_t, v_t, hc_t, cache_t = agent.forward(
                    x_it.view(1, 1, -1), hc_t)
                # after delay period, compute loss
                a_t, p_a_t = agent.pick_action(pi_a_t)
                # get reward
                ''' REMOVE
                r_t = get_reward(a_t, Y_mn[t], penalty_val)
                # cache the results for later RL loss computation REMOVE
                rewards.append(r_t)
                values.append(v_t)
                probs.append(p_a_t)
                ents.append(entropy(pi_a_t))
                ep_rewards.append(r_t)
                '''

        # update WM/EM bsaed on the condition
        # if distant memory, flush after observations:
        if cond == 'DM':
            hc_t = agent.get_init_states()

        # turn off encoding for MS
        agent.encoding_off()
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

                # testing condition REMOVE
                #if slience_recall_time is not None:
                #    slience_recall(t_relative, in_2nd_part,
                #                   slience_recall_time, agent)
                # forward (CHANGE: might need torch conversion)
                torch_x_i_t = torch.from_numpy(seed_dictX["seed_X{0}".format(t)])
                x_it = append_prev_info(torch_x_i_t.type(torch.FloatTensor),
                [penalty_rep]
                )

                pi_a_t, v_t, hc_t, cache_t = agent.forward(
                    x_it.view(1, 1, -1), hc_t)
                # after delay period, compute loss
                a_t, p_a_t = agent.pick_action(pi_a_t)
                # cache, important for input gate
                if get_cache:
                    log_cache_i[t] = unpack_inps_t(cache_t)

            elif (seed_num-1)==t:
                # add in case for t=k, for first seed, but also sims_data

                # forward (CHANGE: might need torch conversion)
                torch_x_i_t = torch.from_numpy(seed_dictX["seed_X{0}".format(t)])
                x_it = append_prev_info(torch_x_i_t.type(torch.FloatTensor),
                [penalty_rep]
                )

                pi_a_t, v_t, hc_t, cache_t = agent.forward(
                    x_it.view(1, 1, -1), hc_t)

                # cache, important for input gate
                if get_cache:
                    log_cache_i[t] = unpack_inps_t(cache_t)

                # after delay period, compute loss
                a_t, p_a_t = agent.pick_action(pi_a_t)
                # get reward
                r_t = get_reward_ms(a_t, seed_dictY["seed_Y{0}".format(0)],
                penalty_val
                )

                ''' value prints
                print("r_t: ", r_t.item())
                print("t at stage 2: ", t)
                print("a_t: ", a_t.item())
                print("p_a_t: ", p_a_t.item())'''

                # cache the results for later RL loss computation REMOVE
                rewards.append(r_t)
                probs.append(p_a_t)
                values.append(v_t)
                ents.append(entropy(pi_a_t))

                log_a_t.append(a_t)

                # convert model prediction to input for next timesteps
                X_i_t = io_convert(a_t, t, p.env.n_param,
                seed_dictY["seed_Y{0}".format(0)].shape[0]
                )
                # if don't know, break
                if seed_dictY["seed_Y{0}".format(0)].shape[0] == a_t:
                    break

                else:
                    # save memories
                    memory1 = X_dict["X_{0}".format(0)]
                    memory2 = X_dict["X_{0}".format(1)]

                    # compute origin of model output
                    m1_match, m2_match, n_match, b_match = compare_output(X_i_t,
                    memory1, memory2, out_leng)
                    # save results
                    mem1_matches.append(m1_match)
                    mem2_matches.append(m2_match)
                    no_matches.append(n_match)
                    both_matches.append(b_match)
                    step_num.append(1)


            else:
                # now just predict for rest, once k<t
                #start prediction task immediately after the last seed (@t=k)
                penalty_val, penalty_rep = penalty_val_p1, penalty_rep_p1

                # testing condition REMOVE
                #if slience_recall_time is not None:
                #    slience_recall(t_relative, in_2nd_part,
                #                   slience_recall_time, agent)
                # whether to encode

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
                # cache, important for input gate
                if get_cache:
                    log_cache_i[t] = unpack_inps_t(cache_t)

                # convert model output to onehotinput for t+1 (action, time, total timesteps, total vals)
                X_i_t = io_convert(a_t, t, p.env.n_param,
                seed_dictY["seed_Y{0}".format(0)].shape[0]
                )

                # cache the results for later RL loss computation
                rewards.append(r_t)
                values.append(v_t)
                probs.append(p_a_t)
                ents.append(entropy(pi_a_t))
                log_a_t.append(a_t)

                # for behavioral stuff, only record prediction time steps
                '''if t % T_part >= pad_len:
                    log_dist_a[i].append(to_sqnp(pi_a_t))
                    log_targ_a[i].append(to_sqnp(torch.from_numpy(Y_i[t])))'''

                # if don't know, break, except if its the first two!
                if seed_dictY["seed_Y{0}".format(0)].shape[0] == a_t:
                    break
                # if not don't know, save origin of output
                else:
                    # save memories
                    memory1 = X_dict["X_{0}".format(0)]
                    memory2 = X_dict["X_{0}".format(1)]

                    # compute origin of model output
                    m1_match, m2_match, n_match, b_match = compare_output(X_i_t,
                    memory1, memory2, out_leng)
                    # save results
                    mem1_matches.append(m1_match)
                    mem2_matches.append(m2_match)
                    no_matches.append(n_match)
                    step_num.append(1)
                    both_matches.append(b_match)


        # log sim length after t loop
        log_sim_lengths[i] = t

        # compute RL loss (just merge these together from two tasks)
        returns = compute_returns(rewards, normalize=p.env.normalize_return)
        loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
        pi_ent = torch.stack(ents).sum()
        print('rewards: ', rewards)
        print("returns: ", returns)
        if learning:
            loss = loss_actor + loss_critic - pi_ent * p.net.eta
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
        #log_cond[i] = TZ_COND_DICT.inverse[cond_i]
        if get_cache:
            log_cache[i,:] = log_cache_i
            #log_cache_sem[i] = sem(log_cache_i)

        # cache averages across example
        av_reward[i] = np.sum(rewards)
        print("av_reward:", av_reward[i])
        print("t: ", t)
        print(i)

        #if t==1 and a_t==4:pdb.set_trace()
        # output origins
        if np.sum(np.sum(step_num))!=0:
            step_num_ratio[i] = np.sum(np.sum(step_num))
        mem1_matches_ratio[i] = np.divide(np.sum(np.sum(mem1_matches)),
                                          step_num_ratio[i])
        mem2_matches_ratio[i] = np.divide(np.sum(np.sum(mem2_matches)),
                                          step_num_ratio[i])
        no_matches_ratio[i] = np.divide(np.sum(np.sum(no_matches)),
                                        step_num_ratio[i])
        both_match_ratio[i] = np.divide(np.sum(np.sum(both_matches)),
                                        step_num_ratio[i])
        ''' match prints
        print("step_num:", step_num_ratio[i])
        print("No matches:", no_matches_ratio[i])
        print("mem1_matches:", mem1_matches_ratio[i])
        print("mem2_matches:", mem2_matches_ratio[i])
        print("both_matches:", both_match_ratio[i])
        print("does it sum:",
        np.sum((no_matches_ratio[i],mem1_matches_ratio[i],mem2_matches_ratio[i])))
        '''
    # pre-proces cache
    out_log_cache = np.nanmean(log_cache, axis=0)
    log_cache_sem = sem(log_cache, axis=0)
    print("log_cache_sem dims", np.shape(log_cache_sem))
    # return cache
    log_dist_a = np.array(log_dist_a)
    log_targ_a = np.array(log_targ_a)
    results = [log_dist_a, log_targ_a, out_log_cache, log_cond, log_cache_sem]
    metrics = [log_loss_sup, log_loss_actor, log_loss_critic,
               log_return, log_pi_ent]
    out = [results, metrics]

    #pdb.set_trace()
    # add in simulation length data
    av_sims_data = np.mean(log_sim_lengths)
    all_sims_data = log_sim_lengths
    sims_data = [av_sims_data,all_sims_data]
    out.append(sims_data)

    # add reward data
    reward_data = [np.mean(av_reward)]
    out.append(reward_data)

    # add in sim origin data
    sim_origins = [np.mean(mem1_matches_ratio),
                   np.mean(mem2_matches_ratio),
                   np.mean(no_matches_ratio),
                   np.mean(both_match_ratio)]
    out.append(sim_origins)

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


def compare_output(X_i_t, memory1, memory2, out_leng):
    '''compare output to events in em
    returns 1 for mem1, 2 for mem 2, 0 for no mem
    '''
    # init counters
    mem1_matches = []
    mem2_matches = []
    no_matches = []
    both_match = []
    rag = np.int((np.shape(memory1)[0])/2)
    # trim model output
    output = X_i_t[:out_leng,]

    no_match = 0
    # loop through all time steps of each row
    for row in range(rag):
        # pull 1 row for each mem
        memstep1 = memory1[row,:]
        memstep2 = memory2[row,:]
        memstep_short_1 = memstep1[:out_leng,]
        memstep_short_2 = memstep2[:out_leng,]

        if np.all(memstep_short_1 == output):
            mem1_matches.append(1)
            no_match = 1

        if np.all(memstep_short_2 == output):
            mem2_matches.append(1)
            no_match = 1

        if np.all(memstep_short_2 == output) and np.all(memstep_short_1 == output):
            both_match.append(1)

    if no_match == 0:
        no_matches.append(1)

    out = [mem1_matches, mem2_matches, no_matches, both_match]
    return out

def get_seed_dicts(T_part, seed_num, X_dict, Y_dict, counter_fact):
    "get seeds for mental simulation"

    # init seed dicts
    seed_dictX = {}
    seed_dictY = {}
    pair_nums = np.random.choice(T_part,seed_num,replace=False)

    if not counter_fact:
        # rand X,Y from X_dict/Y_dict for seeds
        j = np.random.choice(len(X_dict))

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

    return seed_dictX, seed_dictY


def set_encoding_ms(t, enc_times, cond, em, agent):
    if t in enc_times and cond != 'NM' and em:
        agent.encoding_on()
        print("encoding at: ", t)
    else:
        agent.encoding_off()

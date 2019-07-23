import torch
import numpy as np
import torch.nn.functional as F

from analysis import entropy
from utils.utils import to_sqnp
from utils.constants import TZ_COND_DICT, P_TZ_CONDS
from models import get_reward, compute_returns, compute_a2c_loss


def run_tz(
        agent, optimizer, task, p, n_examples, supervised,
        cond=None, learning=True, get_cache=True, get_data=False,
        slience_recall_time=None,
):
    # sample data
    X, Y = task.sample(n_examples, to_torch=True)
    # misc
    a_0, r_0 = torch.tensor(p.dk_id), torch.tensor(0)
    # logger
    log_return, log_pi_ent = 0, 0
    log_loss_sup, log_loss_actor, log_loss_critic = 0, 0, 0
    log_cond = np.zeros(n_examples,)
    log_dist_a = [[] for _ in range(n_examples)]
    log_targ_a = [[] for _ in range(n_examples)]
    log_cache = [None] * n_examples

    for i in range(n_examples):
        # pick a condition
        cond_i = pick_condition(p, rm_only=supervised, fix_cond=cond)
        # get the example for this trial
        X_i, Y_i = X[i], Y[i]
        # get time info
        T_total = np.shape(X_i)[0]
        T_part, pad_len, event_ends, event_bond = task.get_time_param(T_total)
        enc_times = get_enc_times(p.net.enc_size, task.n_param, pad_len)

        # prealloc
        loss_sup = 0
        probs, rewards, values, ents = [], [], [], []
        log_cache_i = [None] * T_total

        # init model wm and em
        penalty = sample_penalty(p)
        hc_t = agent.get_init_states()
        agent.retrieval_off()
        agent.encoding_off()
        a_t, r_t = a_0, r_0

        for t in range(T_total):
            t_relative = t % T_part
            in_2nd_part = t >= T_part
            # testing condition
            if slience_recall_time is not None:
                slience_recall(t_relative, in_2nd_part,
                               slience_recall_time, agent)
            # whether to encode
            if not supervised:
                set_encoding_flag(t, enc_times, cond_i, agent)

            # forward
            x_it = append_prev_info(X_i[t], [a_t, r_t, penalty])
            # x_it = X_i[t]
            pi_a_t, v_t, hc_t, cache_t = agent.forward(
                x_it.view(1, 1, -1), hc_t)
            # after delay period, compute loss
            a_t, p_a_t = agent.pick_action(pi_a_t)
            # get reward
            r_t = get_reward(a_t, Y_i[t], penalty)

            # cache the results for later RL loss computation
            rewards.append(r_t)
            values.append(v_t)
            probs.append(p_a_t)
            ents.append(entropy(pi_a_t))
            # compute supervised loss
            yhat_t = torch.squeeze(pi_a_t)[:-1]
            loss_sup += F.mse_loss(yhat_t, Y_i[t])

            if not supervised:
                # update WM/EM bsaed on the condition
                hc_t = cond_manipulation(
                    cond_i, t, event_ends[0], hc_t, agent)

            # cache results for later analysis
            if get_cache:
                log_cache_i[t] = cache_t
            # for behavioral stuff, only record prediction time steps
            if t % T_part >= pad_len:
                log_dist_a[i].append(to_sqnp(pi_a_t))
                log_targ_a[i].append(to_sqnp(Y_i[t]))

        # compute RL loss
        returns = compute_returns(rewards, normalize=True)
        loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
        pi_ent = torch.stack(ents).sum()
        # if learning and not supervised
        if learning:
            if supervised:
                loss = loss_sup
            else:
                loss = loss_actor + loss_critic - pi_ent * p.net.eta
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1)
            optimizer.step()

        # after every event sequence, log stuff
        log_loss_sup += loss_sup / n_examples
        log_pi_ent += pi_ent.item() / n_examples
        log_return += torch.stack(rewards).sum().item()/n_examples
        log_loss_actor += loss_actor.item()/n_examples
        log_loss_critic += loss_critic.item()/n_examples
        log_cond[i] = TZ_COND_DICT.inverse[cond_i]
        if get_cache:
            log_cache[i] = log_cache_i

    # return cache
    log_dist_a = np.array(log_dist_a)
    log_targ_a = np.array(log_targ_a)
    results = [log_dist_a, log_targ_a, log_cache, log_cond]
    metrics = [log_loss_sup, log_loss_actor, log_loss_critic,
               log_return, log_pi_ent]
    out = [results, metrics]
    if get_data:
        X_array_list = [to_sqnp(X[i]) for i in range(n_examples)]
        Y_array_list = [to_sqnp(Y[i]) for i in range(n_examples)]
        training_data = [X_array_list, Y_array_list]
        out.append(training_data)
    return out


def append_prev_info(x_it_, scalar_list):
    for s in scalar_list:
        x_it_ = torch.cat([x_it_, s.type(torch.FloatTensor).view(1)])
    # a_prev = a_prev.type(torch.FloatTensor).view(1)
    # r_prev = r_prev.type(torch.FloatTensor).view(1)
    # x_it = torch.cat([x_it_, a_prev, r_prev])
    return x_it_


def get_enc_times(enc_size, n_param, pad_len):
    n_segments = n_param // enc_size
    enc_times_ = [enc_size * (k+1) for k in range(n_segments)]
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


def slience_recall(t_relative, in_2nd_part, slience_recall_time, agent):
    if in_2nd_part:
        if t_relative == slience_recall_time:
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


def sample_penalty(p):
    penalty = p.env.penalty
    if p.env.penalty_random:
        penalty = np.random.uniform(0, p.env.penalty)
    return torch.tensor(penalty)


# def cond_manipulation(tz_cond, t, event_bond, hc_t, agent, n_lures=1):
#     '''condition specific manipulation
#     such as flushing, insert lure, etc.
#     '''
#     if t == event_bond:
#         agent.retrieval_on()
#         if tz_cond == 'DM':
#             # RM: has EM, no WM
#             hc_t = agent.get_init_states()
#             agent.add_simple_lures(n_lures)
#         elif tz_cond == 'NM':
#             # RM: no WM, EM
#             hc_t = agent.get_init_states()
#             agent.flush_episodic_memory()
#             agent.add_simple_lures(n_lures+1)
#         elif tz_cond == 'RM':
#             # RM: has WM, EM
#             agent.add_simple_lures(n_lures)
#         else:
#             raise ValueError('unrecog tz condition')
#     return hc_t

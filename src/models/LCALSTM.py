import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from models.EM import EM
from torch.distributions import Categorical
from models.initializer import initialize_weights

# constants
# number of vector signal (lstm gates)
N_VSIG = 3
# number of scalar signal (sigma)
N_SSIG = 2
# the ordering in the cache
scalar_signal_names = ['input strength', 'competition']
vector_signal_names = ['f', 'i', 'o']

sigmoid = nn.Sigmoid()
gain = 1


class LCALSTM(nn.Module):

    def __init__(
            self,
            input_dim, output_dim,
            rnn_hidden_dim, dec_hidden_dim,
            kernel='cosine', dict_len=100,
            weight_init_scheme='ortho',
            init_state_trainable=False,
            noisy_encoding=0, cmpt=.8
    ):
        super(LCALSTM, self).__init__()
        self.cmpt = cmpt
        self.input_dim = input_dim + 1
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_hidden_total = (N_VSIG + 1) * rnn_hidden_dim + N_SSIG
        # rnn module (input to hidden is h to x)
        self.i2h = nn.Linear(self.input_dim, self.n_hidden_total)
        self.h2h = nn.Linear(rnn_hidden_dim, self.n_hidden_total)
        # deicion module
        self.ih = nn.Linear(rnn_hidden_dim, dec_hidden_dim) #input to hidden
        self.actor = nn.Linear(dec_hidden_dim, output_dim) #
        self.critic = nn.Linear(dec_hidden_dim, 1)
        # memory
        self.hpc = nn.Linear(rnn_hidden_dim + dec_hidden_dim, N_SSIG) #this is the retrieval control layer
        self.em = EM(dict_len, rnn_hidden_dim, kernel) # this has no network layer
        # the RL mechanism
        self.weight_init_scheme = weight_init_scheme
        self.init_state_trainable = init_state_trainable
        if noisy_encoding == 0:
            self.noisy_encoding = False
        elif noisy_encoding == 1:
            self.noisy_encoding = True
        else:
            raise ValueError('noisy_encoding arg must be 0 or 1')
        self.init_model()

    def init_model(self):
        # add name fields
        self.n_ssig = N_SSIG
        self.n_vsig = N_VSIG
        self.vsig_names = vector_signal_names
        self.ssig_names = scalar_signal_names
        # init params
        initialize_weights(self, self.weight_init_scheme)
        if self.init_state_trainable:
            self.init_init_states()

    def init_init_states(self):
        scale = 1 / self.rnn_hidden_dim
        self.h_0 = torch.nn.Parameter(
            sample_random_vector(self.rnn_hidden_dim, scale), requires_grad=True
        )
        self.c_0 = torch.nn.Parameter(
            sample_random_vector(self.rnn_hidden_dim, scale), requires_grad=True
        )

    def get_init_states(self, scale=.1, device='cpu'):
        if self.init_state_trainable:
            h_0_, c_0_ = self.h_0, self.c_0
        else:
            h_0_ = sample_random_vector(self.rnn_hidden_dim, scale)
            c_0_ = sample_random_vector(self.rnn_hidden_dim, scale)
        return (h_0_, c_0_)

    def forward(self, x_t, hc_prev, beta=1):
        # unpack activity
        (h_prev, c_prev) = hc_prev
        h_prev = h_prev.view(h_prev.size(1), -1)
        c_prev = c_prev.view(c_prev.size(1), -1) # cell state from t-1
        x_t = x_t.view(x_t.size(1), -1)
        # transform the input info
        preact = self.i2h(x_t) + self.h2h(h_prev) #preactivation because it hasnt been transform by nonlin
        # get all gate values
        gates = preact[:, : N_VSIG * self.rnn_hidden_dim].sigmoid()
        c_t_new = preact[:, N_VSIG * self.rnn_hidden_dim + N_SSIG:].tanh()
        # split input(write) gate, forget gate, output(read) gate
        f_t = gates[:, :self.rnn_hidden_dim]
        o_t = gates[:, self.rnn_hidden_dim:2 * self.rnn_hidden_dim]
        i_t = gates[:, -self.rnn_hidden_dim:]
        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = torch.mul(c_prev, f_t) + torch.mul(i_t, c_t_new)
        # make 1st decision attempt
        h_t = torch.mul(o_t, c_t.tanh())
        dec_act_t = F.relu(self.ih(h_t))
        # recall / encode
        hpc_input_t = torch.cat([c_t, dec_act_t], dim=1)
        phi_t = sigmoid(self.hpc(hpc_input_t))
        [inps_t, comp_t] = torch.squeeze(phi_t)
        m_t = self.recall(c_t, inps_t)
        cm_t = c_t + m_t
        self.encode(cm_t)
        '''final decision attempt'''
        # make final dec
        h_t = torch.mul(o_t, cm_t.tanh())
        dec_act_t = F.relu(self.ih(h_t))
        pi_a_t = _softmax(self.actor(dec_act_t), beta)
        value_t = self.critic(dec_act_t)
        # reshape data
        h_t = h_t.view(1, h_t.size(0), -1)
        cm_t = cm_t.view(1, cm_t.size(0), -1)
        # scache results
        scalar_signal = [inps_t, 0, comp_t]
        vector_signal = [f_t, i_t, o_t]
        misc = [h_t, m_t, cm_t, dec_act_t, self.em.get_vals()]
        cache = [vector_signal, scalar_signal, misc]
        return pi_a_t, value_t, (h_t, cm_t), cache

    def recall(self, c_t, inps_t, comp_t=None):
        """run the "pattern completion" procedure

        Parameters
        ----------
        c_t : torch.tensor, vector
            cell state
        leak_t : torch.tensor, scalar
            LCA param, leak
        comp_t : torch.tensor, scalar
            LCA param, lateral inhibition
        inps_t : torch.tensor, scalar
            LCA param, input strength / feedforward weights

        Returns
        -------
        tensor, tensor
            updated cell state, recalled item

        """
        if comp_t is None:
            comp_t = self.cmpt

        if self.em.retrieval_off:
            m_t = torch.zeros_like(c_t)
        else:
            # retrieve memory
            m_t = self.em.get_memory(
                c_t, leak=0, comp=comp_t, w_input=inps_t
            )
        return m_t

    def encode(self, cm_t):
        if not self.em.encoding_off:
            if self.noisy_encoding:
                # a two memory case, a bit artificial
                # can generalize to n-memory case with random noise
                noise = sample_random_vector(self.rnn_hidden_dim, scale=1)
                self.em.save_memory(cm_t + noise)
                self.em.save_memory(cm_t - noise)
            else:
                self.em.save_memory(cm_t)

    def pick_action(self, action_distribution):
        """action selection by sampling from a multinomial.

        Parameters
        ----------
        action_distribution : 1d torch.tensor
            action distribution, pi(a|s)

        Returns
        -------
        torch.tensor(int), torch.tensor(float)
            sampled action, log_prob(sampled action)

        """
        m = Categorical(action_distribution)
        a_t = m.sample()
        log_prob_a_t = m.log_prob(a_t)
        return a_t, log_prob_a_t

    def add_simple_lures(self, n_lures=1):
        lures = [sample_random_vector(self.rnn_hidden_dim)
                 for _ in range(n_lures)]
        self.em.inject_memories(lures)

    def init_em_config(self):
        self.flush_episodic_memory()
        self.encoding_off()
        self.retrieval_off()

    def flush_episodic_memory(self):
        self.em.flush()

    def encoding_off(self):
        self.em.encoding_off = True

    def retrieval_off(self):
        self.em.retrieval_off = True

    def encoding_on(self):
        self.em.encoding_off = False

    def retrieval_on(self):
        self.em.retrieval_off = False


def sample_random_vector(n_dim, scale=.1):
    return torch.randn(1, 1, n_dim) * scale


def _softmax(z, beta):
    """helper function, softmax with beta

    Parameters
    ----------
    z : torch tensor, has 1d underlying structure after torch.squeeze
        the raw logits
    beta : float, >0
        softmax temp, big value -> more "randomness"

    Returns
    -------
    1d torch tensor
        a probability distribution | beta

    """
    assert beta > 0
    # softmax the input to a valid PMF
    pi_a = F.softmax(torch.squeeze(z / beta), dim=0)
    # make sure the output is valid
    if torch.any(torch.isnan(pi_a)):
        raise ValueError(f'Softmax produced nan: {z} -> {pi_a}')
    return pi_a

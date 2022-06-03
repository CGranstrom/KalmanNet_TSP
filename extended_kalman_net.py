"""# **Class: KalmanNet**"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as func

from path_models import path_model

sys.path.insert(1, path_model)
from model import getJacobian

nGRU = 2


class KalmanNetNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # initialize Kalman gain network
        # applies to both build() and _init_k_gain_net()?

    def build(self, ss_model: "SystemModel", info_string="fullInfo"):

        self._init_system_dynamics(
            ss_model.f, ss_model.h, ss_model.m, ss_model.n, info_string="fullInfo"
        )
        self.init_sequence(ss_model.m1x_0, ss_model.t)

        # Number of neurons in the 1st hidden layer
        H1_KNet = (ss_model.m + ss_model.n) * 10 * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ss_model.m * ss_model.n) * 1 * 4

        self._init_k_gain_net(H1_KNet, H2_KNet)

    def _init_k_gain_net(self, H1, H2):
        """Initialize the Kalman gain network."""

        # input dimensions (+1 for time input)
        input_dims = self.m + self.m + self.n  # F1,3,4
        output_dims = self.m * self.n

        # Kalman Gain

        # input layer

        # linear Layer
        self.KG_l1 = nn.Linear(input_dims, H1, bias=True)

        # ReLU (Rectified linear Unit) Activation Function
        self.KG_relu1 = nn.ReLU()

        # GRU

        # input dimension
        self.input_dim = H1

        # hidden dimension
        self.hidden_dim = ((self.n * self.n) + (self.m * self.m)) * 10 * 1

        # number of layers
        self.n_layers = nGRU

        # batch size
        self.batch_size = 1

        # input sequence length
        self.seq_len_input = 1

        # hidden sequence length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # initialize a tensor for GRU input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # initialize a tensor for hidden state
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim)

        # initialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

        # hidden layer
        self.KG_l2 = nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified linear Unit) Activation Function
        self.KG_relu2 = nn.ReLU()

        # output layer
        self.KG_l3 = nn.Linear(H2, output_dims, bias=True)

    def _init_system_dynamics(self, f, h, m, n, info_string="fullInfo"):
        """Initialize system dynamics."""

        if info_string == "partialInfo":
            self.fString = "ModInacc"
            self.hString = "ObsInacc"
        else:
            self.fString = "ModAcc"
            self.hString = "ObsAcc"

        # set state evolution matrix
        self.f = f
        self.m = m

        # set observation matrix
        self.h = h
        self.n = n

    def init_sequence(self, M1_0, T):
        """Initialize sequence."""

        self.m1x_posterior = torch.squeeze(M1_0)
        self.m1x_posterior_previous = 0  # for t=0

        self.T = T
        self.x_out = torch.empty(self.m, T)

        self.state_process_posterior_0 = torch.squeeze(M1_0)
        self.m1x_prior_previous = self.m1x_posterior

        # KGain saving
        self.i = 0
        self.KGain_array = self.KG_array = torch.zeros((self.T, self.m, self.n))

    def step_prior(self):  # rename to compute_priors()?
        """Compute priors."""

        # predict the 1st moment of x
        self.m1x_prior = torch.squeeze(self.f(self.m1x_posterior))

        # predict the 1st moment of y
        self.m1y = torch.squeeze(self.h(self.m1x_prior))

        # update Jacobians
        # self.JFt = get_Jacobian(self.m1x_posterior, self.fString)
        # self.JHt = get_Jacobian(self.m1x_prior, self.hString)

        self.state_process_prior_0 = torch.squeeze(
            self.f(self.state_process_posterior_0)
        )
        self.obs_process_0 = torch.squeeze(self.h(self.state_process_prior_0))

    def _step_k_gain_est(self, y):
        """Kalman gain estimation."""

        # feature 1: yt - yt-1
        try:
            my_f1_0 = y - torch.squeeze(self.y_previous)
        except:
            my_f1_0 = y - torch.squeeze(self.obs_process_0)  # when t=0

        # my_f1_reshape = torch.squeeze(my_f1_0)
        y_f1_norm = func.normalize(my_f1_0, p=2, dim=0, eps=1e-12, out=None)

        # feature 2: yt - y_t+1|t
        # my_f2_0 = y - torch.squeeze(self.m1y)
        # my_f2_reshape = torch.squeeze(my_f2_0)
        # y_f2_norm = func.normalize(my_f2_reshape, p=2, dim=0, eps=1e-12, out=None)

        # feature 3: x_t|t - x_t-1|t-1
        m1x_f3_0 = self.m1x_posterior - self.m1x_posterior_previous
        m1x_f3_reshape = torch.squeeze(m1x_f3_0)
        m1x_f3_norm = func.normalize(m1x_f3_reshape, p=2, dim=0, eps=1e-12, out=None)

        # reshape and normalize m1x posterior
        # m1x_post_0 = self.m1x_posterior - self.state_process_posterior_0 # Option 1

        # feature 4: x_t|t - x_t|t-1
        m1x_f4_0 = self.m1x_posterior - self.m1x_prior_previous
        # m1x_reshape = torch.squeeze(self.m1x_posterior) # Option 3
        m1x_f4_reshape = torch.squeeze(m1x_f4_0)
        m1x_f4_norm = func.normalize(m1x_f4_reshape, p=2, dim=0, eps=1e-12, out=None)

        # normalize y
        # my_0 = y - torch.squeeze(self.obs_process_0) # Option 1
        # my_0 = y - torch.squeeze(self.m1y) # Option 2
        # my_0 = y
        # y_norm = func.normalize(my_0, p=2, dim=0, eps=1e-12, out=None)
        # y_norm = func.normalize(y, p=2, dim=0, eps=1e-12, out=None);

        # input for counting
        count_norm = func.normalize(
            torch.tensor([self.i]).float(), dim=0, eps=1e-12, out=None
        )

        # kalman gain net input
        k_gain_net_in = torch.cat([y_f1_norm, m1x_f3_norm, m1x_f4_norm], dim=0)

        # kalman gain network atep
        KG = self._k_gain_step(k_gain_net_in)

        # reshape Kalman gain to a matrix
        self.k_gain = torch.reshape(KG, (self.m, self.n))

    def _k_net_step(self, y):
        """KalmanNet step."""

        # compute priors
        self.step_prior()

        # compute kalman gain
        self._step_k_gain_est(y)

        # save Kalman gain in array
        self.KGain_array[self.i] = self.k_gain
        self.i += 1

        # innovation
        # y_obs = torch.unsqueeze(y, 1)
        dy = y - self.m1y

        # compute the 1st posterior moment
        innov = torch.matmul(self.k_gain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + innov

        self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y

        return torch.squeeze(self.m1x_posterior)

    def _k_gain_step(self, k_gain_net_in):
        """Kalman gain step."""

        # input layer
        L1_out = self.KG_l1(k_gain_net_in)
        La1_out = self.KG_relu1(L1_out)

        # GRU
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)
        GRU_in[0, 0, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))

        # hidden layer
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)

        # output layer
        L3_out = self.KG_l3(La2_out)
        return L3_out

    def forward(self, y):
        yt = torch.squeeze(y)
        """
        for t in range(0, self.t):
            self.x_out[:, t] = self.KNet_step(y[:, t])
        """
        self.x_out = self._k_net_step(yt)

        return self.x_out

    def init_hidden(self):
        """Initialize hidden state."""

        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data

"""# **Class: KalmanNet**"""

import torch
from torch import nn
from torch.nn import functional as func

# import sys
# from path_models import path_model
# sys.path.insert(1, path_model)
from system_models import LinearSystemModel


class KalmanNet(nn.Module):
    def __init__(
        self,
        ss_model: LinearSystemModel = None,
        time_input_for_extended_net: bool = False,
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_gru = 1

        # do not execute below if instance is going to immediately overwrite self.ss_model via torch.load()
        if ss_model:
            # moved from build()
            self._init_system_dynamics(ss_model.f, ss_model.h, ss_model.m, ss_model.n)

            # number of neurons in the 1st hidden layer
            H1_KNet = (ss_model.m + ss_model.n) * 10 * 8

            # number of neurons in the 2nd hidden layer
            H2_KNet = (ss_model.m * ss_model.n) * 1 * 4

            self._init_k_gain_net(H1_KNet, H2_KNet, time_input_for_extended_net)
        else:
            self.f = None
            self.h = None
            self.m = None
            self.n = None
            self.KG_l1 = None
            self.KG_relu1 = None
            self.input_dim = None
            self.hidden_dim = None
            self.n_layers = self.num_gru
            self.batch_size = 1
            self.seq_len_input = 1
            self.seq_len_hidden = self.n_layers
            self.hn = None
            self.rnn_GRU = None
            self.KG_l2 = None
            self.KG_relu2 = None
            self.KG_l3 = None

        # initialize attributes that are defined in other methods

        # initialize attributes from init_sequence()
        self.m1x_prior = None
        self.m1x_posterior = None
        self.state_process_posterior_0 = None

        # initialize attributes from _step_prior()
        self.state_process_prior_0 = None
        self.obs_process_0 = None
        self.m1x_prev_prior = None
        self.m1y = None

        # initialize attributes from _step_prior()
        self.k_gain = None

    def _init_system_dynamics(self, f, h, m, n):
        """Initialize system dynamics."""

        # set state evolution matrix
        self.f = f.to(self.device, non_blocking=True)
        self.h = h.to(self.device, non_blocking=True)
        self.m = self.f.size()[0]
        self.n = self.h.size()[0]

    # def build(self, ss_model: "SystemModel", time_input_for_extended_net: bool = False):
    #
    #     self._init_system_dynamics(ss_model.f, ss_model.h, ss_model.m, ss_model.n)
    #
    #     # number of neurons in the 1st hidden layer
    #     H1_KNet = (ss_model.m + ss_model.n) * 10 * 8
    #
    #     # number of neurons in the 2nd hidden layer
    #     H2_KNet = (ss_model.m * ss_model.n) * 1 * 4
    #
    #     self._init_k_gain_net(H1_KNet, H2_KNet, time_input_for_extended_net)

    def _init_k_gain_net(self, H1, H2, time_input_for_extended_net):
        """Initialize the Kalman gain network."""

        if time_input_for_extended_net:
            # (+1 for time input)
            input_dims = self.m + self.m + self.n  # F1,3,4
        else:
            input_dims = self.m + self.n  # x(t-1), y(t)

        output_dims = self.m * self.n

        # Kalman gain
        # input layer

        # linear layer
        self.KG_l1 = nn.Linear(input_dims, H1, bias=True)

        # ReLU (Rectified linear Unit) Activation Function
        self.KG_relu1 = nn.ReLU()

        # GRU

        # input dimension
        self.input_dim = H1

        # hidden dimension
        self.hidden_dim = (self.m ** 2 + self.n ** 2) * 10

        # number of layers
        self.n_layers = self.num_gru

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
        if not isinstance(self, ExtendedKalmanNet):
            self.hn = torch.randn(
                self.seq_len_hidden, self.batch_size, self.hidden_dim
            ).to(self.device, non_blocking=True)
        else:
            self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim)

        # initialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

        # hidden layer
        self.KG_l2 = nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified linear Unit) Activation Function
        self.KG_relu2 = nn.ReLU()

        # output layer
        self.KG_l3 = nn.Linear(H2, output_dims, bias=True)

    def init_sequence(self, M1_0, t=None):
        """Initialize sequence."""

        self.m1x_prior = M1_0.to(self.device, non_blocking=True)
        self.m1x_posterior = M1_0.to(self.device, non_blocking=True)
        self.state_process_posterior_0 = M1_0.to(self.device, non_blocking=True)

    def _step_prior(self):  # rename to compute_priors()?
        """Compute priors."""

        # compute the 1st moment of x based on model knowledge and without process noise
        self.state_process_prior_0 = torch.matmul(
            self.f, self.state_process_posterior_0
        )

        # compute the 1st moment of y based on model knowledge and without noise
        self.obs_process_0 = torch.matmul(self.h, self.state_process_prior_0)

        # predict the 1st moment of x
        self.m1x_prev_prior = self.m1x_prior
        self.m1x_prior = torch.matmul(self.f, self.m1x_posterior)

        # predict the 1st moment of y
        self.m1y = torch.matmul(self.h, self.m1x_prior)

    def _step_k_gain_est(self, y):
        """Kalman gain estimation."""

        # reshape and normalize the difference in X prior
        # feature 4: x_t|t - x_t|t-1
        # dm1x = self.m1x_prior - self.state_process_prior_0
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # feature 2: yt - y_t+1|t
        dm1y = y - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)

        # kalman gain net input
        k_gain_net_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)

        # kalman gain network step
        k_gain = self._k_gain_step(k_gain_net_in)

        # reshape Kalman gain to a matrix
        self.k_gain = torch.reshape(k_gain, (self.m, self.n))

    def _k_net_step(self, y):
        """KalmanNet step."""

        # compute priors
        self._step_prior()

        # compute Kalman gain
        self._step_k_gain_est(y)

        # innovation
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs - self.m1y

        # compute the 1st posterior moment
        innov = torch.matmul(self.k_gain, dy)
        self.m1x_posterior = self.m1x_prior + innov

        return torch.squeeze(self.m1x_posterior)

    def _k_gain_step(self, k_gain_net_in):
        """Kalman gain step."""

        # input layer
        L1_out = self.KG_l1(k_gain_net_in)
        La1_out = self.KG_relu1(L1_out)

        # GRU
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(
            self.device, non_blocking=True
        )
        GRU_in[0, 0, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))

        # hidden layer
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)

        # output layer
        L3_out = self.KG_l3(La2_out)
        return L3_out

    def forward(self, yt):
        yt = yt.to(self.device, non_blocking=True)
        return self._k_net_step(yt)

    def init_hidden(self):
        """Initialize hidden state."""

        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data


class ExtendedKalmanNet(KalmanNet):
    def __init__(
        self,
        ss_model: LinearSystemModel = None,
        time_input_for_extended_net: bool = True,
    ):
        super().__init__(ss_model, time_input_for_extended_net)
        self.num_gru = 2

        # initialize attributes that are defined in other methods
        # initialize attributes from init_sequence()
        self.m1x_posterior_previous = None
        self.k_gain_array = None
        self.i = None
        self.t = None
        self.x_out = None
        self.m1x_prior_previous = None

        # moved from build(). Don't execute if instance is going to immediately overwrite self.ss_model via torch.load()
        if ss_model:
            # initialize Kalman gain network
            self.init_sequence(ss_model.m1x_0, ss_model.t)
        else:
            self.m1x_posterior = None
            self.m1x_posterior_previous = None
            self.t = None
            self.x_out = None
            self.state_process_posterior_0 = None
            self.m1x_prior_previous = None
            self.i = None
            self.k_gain_array = None

    # initialize Kalman gain network
    # applies to both build() and _init_k_gain_net()?

    # def build(self, ss_model: "SystemModel", time_input_for_extended_net: bool = True):
    #
    #     super().build(ss_model, time_input_for_extended_net)

    def _init_system_dynamics(self, f, h, m=None, n=None):
        """Initialize system dynamics."""

        # set state evolution matrix
        self.f, self.h, self.m, self.n = f, h, m, n

    def init_sequence(self, M1_0, t=None):
        """Initialize sequence."""

        self.m1x_posterior = torch.squeeze(M1_0)
        self.m1x_posterior_previous = 0  # for t=0

        self.t = t
        self.x_out = torch.empty(self.m, t)

        self.state_process_posterior_0 = torch.squeeze(M1_0)
        self.m1x_prior_previous = self.m1x_posterior

        # KGain saving
        self.i = 0
        self.k_gain_array = torch.zeros((self.t, self.m, self.n))

    def _step_prior(self):  # rename to compute_priors()?
        """Compute priors."""

        # predict the 1st moment of x
        self.m1x_prior = torch.squeeze(self.f(self.m1x_posterior))

        # predict the 1st moment of y
        self.m1y = torch.squeeze(self.h(self.m1x_prior))

        # update Jacobians
        # self.JFt = get_Jacobian(self.m1x_posterior, self.f_string)
        # self.JHt = get_Jacobian(self.m1x_prior, self.h_string)

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

        # kalman gain net input
        k_gain_net_in = torch.cat([y_f1_norm, m1x_f3_norm, m1x_f4_norm], dim=0)

        # kalman gain network step
        k_gain = self._k_gain_step(k_gain_net_in)

        # reshape Kalman gain to a matrix
        self.k_gain = torch.reshape(k_gain, (self.m, self.n))

    def _k_net_step(self, y):
        """KalmanNet step."""

        # compute priors
        self._step_prior()

        # compute kalman gain
        self._step_k_gain_est(y)

        # save Kalman gain in array
        self.k_gain_array[self.i] = self.k_gain
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

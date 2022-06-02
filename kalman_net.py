"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func


class KalmanNetNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def build(self, ss_model: "SystemModel"):

        self._init_system_dynamics(ss_model.F, ss_model.H)

        # number of neurons in the 1st hidden layer
        H1_KNet = (ss_model.m + ss_model.n) * 10 * 8

        # number of neurons in the 2nd hidden layer
        H2_KNet = (ss_model.m * ss_model.n) * 1 * 4

        self._init_k_gain_net(H1_KNet, H2_KNet)

    def _init_k_gain_net(self, H1, H2):
        """Initialize the Kalman gain network."""

        input_dims = self.m + self.n  # x(t-1), y(t)
        output_dims = self.m * self.n  # Kalman Gain

        # input layer

        # linear Layer
        self.KG_l1 = nn.Linear(input_dims, H1, bias=True)

        # ReLU (Rectified linear Unit) Activation Function
        self.KG_relu1 = nn.ReLU()

        # GRU

        # input dimension
        self.input_dim = H1

        # hidden dimension
        self.hidden_dim = (self.m * self.m + self.n * self.n) * 10

        # number of layers
        self.n_layers = 1

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
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(
            self.device, non_blocking=True
        )

        # initialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

        # hidden layer
        self.KG_l2 = nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified linear Unit) Activation Function
        self.KG_relu2 = nn.ReLU()

        # output layer
        self.KG_l3 = nn.Linear(H2, output_dims, bias=True)

    def _init_system_dynamics(self, F, H):
        """Initialize system dynamics."""

        # set state evolution matrix
        self.F = F.to(self.device, non_blocking=True)
        self.F_T = torch.transpose(F, 0, 1)
        self.m = self.F.size()[0]

        # set observation matrix
        self.H = H.to(self.device, non_blocking=True)
        self.H_T = torch.transpose(H, 0, 1)
        self.n = self.H.size()[0]

    def init_sequence(self, M1_0):
        """Initialize sequence."""

        self.m1x_prior = M1_0.to(self.device, non_blocking=True)

        self.m1x_posterior = M1_0.to(self.device, non_blocking=True)

        self.state_process_posterior_0 = M1_0.to(self.device, non_blocking=True)

    def _step_prior(self):  # rename to compute_priors()?
        """Compute priors."""

        # compute the 1st moment of x based on model knowledge and without process noise
        self.state_process_prior_0 = torch.matmul(
            self.F, self.state_process_posterior_0
        )

        # compute the 1st moment of y based on model knowledge and without noise
        self.obs_process_0 = torch.matmul(self.H, self.state_process_prior_0)

        # predict the 1st moment of x
        self.m1x_prev_prior = self.m1x_prior
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)

        # predict the 1st moment of y
        self.m1y = torch.matmul(self.H, self.m1x_prior)

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

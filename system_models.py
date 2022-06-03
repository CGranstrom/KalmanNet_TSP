import sys

import torch
from torch.distributions import MultivariateNormal
from torch.distributions.multivariate_normal import MultivariateNormal

from path_models import path_model

sys.path.insert(1, path_model)
from parameters import delta_t, delta_t_gen, variance

_model_param_name_map = {"pendulum": delta_t, "pendulum_gen": delta_t_gen}

if torch.cuda.is_available():
    cuda0 = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    cuda0 = torch.device("cpu")
    print("Running on the CPU")


class LinearSystemModel:
    def __init__(self, f, q, h, r, t, t_test):

        self.outlier_p = 0
        self.rayleigh_sigma = 10000

        # motion model
        self.f = f
        self.m = self.f.size()[0]

        self.q = q
        self.Q = q ** 2 * torch.eye(self.m)

        # observation model
        self.h = h
        self.n = self.h.size()[0]

        self.r = r
        self.R = r ** 2 * torch.eye(self.n)

        # assign t and t_test
        self.t = t
        self.t_test = t_test

    def init_sequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    def update_covariance_gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def update_covariance_matrix(self, Q, R):

        self.Q = Q
        self.R = R

    def _generate_sequence(self, Q_gen, R_gen, T):

        # pre-allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # pre-allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # set x0 to be x previous
        self.x_prev = self.m1x_0

        # Outliers
        if self.outlier_p > 0:
            b_matrix = torch.bernoulli(self.outlier_p * torch.ones(T))

        # generate sequence iteratively
        for t in range(0, T):
            # state evolution
            # process noise
            if self.q == 0:
                xt = self.f.matmul(self.x_prev)
            else:
                xt = self.f.matmul(self.x_prev)
                mean = torch.zeros([self.m])
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                # eq = torch.normal(mean, self.q)
                eq = torch.reshape(eq[:], [self.m, 1])
                # additive process noise
                xt = torch.add(xt, eq)

            # emission
            # observation noise
            if self.r == 0:
                yt = self.h.matmul(xt)
            else:
                yt = self.h.matmul(xt)
                mean = torch.zeros([self.n])
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample()
                er = torch.reshape(er[:], [self.n, 1])
                # mean = torch.zeros([self.n,1])
                # er = torch.normal(mean, self.r)

                # additive observation noise
                yt = torch.add(yt, er)

            # outliers
            if self.outlier_p > 0:
                if b_matrix[t] != 0:
                    btdt = self.rayleigh_sigma * torch.sqrt(
                        -2 * torch.log(torch.rand(self.n, 1))
                    )
                    yt = torch.add(yt, btdt)

            # squeeze to array
            # save current state to trajectory array
            self.x[:, t] = torch.squeeze(xt)

            # save current observation to trajectory array
            self.y[:, t] = torch.squeeze(yt)

            # save current to previous
            self.x_prev = xt

    def generate_batch(self, size, t, random_init=False):

        seq_init = False
        t_test = 0

        # allocate empty arrays for input and target
        self.input = torch.empty(size, self.n, t)
        self.target = torch.empty(size, self.m, t)

        init_conditions = self.m1x_0

        # generate examples
        for i in range(0, size):
            # generate sequence
            # randomize initial conditions to get a rich dataset

            if random_init:
                init_conditions = torch.rand_like(self.m1x_0) * variance
            if seq_init:
                init_conditions = self.x_prev
                if (i * t % t_test) == 0:
                    init_conditions = torch.zeros_like(self.m1x_0)

            self.init_sequence(init_conditions, self.m2x_0)
            self._generate_sequence(self.Q, self.R, t)

            # training sequence input and output
            self.input[i, :, :] = self.y
            self.target[i, :, :] = self.x


class ExtendedSystemModel(LinearSystemModel):
    def __init__(self, f, q, h, r, t, t_test, m, n, model_name):
        super().__init__(f, q, h, r, t, t_test)

        # motion model
        self.model_name = model_name

        self.m = m

        self.delta_t = delta_t
        param_name = _model_param_name_map.get(self.model_name)
        if param_name:
            self.Q = (
                q ** 2
                * torch.tensor(
                    [
                        [(param_name ** 3) / 3, (param_name ** 2) / 2],
                        [(param_name ** 2) / 2, param_name],
                    ]
                )
            )
        else:
            self.Q = q ** 2 * torch.eye(self.m)

        # observation model
        self.n = n

    def init_sequence(self, m1x_0, m2x_0):

        super().init_sequence(
            m1x_0=torch.squeeze(m1x_0).to(cuda0), m2x_0=torch.squeeze(m2x_0).to(cuda0)
        )

    def _generate_sequence(self, Q_gen, R_gen, T):

        # pre-allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # pre-allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # set x0 to be x previous
        self.x_prev = self.m1x_0

        # generate sequence iteratively
        for t in range(0, T):
            # state evolution
            # process noise
            if self.q == 0:
                xt = self.f(self.x_prev)
            else:
                xt = self.f(self.x_prev)
                mean = torch.zeros([self.m])
                if self.model_name == "pendulum":
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                    eq = distrib.rsample()
                else:
                    eq = torch.normal(mean, self.q)

                # Additive Process Noise
                xt = torch.add(xt, eq)

            # emission
            yt = self.h(xt)

            # observation noise
            mean = torch.zeros([self.n])
            er = torch.normal(mean, self.r)
            # er = np.random.multivariate_normal(mean, R_gen, 1)
            # er = torch.transpose(torch.tensor(er), 0, 1)

            # additive observation noise
            yt = torch.add(yt, er)

            # squeeze to array
            # save current state to trajectory array
            self.x[:, t] = torch.squeeze(xt)

            # save current observation to trajectory array
            self.y[:, t] = torch.squeeze(yt)

            # save current to previous
            self.x_prev = xt

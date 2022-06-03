import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from filing_paths import path_model
import sys

from linear_system_model import LinearSystemModel

sys.path.insert(1, path_model)
from parameters import delta_t, delta_t_gen, variance


if torch.cuda.is_available():
    cuda0 = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    cuda0 = torch.device("cpu")
    print("Running on the CPU")


class ExtendedSystemModel(LinearSystemModel):
    def __init__(self, f, q, h, r, t, t_test, m, n, model_name):
        super().__init__(f, q, h, r, t, t_test)

        # motion model
        self.model_name = model_name

        self.m = m

        self.delta_t = delta_t
        if self.model_name == "pendulum":
            self.Q = (
                q
                * q
                * torch.tensor(
                    [
                        [(delta_t ** 3) / 3, (delta_t ** 2) / 2],
                        [(delta_t ** 2) / 2, delta_t],
                    ]
                )
            )
        elif self.model_name == "pendulum_gen":
            self.Q = (
                q
                * q
                * torch.tensor(
                    [
                        [(delta_t_gen ** 3) / 3, (delta_t_gen ** 2) / 2],
                        [(delta_t_gen ** 2) / 2, delta_t_gen],
                    ]
                )
            )
        else:
            self.Q = q * q * torch.eye(self.m)

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

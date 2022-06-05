import autograd.numpy as np
import torch
from autograd import grad, jacobian
from parameters import (
    a_mot,
    a_mot_mod,
    a_obs,
    a_obs_mod,
    alpha_mot,
    alpha_mot_mod,
    alpha_obs,
    alpha_obs_mod,
    beta_mot,
    beta_mot_mod,
    beta_obs,
    beta_obs_mod,
    phi_mot,
    phi_mot_mod,
)
from torch import autograd


def f(x):
    return alpha_mot * torch.sin(beta_mot * x + phi_mot) + a_mot


def h(x):
    return alpha_obs * (beta_obs * x + a_obs) ** 2


def fInacc(x):
    return alpha_mot_mod * torch.sin(beta_mot_mod * x + phi_mot_mod) + a_mot_mod


def hInacc(x):
    return alpha_obs_mod * (beta_obs_mod * x + a_obs_mod) ** 2


def getJacobian(x, a):

    try:
        if x.size()[1] == 1:
            y = torch.reshape((x.t), [x.size()[0]])
    except:
        y = torch.reshape((x.t), [x.size()[0]])

    if a == "ObsAcc":
        g = h
    elif a == "ModAcc":
        g = f
    elif a == "ObsInacc":
        g = hInacc
    elif a == "ModInacc":
        g = fInacc

    return autograd.functional.jacobian(g, y)

import time

import torch
import torch.nn as nn

from EKF import ExtendedKalmanFilter


def EKFTest(SysModel, test_input, test_target, modelKnowledge="full", allStates=True):

    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction="mean")

    # MSE [linear]
    MSE_EKF_linear_arr = torch.empty(N_T)

    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KG_array = torch.zeros_like(EKF.KG_array)
    EKF_out = torch.empty([N_T, SysModel.m, SysModel.t_test])
    start = time.time()
    sub_time = 0
    for j in range(0, N_T):
        EKF.GenerateSequence(test_input[j, :, :], EKF.T_test)
        sub_start = time.time()
        print(
            f"On step {j}, {j/N_T*100}% done, {N_T-j} steps remain. Previous step took {sub_time} seconds."
        )
        if allStates:
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x, test_target[j, :, :]).item()
        else:
            loc = torch.tensor([True, False, True, False])
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x[loc, :], test_target[j, :, :]).item()
        KG_array = torch.add(EKF.KG_array, KG_array)
        EKF_out[j, :, :] = EKF.x
        sub_end = time.time()
        sub_time = sub_end - sub_start
    end = time.time()
    t = end - start
    # Average KG_array over Test Examples
    KG_array /= N_T

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_dB_std = torch.std(MSE_EKF_linear_arr, unbiased=True)
    MSE_EKF_dB_std = 10 * torch.log10(MSE_EKF_dB_std)

    print("EKF - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("EKF - MSE STD:", MSE_EKF_dB_std, "[dB]")
    # Print Run time
    print("Inference time:", t)
    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out]

import time

import torch
import torch.nn as nn

from extended_data import NUM_TEST_POINTS
from linear_kalman_filter import KalmanFilter


def KFTest(SysModel, test_input, test_target):

    # LOSS
    loss_fn = nn.MSELoss(reduction="mean")

    # MSE [linear]
    MSE_KF_linear_arr = torch.empty(NUM_TEST_POINTS)

    start = time.time()
    KF = KalmanFilter(SysModel)
    KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    for j in range(0, NUM_TEST_POINTS):

        KF.GenerateSequence(test_input[j, :, :], KF.T_test)

        MSE_KF_linear_arr[j] = loss_fn(KF.x, test_target[j, :, :]).item()
        # MSE_KF_linear_arr[j] = loss_function(test_input[j, :, :], test_target[j, :, :]).item()
    end = time.time()
    t = end - start

    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    # Standard deviation
    MSE_KF_dB_std = torch.std(MSE_KF_linear_arr, unbiased=True)
    MSE_KF_dB_std = 10 * torch.log10(MSE_KF_dB_std)

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("EKF - MSE STD:", MSE_KF_dB_std, "[dB]")
    # Print Run time
    print("Inference time:", t)

    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg]

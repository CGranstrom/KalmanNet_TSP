import torch.nn as nn
import torch
import time

from filterpy.kalman import (
    UnscentedKalmanFilter,
    MerweScaledSigmaPoints,
    JulierSigmaPoints,
)


def UKFTest(
    SysModel,
    test_input,
    test_target,
    modelKnowledge="full",
    allStates=True,
    init_cond=None,
):

    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction="mean")

    # MSE [linear]
    MSE_UKF_linear_arr = torch.empty(N_T)
    # points = JulierSigmaPoints(n=SysModel.m)
    points = MerweScaledSigmaPoints(SysModel.m, alpha=0.1, beta=2.0, kappa=-1)

    def fx(x, dt):
        return SysModel.f(torch.from_numpy(x).float()).cpu().numpy()

    def hx(x):
        return SysModel.h(torch.from_numpy(x).float()).cpu().numpy()

    UKF = UnscentedKalmanFilter(
        dim_x=SysModel.m,
        dim_z=SysModel.n,
        dt=SysModel.delta_t,
        fx=fx,
        hx=hx,
        points=points,
    )
    UKF.x = SysModel.m1x_0.cpu().numpy()  # initial state
    UKF.P = (
        (SysModel.m2x_0 + 1e-5 * torch.eye(SysModel.m)).cpu().numpy()
    )  # initial uncertainty
    UKF.R = SysModel.R.cpu().numpy()
    UKF.Q = SysModel.Q.cpu().numpy()

    UKF_out = torch.empty([N_T, SysModel.m, SysModel.t_test])

    start = time.time()
    sub_time = 0
    for j in range(0, N_T):
        sub_start = time.time()
        if init_cond is not None:
            UKF.x = torch.unsqueeze(init_cond[j, :], 1).numpy()

        print(
            f"On step {j}, {j/N_T*100}% done, {N_T-j} steps remain. Previous step took {sub_time} seconds."
        )
        for z in range(0, SysModel.t_test):
            UKF.predict()
            UKF.update(test_input[j, :, z].cpu().numpy())
            UKF_out[j, :, z] = torch.from_numpy(UKF.x)

        if allStates:
            MSE_UKF_linear_arr[j] = loss_fn(
                UKF_out[j, :, :], test_target[j, :, :]
            ).item()
        else:
            loc = torch.tensor([True, False, True, False])
            MSE_UKF_linear_arr[j] = loss_fn(
                UKF_out[j, :, :][loc, :], test_target[j, :, :]
            ).item()
        sub_end = time.time()
        sub_time = sub_end - sub_start

    end = time.time()
    t = end - start

    MSE_UKF_linear_avg = torch.mean(MSE_UKF_linear_arr)
    MSE_UKF_dB_avg = 10 * torch.log10(MSE_UKF_linear_avg)
    # Standard deviation
    MSE_UKF_dB_std = torch.std(MSE_UKF_linear_arr, unbiased=True)
    MSE_UKF_dB_std = 10 * torch.log10(MSE_UKF_dB_std)

    print("UKF - MSE LOSS:", MSE_UKF_dB_avg, "[dB]")
    print("UKF - MSE STD:", MSE_UKF_dB_std, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_UKF_linear_arr, MSE_UKF_linear_avg, MSE_UKF_dB_avg, UKF_out]

import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import sys
from datetime import datetime

import torch.nn as nn

from EKF_test import EKFTest
from extended_data import (
    NUM_CROSS_VAL_EXAMPLES,
    NUM_TEST_POINTS,
    NUM_TRAINING_EXAMPLES,
    data_gen,
    data_loader,
    data_loader_gpu,
    decimate_and_perturb_data,
    short_traj_split,
)
from kalman_net import KalmanNet
from path_models import path_model
from PF_test import PFTest
from pipeline_KF import PipelineKF
from plot import Plot_RTS as Plot
from system_models import ExtendedSystemModel
from UKF_test import UKFTest

sys.path.insert(1, path_model)
from model import f, fInacc, h
from parameters import m, m1x_0, m2x_0, n, t, t_test

if torch.cuda.is_available():
    dev = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")


print("Pipeline Start")

################
### Get time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%h:%M:%S")
strTime = strToday + "_" + strNow
print("Current time =", strTime)
path_results = "RTSNet/"


r2 = torch.tensor([16, 4, 1, 0.01, 1e-4])
vdB = 0  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
qopt = torch.sqrt(q2)
# qopt = torch.tensor([0.2, 4, 1, 0.1, 0.01])

for index in range(0, len(r2)):
    ####################
    ### Design Model ###
    ####################

    print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[index]))
    print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[index]))

    # True model
    r = torch.sqrt(r2[index])
    q = torch.sqrt(q2[index])
    sys_model = ExtendedSystemModel(f, q, h, r, t, t_test, m, n, "Toy")
    sys_model.init_sequence(m1x_0, m2x_0)

    # Mismatched model
    sys_model_partial = ExtendedSystemModel(fInacc, q, h, r, t, t_test, m, n, "Toy")
    sys_model_partial.init_sequence(m1x_0, m2x_0)

    ###################################
    ### Data Loader (Generate Data) ###
    ###################################
    dataFolderName = "simulations/toy_problems" + "/"
    dataFileName = "T100.pt"
    print("Start Data Gen")
    data_gen(sys_model, dataFolderName + dataFileName, t, t_test, randomInit=False)
    print("Data Load")
    [
        train_input,
        train_target,
        cv_input,
        cv_target,
        test_input,
        test_target,
    ] = data_loader_gpu(dataFolderName + dataFileName)
    print("trainset size:", train_target.size())
    print("cvset size:", cv_target.size())
    print("testset size:", test_target.size())

    ################################
    ### Evaluate EKF, UKF and PF ###
    ################################

    print("Searched optimal 1/q2 [dB]: ", 10 * torch.log10(1 / qopt[index] ** 2))
    sys_model = ExtendedSystemModel(f, qopt[index], h, r, t, t_test, m, n, "Toy")
    sys_model.init_sequence(m1x_0, m2x_0)

    sys_model_partial = ExtendedSystemModel(
        fInacc, qopt[index], h, r, t, t_test, m, n, "Toy"
    )
    sys_model_partial.init_sequence(m1x_0, m2x_0)
    print("Evaluate Kalman Filter True")
    [
        MSE_EKF_linear_arr,
        MSE_EKF_linear_avg,
        MSE_EKF_dB_avg,
        EKF_KG_array,
        EKF_out,
    ] = EKFTest(sys_model, test_input, test_target)
    print("Evaluate Kalman Filter Partial")
    [
        MSE_KF_linear_arr_partial,
        MSE_KF_linear_avg_partial,
        MSE_KF_dB_avg_partial,
        EKF_KG_array_partial,
        EKF_out_partial,
    ] = EKFTest(sys_model_partial, test_input, test_target)

    print("Evaluate UKF True")
    [MSE_UKF_linear_arr, MSE_UKF_linear_avg, MSE_UKF_dB_avg, UKF_out] = UKFTest(
        sys_model, test_input, test_target
    )
    print("Evaluate UKF Partial")
    [
        MSE_UKF_linear_arr_partial,
        MSE_UKF_linear_avg_partial,
        MSE_UKF_dB_avg_partial,
        UKF_out_partial,
    ] = UKFTest(sys_model_partial, test_input, test_target)

    print("Evaluate PF True")
    [MSE_PF_linear_arr, MSE_PF_linear_avg, MSE_PF_dB_avg, PF_out] = PFTest(
        sys_model, test_input, test_target
    )
    print("Evaluate PF Partial")
    [
        MSE_PF_linear_arr_partial,
        MSE_PF_linear_avg_partial,
        MSE_PF_dB_avg_partial,
        PF_out_partial,
    ] = PFTest(sys_model_partial, test_input, test_target)

    # DatafolderName = 'Data' + '/'
    # DataResultName = '10x10_Ttest1000'
    # torch.save({
    #             'MSE_KF_linear_arr': MSE_KF_linear_arr,
    #             'MSE_KF_dB_avg': MSE_KF_dB_avg,
    #             'MSE_RTS_linear_arr': MSE_RTS_linear_arr,
    #             'MSE_RTS_dB_avg': MSE_RTS_dB_avg,
    #             }, DatafolderName+DataResultName)

    ##################
    ###  KalmanNet ###
    ##################
    print("Start k_net pipeline")
    modelFolder = "k_net" + "/"
    KNet_Pipeline = PipelineKF(strTime, "k_net", "KalmanNet")
    KNet_Pipeline.set_ss_model(sys_model)
    KNet_model = KalmanNet(sys_model)
    KNet_Pipeline.set_k_net_model(KNet_model)
    KNet_Pipeline.setTrainingParams(
        n_Epochs=200, n_Batch=10, learningRate=1e-3, weightDecay=1e-4
    )

    # KNet_Pipeline.k_net_model = torch.load(modelFolder+"model_KNet.pt")

    KNet_Pipeline.NN_train(
        NUM_TRAINING_EXAMPLES,
        train_input,
        train_target,
        NUM_CROSS_VAL_EXAMPLES,
        cv_input,
        cv_target,
    )
    [
        KNet_MSE_test_linear_arr,
        KNet_MSE_test_linear_avg,
        KNet_MSE_test_dB_avg,
        KNet_test,
    ] = KNet_Pipeline.NNTest(NUM_TEST_POINTS, test_input, test_target)
    KNet_Pipeline.save()

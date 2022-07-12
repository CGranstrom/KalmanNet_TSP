import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
from datetime import datetime

import torch.nn as nn

from extended_data import (
    NUM_CROSS_VAL_EXAMPLES,
    NUM_TEST_POINTS,
    NUM_TRAINING_EXAMPLES,
    F,
    F_rotated,
    H,
    H_rotated,
    T,
    T_test,
    data_gen,
    data_loader,
    data_loader_gpu,
    decimate_and_perturb_data,
    m,
    m1_0,
    m2_0,
    n,
    short_traj_split,
)
from kalman_filter_test import KFTest
from kalman_net import KalmanNet
from pipeline_KF import PipelineKF
from plot import Plot_RTS as Plot
from system_models import LinearSystemModel

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

####################
### Design Model ###
####################
r2 = torch.tensor([10, 1.0, 0.1, 1e-2, 1e-3])
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)

for index in range(0, len(r2)):

    print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[index]))
    print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[index]))

    # True model
    r = torch.sqrt(r2[index])
    q = torch.sqrt(q2[index])
    sys_model = LinearSystemModel(F, q, H_rotated, r, T, T_test)
    sys_model.init_sequence(m1_0, m2_0)

    # Mismatched model
    sys_model_partialh = LinearSystemModel(F, q, H, r, T, T_test)
    sys_model_partialh.init_sequence(m1_0, m2_0)

    ###################################
    ### Data Loader (Generate Data) ###
    ###################################
    dataFolderName = "simulations/linear_canonical/H_rotated" + "/"
    dataFileName = [
        "2x2_rq-1010_T100.pt",
        "2x2_rq020_T100.pt",
        "2x2_rq1030_T100.pt",
        "2x2_rq2040_T100.pt",
        "2x2_rq3050_T100.pt",
    ]
    print("Start Data Gen")
    data_gen(
        sys_model, dataFolderName + dataFileName[index], T, T_test, randomInit=False
    )
    print("Data Load")
    [
        train_input,
        train_target,
        cv_input,
        cv_target,
        test_input,
        test_target,
    ] = data_loader_gpu(dataFolderName + dataFileName[index])
    print("trainset size:", train_target.size())
    print("cvset size:", cv_target.size())
    print("testset size:", test_target.size())

    ##############################
    ### Evaluate Kalman Filter ###
    ##############################
    print("Evaluate Kalman Filter True")
    [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(
        sys_model, test_input, test_target
    )
    print("Evaluate Kalman Filter Partial")
    [
        MSE_KF_linear_arr_partialh,
        MSE_KF_linear_avg_partialh,
        MSE_KF_dB_avg_partialh,
    ] = KFTest(sys_model_partialh, test_input, test_target)

    DatafolderName = "filters/linear" + "/"
    DataResultName = "KF_HRotated" + dataFileName[index]
    torch.save(
        {
            "MSE_KF_linear_arr": MSE_KF_linear_arr,
            "MSE_KF_dB_avg": MSE_KF_dB_avg,
            "MSE_KF_linear_arr_partialh": MSE_KF_linear_arr_partialh,
            "MSE_KF_dB_avg_partialh": MSE_KF_dB_avg_partialh,
        },
        DatafolderName + DataResultName,
    )

    ##################
    ###  KalmanNet ###
    ##################
    print("Start k_net pipeline")
    print("k_net with full model info")
    modelFolder = "k_net" + "/"
    KNet_Pipeline = PipelineKF(strTime, "k_net", "KNet_" + dataFileName[index])
    KNet_Pipeline.set_ss_model(sys_model)
    KNet_model = KalmanNet(sys_model)
    KNet_Pipeline.set_k_net_model(KNet_model)
    KNet_Pipeline.setTrainingParams(
        n_Epochs=500, n_Batch=30, learningRate=1e-3, weightDecay=1e-5
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

    print("k_net with partial model info")
    modelFolder = "k_net" + "/"
    KNet_Pipeline = PipelineKF(strTime, "k_net", "KNetPartial_" + dataFileName[index])
    KNet_Pipeline.set_ss_model(sys_model_partialh)
    KNet_model = KalmanNet(sys_model_partialh)
    KNet_Pipeline.set_k_net_model(KNet_model)
    KNet_Pipeline.setTrainingParams(
        n_Epochs=500, n_Batch=30, learningRate=1e-3, weightDecay=1e-5
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

    print("k_net with estimated h")
    modelFolder = "k_net" + "/"
    KNet_Pipeline = PipelineKF(strTime, "k_net", "KNetEstH_" + dataFileName[index])
    print("True Observation matrix h:", H_rotated)
    ### Least square estimation of h
    X = torch.squeeze(train_target[:, :, 0]).to(dev, non_blocking=True)
    Y = torch.squeeze(train_input[:, :, 0]).to(dev, non_blocking=True)
    for t in range(1, T):
        X_t = torch.squeeze(train_target[:, :, t])
        Y_t = torch.squeeze(train_input[:, :, t])
        X = torch.cat((X, X_t), 0)
        Y = torch.cat((Y, Y_t), 0)
    Y_1 = torch.unsqueeze(Y[:, 0], 1)
    Y_2 = torch.unsqueeze(Y[:, 1], 1)
    H_row1 = torch.matmul(
        torch.matmul(torch.inverse(torch.matmul(X.T, X)), X.T), Y_1
    ).to(dev, non_blocking=True)
    H_row2 = torch.matmul(
        torch.matmul(torch.inverse(torch.matmul(X.T, X)), X.T), Y_2
    ).to(dev, non_blocking=True)
    H_hat = torch.cat((H_row1.T, H_row2.T), 0)
    print("Estimated Observation matrix h:", H_hat)

    # Estimated model
    sys_model_esth = LinearSystemModel(F, q, H_hat, r, T, T_test)
    sys_model_esth.init_sequence(m1_0, m2_0)

    KNet_Pipeline.set_ss_model(sys_model_esth)
    KNet_model = KalmanNet(sys_model_esth)
    KNet_Pipeline.set_k_net_model(KNet_model)
    KNet_Pipeline.setTrainingParams(
        n_Epochs=500, n_Batch=30, learningRate=1e-3, weightDecay=1e-5
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

import torch

from simulations.lorenz_attractor.model import fRotate

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import random
import sys
from datetime import datetime

import torch.nn as nn

from EKF_test import EKFTest
from extended_data import (
    NUM_CROSS_VAL_EXAMPLES,
    NUM_TEST_POINTS,
    NUM_TRAINING_EXAMPLES,
    DataGen_True,
    data_gen,
    data_loader_gpu,
    decimate_and_perturb_data,
    short_traj_split,
)
from kalman_net import ExtendedKalmanNet
from path_models import path_model
from pipeline_KF import PipelineEKF
from plot import Plot_extended as Plot
from system_models import ExtendedSystemModel

sys.path.insert(1, path_model)
from model import f, fInacc, fRotate, h, hInacc
from parameters import (
    delta_t,
    delta_t_gen,
    lambda_q_mod,
    lambda_r_mod,
    m,
    m1x_0,
    m2x_0,
    n,
    t,
    t_test,
)

if torch.cuda.is_available():
    device = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# print("Start Data Gen")
# offset = 0
# r2 = torch.tensor([10])
# r = torch.sqrt(r2)
# q_gen = 0
# DatafolderName = 'simulations/lorenz_attractor/data' + '/'
# data_gen = 'data_gen.pt'
# data_gen_file = torch.load(DatafolderName+data_gen, map_location=device)
# [true_sequence] = data_gen_file['All Data']
# [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, NUM_TEST_POINTS, h, r, offset)

# vdB = -20 # ratio v=q2/r2
# v = 10**(vdB/10)

# q2_gen = torch.mul(v,r2)
# q_gen = torch.sqrt(q2_gen)
# print("data obs noise 1/r2 [dB]: ", 10 * torch.log10(1/r**2))
# print("data process noise 1/q2 [dB]: ", 10 * torch.log10(1/q_gen**2))
# #Model
# sys_model = LinearSystemModel(f, q_gen, h, r, t, t_test, m, n,"Lor")
# sys_model.InitSequence(m1x_0, m2x_0)
# #Generate and load data
# DataGen(sys_model, DatafolderName + dataFileName, t, t_test)
# print("Data Load")
# [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(DatafolderName + dataFileName)
# print(test_target.size())

# dataFileName_long = 'data_pen_highresol_q1e-5_long.pt'
# true_sequence = torch.load(dataFolderName + dataFileName_long, map_location=device)
# [test_target_zeroinit, test_input_zeroinit] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, NUM_TEST_POINTS, h, lambda_r_mod, offset=0)
# test_target = torch.empty(NUM_TEST_POINTS,m,t_test)
# test_input = torch.empty(NUM_TEST_POINTS,n,t_test)
### Random init
# print("random init testing data")
# for test_i in range(NUM_TEST_POINTS):
#    rand_seed = random.randint(0,10000-t_test-1)
#    test_target[test_i,:,:] = test_target_zeroinit[test_i,:,rand_seed:rand_seed+t_test]
#    test_input[test_i,:,:] = test_input_zeroinit[test_i,:,rand_seed:rand_seed+t_test]
# test_target = test_target_zeroinit[:,:,0:t_test]
# test_input = test_input_zeroinit[:,:,0:t_test]

r2_gen = torch.tensor([1])
r_gen = torch.sqrt(r2_gen)
rindex = 0
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2_gen = torch.mul(v, r2_gen)
q_gen = torch.sqrt(q2_gen)
DatafolderName = "simulations/lorenz_attractor/data" + "/"
dataFileName = [
    "data_lor_v20_rq020_T1000.pt"
]  # ,'data_lor_v20_r1e-1_T2000.pt','data_lor_v20_r1e-2_T2000.pt']

sys_model = ExtendedSystemModel(f, q_gen, h, r_gen, t, t_test, m, n, "Lor")
sys_model.init_sequence(m1x_0, m2x_0)

# [train_input_long, train_target_long, cv_input_long, cv_target_long, test_input, test_target] = DataLoader_GPU(DatafolderName + dataFileName[rindex])
# test_input = test_input[50:51,:,:]
# test_target = test_target[50:51,:,:]

print("Start Data Gen")
T = 1000
data_gen(sys_model, DatafolderName + dataFileName[rindex], T, t_test)
print("Data Load")
[
    train_input_long,
    train_target_long,
    cv_input_long,
    cv_target_long,
    test_input,
    test_target,
] = torch.load(DatafolderName + dataFileName[rindex], map_location=device)
print("trainset long:", train_target_long.size())

print("testset:", test_target.size())

r2 = torch.tensor([1e-2])
r = torch.sqrt(r2)
print("data obs noise 1/r2 [dB]: ", 10 * torch.log10(1 / r_gen ** 2))
print("data process noise 1/q2 [dB]: ", 10 * torch.log10(1 / q_gen ** 2))
# dataFileName = ['data_pen_r1_1.pt','data_pen_r1_2.pt','data_pen_r1_3.pt','data_pen_r1_4.pt','data_pen_r1_5.pt']
for index in range(0, len(r)):

    # Model

    # sys_model_partialf = LinearSystemModel(fInacc, q_gen, h, r_gen, t, t_test, m, n,'lor')
    # sys_model_partialf.InitSequence(m1x_0, m2x_0)

    # sys_model_partialf_optq = LinearSystemModel(fInacc, q[index], h, r_gen, t, t_test, m, n,'lor')
    # sys_model_partialf_optq.InitSequence(m1x_0, m2x_0)

    sys_model_partialh = ExtendedSystemModel(
        f, q_gen, hInacc, r_gen, T, t_test, m, n, "lor"
    )
    sys_model_partialh.init_sequence(m1x_0, m2x_0)

    sys_model_partialh_optr = ExtendedSystemModel(
        f, q_gen, hInacc, r[index], T, t_test, m, n, "lor"
    )
    sys_model_partialh_optr.init_sequence(m1x_0, m2x_0)

    # Evaluate EKF True
    [
        MSE_EKF_linear_arr,
        MSE_EKF_linear_avg,
        MSE_EKF_dB_avg,
        EKF_KG_array,
        EKF_out,
    ] = EKFTest(sys_model, test_input, test_target)

    ### Search EKF process model mismatch
    # print("search 1/q2 [dB]: ", 10 * torch.log10(1/q[index]**2))
    # [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partialf, test_input, test_target)
    # [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partialf_optq, test_input, test_target)
    ### Search EKF observation model mismatch
    print("search 1/r2 [dB]: ", 10 * torch.log10(1 / r[index] ** 2))
    [
        MSE_EKF_linear_arr_partial,
        MSE_EKF_linear_avg_partial,
        MSE_EKF_dB_avg_partial,
        EKF_KG_array_partial,
        EKF_out_partial,
    ] = EKFTest(sys_model_partialh, test_input, test_target)
    [
        MSE_EKF_linear_arr_partial,
        MSE_EKF_linear_avg_partial,
        MSE_EKF_dB_avg_partial,
        EKF_KG_array_partial,
        EKF_out_partial,
    ] = EKFTest(sys_model_partialh_optr, test_input, test_target)

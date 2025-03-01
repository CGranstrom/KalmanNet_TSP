import math
import os

import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if torch.cuda.is_available():
    dev = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

# Size of dataset

NUM_TRAINING_EXAMPLES = 1000

# Number of Cross Validation Examples
NUM_CROSS_VAL_EXAMPLES = 100

# I think this is supposed to be test points...
NUM_TEST_POINTS = 200

# Sequence Length for linear Case
T = 100
T_test = 100

#################
## Design #10 ###
#################
F10 = torch.tensor(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)

H10 = torch.tensor(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)

############
## 2 x 2 ###
############
m = 2
n = 2
F = F10[0:m, 0:m]
H = torch.eye(2)
m1_0 = torch.tensor([[0.0], [0.0]]).to(dev)
# m1x_0_design = torch.tensor([[10.0], [-10.0]])
m2_0 = 0 * 0 * torch.eye(m).to(dev)


#############
### 5 x 5 ###
#############
# m = 5
# n = 5
# f = F10[0:m, 0:m]
# h = H10[0:n, 10-m:10]
# m1_0 = torch.zeros(m, 1).to(dev)
# # m1x_0_design = torch.tensor([[1.0], [-1.0], [2.0], [-2.0], [0.0]]).to(dev)
# m2_0 = 0 * 0 * torch.eye(m).to(dev)

##############
## 10 x 10 ###
##############
# m = 10
# n = 10
# f = F10[0:m, 0:m]
# h = H10
# m1_0 = torch.zeros(m, 1).to(dev)
# # m1x_0_design = torch.tensor([[10.0], [-10.0]])
# m2_0 = 0 * 0 * torch.eye(m).to(dev)

# Inaccurate model knowledge based on matrix rotation
alpha_degree = 10
rotate_alpha = torch.tensor([alpha_degree / 180 * torch.pi]).to(dev)
cos_alpha = torch.cos(rotate_alpha)
sin_alpha = torch.sin(rotate_alpha)
rotate_matrix = torch.tensor([[cos_alpha, -sin_alpha], [sin_alpha, cos_alpha]]).to(dev)
F_rotated = torch.mm(F, rotate_matrix)  # inaccurate process model
H_rotated = torch.mm(H, rotate_matrix)  # inaccurate observation model


def DataGen_True(SysModel_data, fileName, T):

    SysModel_data.generate_batch(1, T, random_init=False)
    test_input = SysModel_data.input
    test_target = SysModel_data.target

    # torch.save({"True Traj":[test_target],
    #             "Obs":[test_input]},filename)
    torch.save([test_input, test_target], fileName)


def data_gen(SysModel_data, fileName, T, T_test, randomInit=False):

    # generate training sequence
    SysModel_data.generate_batch(NUM_TRAINING_EXAMPLES, T, random_init=randomInit)
    training_input = SysModel_data.input
    training_target = SysModel_data.target

    # generate validation sequence
    SysModel_data.generate_batch(NUM_CROSS_VAL_EXAMPLES, T, random_init=randomInit)
    cv_input = SysModel_data.input
    cv_target = SysModel_data.target

    # generate test sequence
    SysModel_data.generate_batch(NUM_TEST_POINTS, T_test, random_init=randomInit)
    test_input = SysModel_data.input
    test_target = SysModel_data.target

    # save data
    torch.save(
        [training_input, training_target, cv_input, cv_target, test_input, test_target],
        fileName,
    )


def data_loader(filename):

    [
        training_input,
        training_target,
        cv_input,
        cv_target,
        test_input,
        test_target,
    ] = torch.load(filename, map_location=dev)
    return [
        training_input,
        training_target,
        cv_input,
        cv_target,
        test_input,
        test_target,
    ]


def data_loader_gpu(filename):
    [
        training_input,
        training_target,
        cv_input,
        cv_target,
        test_input,
        test_target,
    ] = torch.utils.data.data_loader(torch.load(filename), pin_memory=False)
    training_input = training_input.squeeze().to(dev)
    training_target = training_target.squeeze().to(dev)
    cv_input = cv_input.squeeze().to(dev)
    cv_target = cv_target.squeeze().to(dev)
    test_input = test_input.squeeze().to(dev)
    test_target = test_target.squeeze().to(dev)
    return [
        training_input,
        training_target,
        cv_input,
        cv_target,
        test_input,
        test_target,
    ]


def decimate_data(all_tensors, t_gen, t_mod, offset=0):

    # ratio: defines the relation between the sampling time of the true process and of the model (has to be an integer)
    ratio = round(t_mod / t_gen)

    i = 0
    all_tensors_out = all_tensors
    for tensor in all_tensors:
        tensor = tensor[:, (0 + offset) :: ratio]
        if i == 0:
            all_tensors_out = torch.cat([tensor], dim=0).view(
                1, all_tensors.size()[1], -1
            )
        else:
            all_tensors_out = torch.cat([all_tensors_out, tensor], dim=0)
        i += 1

    return all_tensors_out


def decimate_and_perturb_data(
    true_process, delta_t, delta_t_mod, N_examples, h, lambda_r, offset=0
):

    # Decimate high resolution process
    decimated_process = decimate_data(true_process, delta_t, delta_t_mod, offset)

    noise_free_obs = get_obs(decimated_process, h)

    # Replicate for computation purposes
    decimated_process = torch.cat(int(N_examples) * [decimated_process])
    noise_free_obs = torch.cat(int(N_examples) * [noise_free_obs])

    # Observations; additive Gaussian Noise
    observations = noise_free_obs + torch.randn_like(decimated_process) * lambda_r

    return [decimated_process, observations]


def get_obs(sequences, h):
    i = 0
    sequences_out = torch.zeros_like(sequences)
    for sequence in sequences:
        for t in range(sequence.size()[1]):
            sequences_out[i, :, t] = h(sequence[:, t])
    i = i + 1

    return sequences_out


def short_traj_split(data_target, data_input, T):
    data_target = list(torch.split(data_target, T, 2))
    data_input = list(torch.split(data_input, T, 2))
    data_target.pop()
    data_input.pop()
    data_target = torch.squeeze(torch.cat(list(data_target), dim=0))
    data_input = torch.squeeze(torch.cat(list(data_input), dim=0))
    return [data_target, data_input]

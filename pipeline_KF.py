import random
import time

import torch
import torch.nn as nn
from torch import nn as nn

from kalman_net import KalmanNet
from plot import Plot


class PipelineKF:
    def __init__(self, time, folder_name, model_name):
        super().__init__()
        self.time = time
        self.folder_name = folder_name + "/"
        self.model_name = model_name
        self.model_filename = self.folder_name + "model_" + self.model_name
        self.pipeline_name = self.folder_name + "pipeline_" + self.model_name

    def save(self):
        torch.save(self, self.pipeline_name)

    def set_ss_model(self, ss_model):
        self.ss_model = ss_model

    def set_model(self, model):
        self.model = model

    def set_training_params(
        self, n_training_epochs, n_batch_samples, learning_rate, weight_decay
    ):
        self.n_epochs = n_training_epochs
        self.num_batch_samples = n_batch_samples
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  # L2 Weight Regularization - Weight Decay

        self.loss_function = nn.MSELoss(reduction="mean")

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def NN_train(
        self, n_examples, train_input, train_target, n_CV, cv_input, cv_target
    ):

        self.n_examples = n_examples
        self.n_CV = n_CV

        MSE_cv_linear_batch = torch.empty([self.n_CV])
        self.MSE_cv_linear_epoch = torch.empty([self.n_epochs])
        self.MSE_cv_dB_epoch = torch.empty([self.n_epochs])

        MSE_train_linear_batch = torch.empty([self.num_batch_samples])
        self.MSE_train_linear_epoch = torch.empty([self.n_epochs])
        self.MSE_train_dB_epoch = torch.empty([self.n_epochs])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.n_epochs):

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            for j in range(0, self.n_CV):
                y_cv = cv_input[j, :, :]
                if isinstance(self.model, KalmanNet):
                    self.model.init_sequence(self.ss_model.m1x_0)
                else:
                    self.model.init_sequence(self.ss_model.m1x_0, self.ss_model.t_test)

                if isinstance(self, PipelineEKF):
                    dim = self.ss_model.t_test
                else:
                    dim = self.ss_model.t

                x_out_cv = torch.empty(self.ss_model.m, dim)
                for t in range(0, dim):
                    x_out_cv[:, t] = self.model(y_cv[:, t])

                # Compute Training Loss
                MSE_cv_linear_batch[j] = self.loss_function(
                    x_out_cv, cv_target[j, :, :]
                ).item()

            # Average
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

            if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.model_filename)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.num_batch_samples):
                n_e = random.randint(0, self.n_examples - 1)

                y_training = train_input[n_e, :, :]
                if isinstance(self, PipelineEKF):
                    self.model.init_sequence(self.ss_model.m1x_0, self.ss_model.t)
                else:
                    self.model.init_sequence(self.ss_model.m1x_0)

                x_out_training = torch.empty(self.ss_model.m, self.ss_model.t)
                for t in range(0, self.ss_model.t):
                    x_out_training[:, t] = self.model(y_training[:, t])

                # Compute Training Loss
                LOSS = self.loss_function(x_out_training, train_target[n_e, :, :])
                MSE_train_linear_batch[j] = LOSS.item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(
                self.MSE_train_linear_epoch[ti]
            )

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = (
                Batch_Optimizing_LOSS_sum / self.num_batch_samples
            )
            Batch_Optimizing_LOSS_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            ########################
            ### Training Summary ###
            ########################
            print(
                ti,
                "MSE Training :",
                self.MSE_train_dB_epoch[ti],
                "[dB]",
                "MSE Validation :",
                self.MSE_cv_dB_epoch[ti],
                "[dB]",
            )

            if ti > 1:
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print(
                    "diff MSE Training :",
                    d_train,
                    "[dB]",
                    "diff MSE Validation :",
                    d_cv,
                    "[dB]",
                )

            print(
                "Optimal idx:",
                self.MSE_cv_idx_opt,
                "Optimal :",
                self.MSE_cv_dB_opt,
                "[dB]",
            )

    def NNTest(self, n_test, test_input, test_target):

        self.n_test = n_test

        self.MSE_test_linear_arr = torch.empty([self.n_test])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction="mean")

        self.model = torch.load(self.model_filename)

        self.model.eval()

        torch.no_grad()

        if isinstance(self, PipelineEKF):
            x_out_array = torch.empty(
                self.n_test, self.ss_model.m, self.ss_model.t_test
            )
        else:
            x_out_array = torch.empty(self.n_test, self.ss_model.m, self.ss_model.t)

        start = time.time()

        for j in range(0, self.n_test):

            y_mdl_tst = test_input[j, :, :]

            if isinstance(self.model, KalmanNet):
                self.model.init_sequence(self.ss_model.m1x_0)
            else:
                self.model.init_sequence(self.ss_model.m1x_0, self.ss_model.t_test)

            if isinstance(self, PipelineEKF):
                dim = self.ss_model.t_test
            else:
                dim = self.ss_model.t
            x_out_test = torch.empty(self.ss_model.m, dim)

            for t in range(0, dim):
                x_out_test[:, t] = self.model(y_mdl_tst[:, t])

            self.MSE_test_linear_arr[j] = loss_fn(
                x_out_test, test_target[j, :, :]
            ).item()
            x_out_array[j, :, :] = x_out_test

        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_dB_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        self.MSE_test_dB_std = 10 * torch.log10(self.MSE_test_dB_std)

        # Print MSE Cross Validation
        str = self.model_name + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.model_name + "-" + "STD Test:"
        print(str, self.MSE_test_dB_std, "[dB]")
        # Print Run time
        print("Inference time:", t)

        return [
            self.MSE_test_linear_arr,
            self.MSE_test_linear_avg,
            self.MSE_test_dB_avg,
            x_out_array,  # x_out_test?
        ]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folder_name, self.model_name)

        self.Plot.NNPlot_epochs(
            self.n_epochs,
            MSE_KF_dB_avg,
            self.MSE_test_dB_avg,
            self.MSE_cv_dB_epoch,
            self.MSE_train_dB_epoch,
        )

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)


class PipelineEKF(PipelineKF):
    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folder_name, self.model_name)

        self.Plot.NNPlot_epochs(
            self.n_epochs,
            MSE_KF_dB_avg,
            self.MSE_test_dB_avg,
            self.MSE_cv_dB_epoch,
            self.MSE_train_dB_epoch,
        )

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)

import torch
from torch.utils.data import Dataset
import scipy.io
import os
import numpy as np

class NarrowHJI_Supervised(Dataset):
    def __init__(self, Hybrid_use, seed=0, rank=0, ):

        super().__init__()
        torch.manual_seed(0)
        self.hybrid_use = Hybrid_use

        current_dir = os.path.dirname(os.path.abspath(__file__))

        if not self.hybrid_use:
            # supervised neural operator
            data_path1 = current_dir + '/validation_scripts/train_data/data_train_narrowroad_1_1_2000.mat'
            data_path2 = current_dir + '/validation_scripts/train_data/data_train_narrowroad_1_5_2000.mat'
            data_path3 = current_dir + '/validation_scripts/train_data/data_train_narrowroad_5_1_2000.mat'
            data_path4 = current_dir + '/validation_scripts/train_data/data_train_narrowroad_5_5_2000.mat'
        else:
            # hybrid neural operator
            data_path1 = current_dir + '/validation_scripts/train_data/data_train_narrowroad_1_1_1000.mat'
            data_path2 = current_dir + '/validation_scripts/train_data/data_train_narrowroad_1_5_1000.mat'
            data_path3 = current_dir + '/validation_scripts/train_data/data_train_narrowroad_5_1_1000.mat'
            data_path4 = current_dir + '/validation_scripts/train_data/data_train_narrowroad_5_5_1000.mat'

        train_data1 = scipy.io.loadmat(data_path1)
        train_data2 = scipy.io.loadmat(data_path2)
        train_data3 = scipy.io.loadmat(data_path3)
        train_data4 = scipy.io.loadmat(data_path4)
        self.train_data1 = train_data1
        self.train_data2 = train_data2
        self.train_data3 = train_data3
        self.train_data4 = train_data4

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = current_dir + '/validation_scripts/train_data/narrowroad_param_fun_50.mat'

        self.input_fun = scipy.io.loadmat(data_path)

        # Set the seed
        torch.manual_seed(seed)

        self.rank = rank

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        self.t_train1 = torch.tensor(self.train_data1['t'], dtype=torch.float32).flip(1)
        self.X_train1 = torch.tensor(self.train_data1['X'], dtype=torch.float32)
        self.A_train1 = torch.tensor(self.train_data1['A'], dtype=torch.float32)
        self.V_train1 = torch.tensor(self.train_data1['V'], dtype=torch.float32)
        self.t_train2 = torch.tensor(self.train_data2['t'], dtype=torch.float32).flip(1)
        self.X_train2 = torch.tensor(self.train_data2['X'], dtype=torch.float32)
        self.A_train2 = torch.tensor(self.train_data2['A'], dtype=torch.float32)
        self.V_train2 = torch.tensor(self.train_data2['V'], dtype=torch.float32)
        self.t_train3 = torch.tensor(self.train_data3['t'], dtype=torch.float32).flip(1)
        self.X_train3 = torch.tensor(self.train_data3['X'], dtype=torch.float32)
        self.A_train3 = torch.tensor(self.train_data3['A'], dtype=torch.float32)
        self.V_train3 = torch.tensor(self.train_data3['V'], dtype=torch.float32)
        self.t_train4 = torch.tensor(self.train_data4['t'], dtype=torch.float32).flip(1)
        self.X_train4 = torch.tensor(self.train_data4['X'], dtype=torch.float32)
        self.A_train4 = torch.tensor(self.train_data4['A'], dtype=torch.float32)
        self.V_train4 = torch.tensor(self.train_data4['V'], dtype=torch.float32)

        self.lb = torch.tensor([[15], [31], [-0.2], [18], [15], [31], [-0.2], [18]], dtype=torch.float32)
        self.ub = torch.tensor([[95], [39], [0.2], [29], [95], [39], [0.2], [29]], dtype=torch.float32)

        self.X_train1 = 2.0 * (self.X_train1 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train2 = 2.0 * (self.X_train2 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train3 = 2.0 * (self.X_train3 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train4 = 2.0 * (self.X_train4 - self.lb) / (self.ub - self.lb) - 1.

        self.X_train = torch.cat((self.X_train1, self.X_train2,
                                  self.X_train3, self.X_train4), dim=1)
        self.t_train = torch.cat((self.t_train1, self.t_train2,
                                  self.t_train3, self.t_train4), dim=1)

        coords_1 = self.X_train.T
        coords_2 = torch.cat((coords_1[:, 4:], coords_1[:, :4]), dim=1)

        coords_1 = torch.cat((self.t_train.T, coords_1), dim=1)
        coords_2 = torch.cat((self.t_train.T, coords_2), dim=1)

        # set up ground truth for values and costates
        groundtruth_values1 = torch.cat((self.V_train1[0, :].reshape(-1, 1), self.V_train2[0, :].reshape(-1, 1),
                                         self.V_train3[0, :].reshape(-1, 1), self.V_train4[0, :].reshape(-1, 1)), dim=0)
        groundtruth_values2 = torch.cat((self.V_train1[1, :].reshape(-1, 1), self.V_train2[1, :].reshape(-1, 1),
                                         self.V_train3[1, :].reshape(-1, 1), self.V_train4[1, :].reshape(-1, 1)), dim=0)
        groundtruth_values = torch.cat((groundtruth_values1, groundtruth_values2), dim=0)

        groundtruth_costates1 = torch.cat((self.A_train1[:8, :].T, self.A_train2[:8, :].T,
                                           self.A_train3[:8, :].T, self.A_train4[:8, :].T), dim=0)
        groundtruth_costates2 = torch.cat((self.A_train1[8:, :].T, self.A_train2[8:, :].T,
                                           self.A_train3[8:, :].T, self.A_train4[8:, :].T), dim=0)
        groundtruth_costates = torch.cat((groundtruth_costates1, groundtruth_costates2), dim=0)

        theta_11 = torch.tensor(self.input_fun['theta_11'], dtype=torch.float32).clone()
        theta_15 = torch.tensor(self.input_fun['theta_15'], dtype=torch.float32).clone()
        theta_51 = torch.tensor(self.input_fun['theta_51'], dtype=torch.float32).clone()
        theta_55 = torch.tensor(self.input_fun['theta_55'], dtype=torch.float32).clone()

        num_input = coords_1.shape[0] // 4

        theta_11_hji = theta_11.unsqueeze(0).repeat(num_input, 1, 1).flatten(start_dim=1)
        theta_15_hji = theta_15.unsqueeze(0).repeat(num_input, 1, 1).flatten(start_dim=1)
        theta_51_hji = theta_51.unsqueeze(0).repeat(num_input, 1, 1).flatten(start_dim=1)
        theta_55_hji = theta_55.unsqueeze(0).repeat(num_input, 1, 1).flatten(start_dim=1)

        input_fun = torch.cat((theta_11_hji, theta_15_hji, theta_51_hji, theta_55_hji,
                               theta_11_hji, theta_51_hji, theta_15_hji, theta_55_hji), dim=0)

        coords = torch.cat((coords_1, coords_2), dim=0)
        return {'coords': coords, 'input_fun': input_fun}, \
               {'groundtruth_values': groundtruth_values,
                'groundtruth_costates': groundtruth_costates}


class NarrowHJI_Hybrid(Dataset):
    def __init__(self, numpoints, tMin=0.0, tMax=3, counter_start=0, counter_end=100e3, num_src_samples=1000,
                 seed=0):

        super().__init__()
        torch.manual_seed(0)

        self.numpoints = numpoints
        self.num_states = 8

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.full_count = counter_end
        self.alpha = 1e-6

        # Set the seed
        torch.manual_seed(seed)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path1 = current_dir + '/validation_scripts/train_data/data_train_narrowroad_1_1_1000.mat'
        data_path2 = current_dir + '/validation_scripts/train_data/data_train_narrowroad_1_5_1000.mat'
        data_path3 = current_dir + '/validation_scripts/train_data/data_train_narrowroad_5_1_1000.mat'
        data_path4 = current_dir + '/validation_scripts/train_data/data_train_narrowroad_5_5_1000.mat'

        train_data1 = scipy.io.loadmat(data_path1)
        train_data2 = scipy.io.loadmat(data_path2)
        train_data3 = scipy.io.loadmat(data_path3)
        train_data4 = scipy.io.loadmat(data_path4)
        self.train_data1 = train_data1
        self.train_data2 = train_data2
        self.train_data3 = train_data3
        self.train_data4 = train_data4

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = current_dir + '/validation_scripts/train_data/narrowroad_param_fun_50.mat'

        self.input_fun = scipy.io.loadmat(data_path)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # supervised learning data
        self.t_train1 = torch.tensor(self.train_data1['t'], dtype=torch.float32).flip(1)
        self.X_train1 = torch.tensor(self.train_data1['X'], dtype=torch.float32)
        self.A_train1 = torch.tensor(self.train_data1['A'], dtype=torch.float32)
        self.V_train1 = torch.tensor(self.train_data1['V'], dtype=torch.float32)
        self.t_train2 = torch.tensor(self.train_data2['t'], dtype=torch.float32).flip(1)
        self.X_train2 = torch.tensor(self.train_data2['X'], dtype=torch.float32)
        self.A_train2 = torch.tensor(self.train_data2['A'], dtype=torch.float32)
        self.V_train2 = torch.tensor(self.train_data2['V'], dtype=torch.float32)
        self.t_train3 = torch.tensor(self.train_data3['t'], dtype=torch.float32).flip(1)
        self.X_train3 = torch.tensor(self.train_data3['X'], dtype=torch.float32)
        self.A_train3 = torch.tensor(self.train_data3['A'], dtype=torch.float32)
        self.V_train3 = torch.tensor(self.train_data3['V'], dtype=torch.float32)
        self.t_train4 = torch.tensor(self.train_data4['t'], dtype=torch.float32).flip(1)
        self.X_train4 = torch.tensor(self.train_data4['X'], dtype=torch.float32)
        self.A_train4 = torch.tensor(self.train_data4['A'], dtype=torch.float32)
        self.V_train4 = torch.tensor(self.train_data4['V'], dtype=torch.float32)

        self.lb = torch.tensor([[15], [31], [-0.2], [18], [15], [31], [-0.2], [18]], dtype=torch.float32)
        self.ub = torch.tensor([[95], [39], [0.2], [29], [95], [39], [0.2], [29]], dtype=torch.float32)

        self.X_train1 = 2.0 * (self.X_train1 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train2 = 2.0 * (self.X_train2 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train3 = 2.0 * (self.X_train3 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train4 = 2.0 * (self.X_train4 - self.lb) / (self.ub - self.lb) - 1.

        self.X_train = torch.cat((self.X_train1, self.X_train2,
                                  self.X_train3, self.X_train4), dim=1)
        self.t_train = torch.cat((self.t_train1, self.t_train2,
                                  self.t_train3, self.t_train4), dim=1)
        coords_1 = self.X_train.T
        coords_2 = torch.cat((coords_1[:, 4:], coords_1[:, :4]), dim=1)

        coords_1_supervised = torch.cat((self.t_train.T, coords_1), dim=1)
        coords_2_supervised = torch.cat((self.t_train.T, coords_2), dim=1)

        # set up ground truth for values and costates
        groundtruth_values1 = torch.cat((self.V_train1[0, :].reshape(-1, 1), self.V_train2[0, :].reshape(-1, 1),
                                         self.V_train3[0, :].reshape(-1, 1), self.V_train4[0, :].reshape(-1, 1)), dim=0)
        groundtruth_values2 = torch.cat((self.V_train1[1, :].reshape(-1, 1), self.V_train2[1, :].reshape(-1, 1),
                                         self.V_train3[1, :].reshape(-1, 1), self.V_train4[1, :].reshape(-1, 1)), dim=0)
        groundtruth_values = torch.cat((groundtruth_values1, groundtruth_values2), dim=0)

        groundtruth_costates1 = torch.cat((self.A_train1[:8, :].T, self.A_train2[:8, :].T,
                                           self.A_train3[:8, :].T, self.A_train4[:8, :].T), dim=0)
        groundtruth_costates2 = torch.cat((self.A_train1[8:, :].T, self.A_train2[8:, :].T,
                                           self.A_train3[8:, :].T, self.A_train4[8:, :].T), dim=0)
        groundtruth_costates = torch.cat((groundtruth_costates1, groundtruth_costates2), dim=0)

        # HJI data(sample entire state space)
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates for both agents
        coords_1 = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        coords_2 = torch.cat((coords_1[:, 4:], coords_1[:, :4]), dim=1)

        # slowly grow time values from start time
        # this currently assumes start_time = 0 and max time value is tMax
        time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (
                self.counter / self.full_count))

        coords_1_hji = torch.cat((time, coords_1), dim=1)
        coords_2_hji = torch.cat((time, coords_2), dim=1)

        # make sure we always have training samples at the initial time
        coords_1_hji[-self.N_src_samples:, 0] = start_time
        coords_2_hji[-self.N_src_samples:, 0] = start_time

        coords_1_hji = coords_1_hji.repeat(4, 1)
        coords_2_hji = coords_2_hji.repeat(4, 1)

        # set up boundary condition: V(T) = alpha*X(T) - (V(T) - V(0))^2
        boundary_values_1 = self.alpha * ((coords_1_hji[:, 1:2] + 1) * (95 - 15) / 2 + 15) - \
                            ((coords_1_hji[:, 4:5] + 1) * (29 - 18) / 2 + 18 - 18) ** 2 - \
                            ((coords_1_hji[:, 2:3] + 1) * (39 - 31) / 2 + 31 - 35) ** 2
        boundary_values_2 = self.alpha * ((coords_2_hji[:, 1:2] + 1) * (95 - 15) / 2 + 15) - \
                            ((coords_2_hji[:, 4:5] + 1) * (29 - 18) / 2 + 18 - 18) ** 2 - \
                            ((coords_2_hji[:, 2:3] + 1) * (39 - 31) / 2 + 31 - 35) ** 2
        boundary_values = torch.cat((boundary_values_1, boundary_values_2), dim=0)

        dirichlet_mask = (coords_1_hji[:, 0, None] == start_time)

        if self.counter < self.full_count:
            self.counter += 1

        coords_1 = torch.cat((coords_1_supervised, coords_1_hji), dim=0)
        coords_2 = torch.cat((coords_2_supervised, coords_2_hji), dim=0)

        coords = torch.cat((coords_1, coords_2), dim=0)

        theta_11 = torch.tensor(self.input_fun['theta_11'], dtype=torch.float32).clone()
        theta_15 = torch.tensor(self.input_fun['theta_15'], dtype=torch.float32).clone()
        theta_51 = torch.tensor(self.input_fun['theta_51'], dtype=torch.float32).clone()
        theta_55 = torch.tensor(self.input_fun['theta_55'], dtype=torch.float32).clone()

        num_sl = coords_1_supervised.shape[0] // 4
        num_hl = coords_1_hji.shape[0] // 4

        theta_11_sl = theta_11.unsqueeze(0).repeat(num_sl, 1, 1).flatten(start_dim=1)
        theta_15_sl = theta_15.unsqueeze(0).repeat(num_sl, 1, 1).flatten(start_dim=1)
        theta_51_sl = theta_51.unsqueeze(0).repeat(num_sl, 1, 1).flatten(start_dim=1)
        theta_55_sl = theta_55.unsqueeze(0).repeat(num_sl, 1, 1).flatten(start_dim=1)

        theta_11_hl = theta_11.unsqueeze(0).repeat(num_hl, 1, 1).flatten(start_dim=1)
        theta_15_hl = theta_15.unsqueeze(0).repeat(num_hl, 1, 1).flatten(start_dim=1)
        theta_51_hl = theta_51.unsqueeze(0).repeat(num_hl, 1, 1).flatten(start_dim=1)
        theta_55_hl = theta_55.unsqueeze(0).repeat(num_hl, 1, 1).flatten(start_dim=1)

        input_fun = torch.cat((theta_11_sl, theta_15_sl, theta_51_sl, theta_55_sl,
                               theta_11_hl, theta_15_hl, theta_51_hl, theta_55_hl,
                               theta_11_sl, theta_51_sl, theta_15_sl, theta_55_sl,
                               theta_11_hl, theta_51_hl, theta_15_hl, theta_55_hl), dim=0)

        return {'coords': coords, 'input_fun': input_fun}, \
               {'groundtruth_values': groundtruth_values, 'groundtruth_costates': groundtruth_costates,
                'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}

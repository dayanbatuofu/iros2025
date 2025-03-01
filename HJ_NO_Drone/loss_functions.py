import torch
import diff_operators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_drone_HJI_supervised(dataset, Weight, alpha):
    def interaction_hji(model_output, gt):
        weight1, weight2 = Weight
        groundtruth_values = gt['groundtruth_values']
        groundtruth_costates = gt['groundtruth_costates']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2

        y1 = model_output['model_out'][:, :cut_index]   # (meta_batch_size, num_points, 1); agent 1's value
        y2 = model_output['model_out'][:, cut_index:]   # (meta_batch_size, num_points, 1); agent 2's value

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((15.5 - 0) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((15.5 - 0) / 2)  # lambda_11
        lam11_3 = dvdx_1[:, 2:3] / ((2.5 - (-2.2)) / 2)  # lambda_11
        lam11_4 = dvdx_1[:, 3:4] / ((4.5 - 0.3) / 2)  # lambda_11
        lam11_5 = dvdx_1[:, 4:5] / ((4.5 - 0.3) / 2)  # lambda_11
        lam11_6 = dvdx_1[:, 5:6] / ((2.2 - (-2)) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 6:7] / ((15.5 - 0) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 7:8] / ((15.5 - 0) / 2)  # lambda_12
        lam12_3 = dvdx_1[:, 8:9] / ((2.5 - (-2.2)) / 2)  # lambda_12
        lam12_4 = dvdx_1[:, 9:10] / ((4.5 - 0.3) / 2)  # lambda_12
        lam12_5 = dvdx_1[:, 10:11] / ((4.5 - 0.3) / 2)  # lambda_12
        lam12_6 = dvdx_1[:, 11:12] / ((2.2 - (-2)) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 6:7] / ((15.5 - 0) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 7:8] / ((15.5 - 0) / 2)  # lambda_21
        lam21_3 = dvdx_2[:, 8:9] / ((2.5 - (-2.2)) / 2)  # lambda_21
        lam21_4 = dvdx_2[:, 9:10] / ((4.5 - 0.3) / 2)  # lambda_21
        lam21_5 = dvdx_2[:, 10:11] / ((4.5 - 0.3) / 2)  # lambda_21
        lam21_6 = dvdx_2[:, 11:12] / ((2.2 - (-2)) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((15.5 - 0) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((15.5 - 0) / 2)  # lambda_22
        lam22_3 = dvdx_2[:, 2:3] / ((2.5 - (-2.2)) / 2)  # lambda_22
        lam22_4 = dvdx_2[:, 3:4] / ((4.5 - 0.3) / 2)  # lambda_22
        lam22_5 = dvdx_2[:, 4:5] / ((4.5 - 0.3) / 2)  # lambda_22
        lam22_6 = dvdx_2[:, 5:6] / ((2.2 - (-2)) / 2)  # lambda_22

        # supervised learning for values
        value1_difference = y1 - alpha * groundtruth_values[:, :y1.shape[1]]
        value2_difference = y2 - alpha * groundtruth_values[:, y2.shape[1]:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1, lam11_2, lam11_3, lam11_4, lam11_5, lam11_6,
                                         lam12_1, lam12_2, lam12_3, lam12_4, lam12_5, lam12_6), dim=1)
        costate2_prediction = torch.cat((lam21_1, lam21_2, lam21_3, lam21_4, lam21_5, lam21_6,
                                         lam22_1, lam22_2, lam22_3, lam22_4, lam22_5, lam22_6), dim=1)
        costate1_difference = costate1_prediction - alpha * groundtruth_costates[:, :y1.shape[1]].squeeze()
        costate2_difference = costate2_prediction - alpha * groundtruth_costates[:, y2.shape[1]:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # A factor of (weight1, weight2) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / weight1,
                'costates_difference': torch.abs(costates_difference).sum() / weight2}
    return interaction_hji


def initialize_drone_HJI_hyrid(dataset, Weight, alpha):
    def interaction_hji(model_output, gt):
        weight1, weight2, weight3, weight4 = Weight
        groundtruth_values = gt['groundtruth_values']
        groundtruth_costates = gt['groundtruth_costates']
        source_boundary_values = gt['source_boundary_values']
        dirichlet_mask = gt['dirichlet_mask']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2
        supervised_index = groundtruth_values.shape[1] // 2
        hji_index = source_boundary_values.shape[1] // 2
        num_sl = supervised_index // 4
        num_hl = hji_index // 4

        y1 = y[:, :cut_index]   # (meta_batch_size, num_points, 1); agent 1's value
        y2 = y[:, cut_index:]   # (meta_batch_size, num_points, 1); agent 2's value
        x1 = x[:, :cut_index]
        x2 = x[:, cut_index:]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((15.5 - 0) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((15.5 - 0) / 2)  # lambda_11
        lam11_3 = dvdx_1[:, 2:3] / ((2.5 - (-2.2)) / 2)  # lambda_11
        lam11_4 = dvdx_1[:, 3:4] / ((4.5 - 0.3) / 2)  # lambda_11
        lam11_5 = dvdx_1[:, 4:5] / ((4.5 - 0.3) / 2)  # lambda_11
        lam11_6 = dvdx_1[:, 5:6] / ((2.2 - (-2)) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 6:7] / ((15.5 - 0) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 7:8] / ((15.5 - 0) / 2)  # lambda_12
        lam12_3 = dvdx_1[:, 8:9] / ((2.5 - (-2.2)) / 2)  # lambda_12
        lam12_4 = dvdx_1[:, 9:10] / ((4.5 - 0.3) / 2)  # lambda_12
        lam12_5 = dvdx_1[:, 10:11] / ((4.5 - 0.3) / 2)  # lambda_12
        lam12_6 = dvdx_1[:, 11:12] / ((2.2 - (-2)) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 6:7] / ((15.5 - 0) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 7:8] / ((15.5 - 0) / 2)  # lambda_21
        lam21_3 = dvdx_2[:, 8:9] / ((2.5 - (-2.2)) / 2)  # lambda_21
        lam21_4 = dvdx_2[:, 9:10] / ((4.5 - 0.3) / 2)  # lambda_21
        lam21_5 = dvdx_2[:, 10:11] / ((4.5 - 0.3) / 2)  # lambda_21
        lam21_6 = dvdx_2[:, 11:12] / ((2.2 - (-2)) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((15.5 - 0) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((15.5 - 0) / 2)  # lambda_22
        lam22_3 = dvdx_2[:, 2:3] / ((2.5 - (-2.2)) / 2)  # lambda_22
        lam22_4 = dvdx_2[:, 3:4] / ((4.5 - 0.3) / 2)  # lambda_22
        lam22_5 = dvdx_2[:, 4:5] / ((4.5 - 0.3) / 2)  # lambda_22
        lam22_6 = dvdx_2[:, 5:6] / ((2.2 - (-2)) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([5.], dtype=torch.float32).to(device)  # road length at y direction
        R2 = torch.tensor([5.], dtype=torch.float32).to(device)  # road length at x direction
        threshold_11 = torch.tensor([0.75], dtype=torch.float32).to(device)  # collision penalty threshold
        threshold_15 = torch.tensor([1.15], dtype=torch.float32).to(device)  # collision penalty threshold
        threshold_51 = torch.tensor([1.15], dtype=torch.float32).to(device)  # collision penalty threshold
        threshold_55 = torch.tensor([1.75], dtype=torch.float32).to(device)  # collision penalty threshold
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio
        gravity = torch.tensor([9.81], dtype=torch.float32).to(device)  # gravity acceleration

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action
        # H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        theta1 = torch.atan(lam11_4 * gravity / (200 * alpha))
        phi1 = torch.atan(-lam11_5 * gravity / (200 * alpha))
        thrust1 = lam11_6 / (2 * alpha) + gravity

        # Agent 2's action
        theta2 = torch.atan(lam22_4 * gravity / (200 * alpha))
        phi2 = torch.atan(-lam22_5 * gravity / (200 * alpha))
        thrust2 = lam22_6 / (2 * alpha) + gravity

        # set up bounds for u1 and u2
        max_acc_theta = torch.tensor([0.05], dtype=torch.float32).to(device)
        min_acc_theta = torch.tensor([-0.05], dtype=torch.float32).to(device)
        max_acc_phi = torch.tensor([0.05], dtype=torch.float32).to(device)
        min_acc_phi = torch.tensor([-0.05], dtype=torch.float32).to(device)
        max_acc_thrust = torch.tensor([11.81], dtype=torch.float32).to(device)
        min_acc_thrust = torch.tensor([7.81], dtype=torch.float32).to(device)

        theta1[torch.where(theta1 > max_acc_theta)] = max_acc_theta
        theta1[torch.where(theta1 < min_acc_theta)] = min_acc_theta
        theta2[torch.where(theta2 > max_acc_theta)] = max_acc_theta
        theta2[torch.where(theta2 < min_acc_theta)] = min_acc_theta

        phi1[torch.where(phi1 > max_acc_phi)] = max_acc_phi
        phi1[torch.where(phi1 < min_acc_phi)] = min_acc_phi
        phi2[torch.where(phi2 > max_acc_phi)] = max_acc_phi
        phi2[torch.where(phi2 < min_acc_phi)] = min_acc_phi

        thrust1[torch.where(thrust1 > max_acc_thrust)] = max_acc_thrust
        thrust1[torch.where(thrust1 < min_acc_thrust)] = min_acc_thrust
        thrust2[torch.where(thrust2 > max_acc_thrust)] = max_acc_thrust
        thrust2[torch.where(thrust2 < min_acc_thrust)] = min_acc_thrust

        # unnormalize the state for agent 1
        dx_11_11_sl = (x1[:, :num_sl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_11_11_sl = (x1[:, :num_sl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_11_11_sl = (x1[:, :num_sl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_11_15_sl = (x1[:, num_sl:2*num_sl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_11_15_sl = (x1[:, num_sl:2*num_sl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_11_15_sl = (x1[:, num_sl:2*num_sl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_11_51_sl = (x1[:, 2*num_sl:3*num_sl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_11_51_sl = (x1[:, 2*num_sl:3*num_sl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_11_51_sl = (x1[:, 2*num_sl:3*num_sl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_11_55_sl = (x1[:, 3*num_sl:4*num_sl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_11_55_sl = (x1[:, 3*num_sl:4*num_sl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_11_55_sl = (x1[:, 3*num_sl:4*num_sl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_11_11_hl = (x1[:, 4*num_sl:4*num_sl+num_hl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_11_11_hl = (x1[:, 4*num_sl:4*num_sl+num_hl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_11_11_hl = (x1[:, 4*num_sl:4*num_sl+num_hl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_11_15_hl = (x1[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_11_15_hl = (x1[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_11_15_hl = (x1[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_11_51_hl = (x1[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_11_51_hl = (x1[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_11_51_hl = (x1[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_11_55_hl = (x1[:, 4*num_sl+3*num_hl:, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_11_55_hl = (x1[:, 4*num_sl+3*num_hl:, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_11_55_hl = (x1[:, 4*num_sl+3*num_hl:, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        vx_11 = (x1[:, :, 4:5] + 1) * (4.5 - 0.3) / 2 + 0.3
        vy_11 = (x1[:, :, 5:6] + 1) * (4.5 - 0.3) / 2 + 0.3
        vz_11 = (x1[:, :, 6:7] + 1) * (2.2 - (-2)) / 2 + (-2)

        # unnormalize the state for agent 2
        dx_12_11_sl = (x1[:, :num_sl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_12_11_sl = (x1[:, :num_sl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_12_11_sl = (x1[:, :num_sl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_12_15_sl = (x1[:, num_sl:2*num_sl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_12_15_sl = (x1[:, num_sl:2*num_sl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_12_15_sl = (x1[:, num_sl:2*num_sl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_12_51_sl = (x1[:, 2*num_sl:3*num_sl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_12_51_sl = (x1[:, 2*num_sl:3*num_sl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_12_51_sl = (x1[:, 2*num_sl:3*num_sl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_12_55_sl = (x1[:, 3*num_sl:4*num_sl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_12_55_sl = (x1[:, 3*num_sl:4*num_sl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_12_55_sl = (x1[:, 3*num_sl:4*num_sl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_12_11_hl = (x1[:, 4*num_sl:4*num_sl+num_hl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_12_11_hl = (x1[:, 4*num_sl:4*num_sl+num_hl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_12_11_hl = (x1[:, 4*num_sl:4*num_sl+num_hl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_12_15_hl = (x1[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_12_15_hl = (x1[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_12_15_hl = (x1[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_12_51_hl = (x1[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_12_51_hl = (x1[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_12_51_hl = (x1[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_12_55_hl = (x1[:, 4*num_sl+3*num_hl:, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_12_55_hl = (x1[:, 4*num_sl+3*num_hl:, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_12_55_hl = (x1[:, 4*num_sl+3*num_hl:, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        vx_12 = (x1[:, :, 10:11] + 1) * (4.5 - 0.3) / 2 + 0.3
        vy_12 = (x1[:, :, 11:12] + 1) * (4.5 - 0.3) / 2 + 0.3
        vz_12 = (x1[:, :, 12:13] + 1) * (2.2 - (-2)) / 2 + (-2)

        # calculate the collision area lower and upper bounds
        dist_diff1_11_sl = (-(torch.sqrt(((R1 - dx_12_11_sl) - dx_11_11_sl) ** 2 + ((R2 - dy_12_11_sl) - dy_11_11_sl) ** 2 +
                           (dz_12_11_sl - dz_11_11_sl) ** 2) - threshold_11) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_15_sl = (-(torch.sqrt(((R1 - dx_12_15_sl) - dx_11_15_sl) ** 2 + ((R2 - dy_12_15_sl) - dy_11_15_sl) ** 2 +
                           (dz_12_15_sl - dz_11_15_sl) ** 2) - threshold_15) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_51_sl = (-(torch.sqrt(((R1 - dx_12_51_sl) - dx_11_51_sl) ** 2 + ((R2 - dy_12_51_sl) - dy_11_51_sl) ** 2 +
                           (dz_12_51_sl - dz_11_51_sl) ** 2) - threshold_51) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_55_sl = (-(torch.sqrt(((R1 - dx_12_55_sl) - dx_11_55_sl) ** 2 + ((R2 - dy_12_55_sl) - dy_11_55_sl) ** 2 +
                           (dz_12_55_sl - dz_11_55_sl) ** 2) - threshold_55) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_11_hl = (-(torch.sqrt(((R1 - dx_12_11_hl) - dx_11_11_hl) ** 2 + ((R2 - dy_12_11_hl) - dy_11_11_hl) ** 2 +
                           (dz_12_11_hl - dz_11_11_hl) ** 2) - threshold_11) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_15_hl = (-(torch.sqrt(((R1 - dx_12_15_hl) - dx_11_15_hl) ** 2 + ((R2 - dy_12_15_hl) - dy_11_15_hl) ** 2 +
                           (dz_12_15_hl - dz_11_15_hl) ** 2) - threshold_15) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_51_hl = (-(torch.sqrt(((R1 - dx_12_51_hl) - dx_11_51_hl) ** 2 + ((R2 - dy_12_51_hl) - dy_11_51_hl) ** 2 +
                           (dz_12_51_hl - dz_11_51_hl) ** 2) - threshold_51) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_55_hl = (-(torch.sqrt(((R1 - dx_12_55_hl) - dx_11_55_hl) ** 2 + ((R2 - dy_12_55_hl) - dy_11_55_hl) ** 2 +
                           (dz_12_55_hl - dz_11_55_hl) ** 2) - threshold_55) * 5).squeeze().reshape(-1, 1).to(device)
        sigmoid1_11_sl = torch.sigmoid(dist_diff1_11_sl)
        sigmoid1_15_sl = torch.sigmoid(dist_diff1_15_sl)
        sigmoid1_51_sl = torch.sigmoid(dist_diff1_51_sl)
        sigmoid1_55_sl = torch.sigmoid(dist_diff1_55_sl)
        sigmoid1_11_hl = torch.sigmoid(dist_diff1_11_hl)
        sigmoid1_15_hl = torch.sigmoid(dist_diff1_15_hl)
        sigmoid1_51_hl = torch.sigmoid(dist_diff1_51_hl)
        sigmoid1_55_hl = torch.sigmoid(dist_diff1_55_hl)

        loss_instant1_11_sl = beta*sigmoid1_11_sl
        loss_instant1_15_sl = beta*sigmoid1_15_sl
        loss_instant1_51_sl = beta*sigmoid1_51_sl
        loss_instant1_55_sl = beta*sigmoid1_55_sl
        loss_instant1_11_hl = beta*sigmoid1_11_hl
        loss_instant1_15_hl = beta*sigmoid1_15_hl
        loss_instant1_51_hl = beta*sigmoid1_51_hl
        loss_instant1_55_hl = beta*sigmoid1_55_hl

        # unnormalize the state for agent 1
        dx_21_11_sl = (x2[:, :num_sl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_21_11_sl = (x2[:, :num_sl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_21_11_sl = (x2[:, :num_sl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_21_15_sl = (x2[:, num_sl:2*num_sl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_21_15_sl = (x2[:, num_sl:2*num_sl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_21_15_sl = (x2[:, num_sl:2*num_sl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_21_51_sl = (x2[:, 2*num_sl:3*num_sl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_21_51_sl = (x2[:, 2*num_sl:3*num_sl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_21_51_sl = (x2[:, 2*num_sl:3*num_sl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_21_55_sl = (x2[:, 3*num_sl:4*num_sl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_21_55_sl = (x2[:, 3*num_sl:4*num_sl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_21_55_sl = (x2[:, 3*num_sl:4*num_sl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_21_11_hl = (x2[:, 4*num_sl:4*num_sl+num_hl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_21_11_hl = (x2[:, 4*num_sl:4*num_sl+num_hl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_21_11_hl = (x2[:, 4*num_sl:4*num_sl+num_hl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_21_15_hl = (x2[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_21_15_hl = (x2[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_21_15_hl = (x2[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_21_51_hl = (x2[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_21_51_hl = (x2[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_21_51_hl = (x2[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_21_55_hl = (x2[:, 4*num_sl+3*num_hl:, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_21_55_hl = (x2[:, 4*num_sl+3*num_hl:, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_21_55_hl = (x2[:, 4*num_sl+3*num_hl:, 9:10] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        vx_21 = (x2[:, :, 10:11] + 1) * (4.5 - 0.3) / 2 + 0.3
        vy_21 = (x2[:, :, 11:12] + 1) * (4.5 - 0.3) / 2 + 0.3
        vz_21 = (x2[:, :, 12:13] + 1) * (2.2 - (-2)) / 2 + (-2)

        # unnormalize the state for agent 2
        dx_22_11_sl = (x2[:, :num_sl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_22_11_sl = (x2[:, :num_sl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_22_11_sl = (x2[:, :num_sl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_22_15_sl = (x2[:, num_sl:2*num_sl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_22_15_sl = (x2[:, num_sl:2*num_sl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_22_15_sl = (x2[:, num_sl:2*num_sl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_22_51_sl = (x2[:, 2*num_sl:3*num_sl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_22_51_sl = (x2[:, 2*num_sl:3*num_sl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_22_51_sl = (x2[:, 2*num_sl:3*num_sl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_22_55_sl = (x2[:, 3*num_sl:4*num_sl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_22_55_sl = (x2[:, 3*num_sl:4*num_sl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_22_55_sl = (x2[:, 3*num_sl:4*num_sl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_22_11_hl = (x2[:, 4*num_sl:4*num_sl+num_hl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_22_11_hl = (x2[:, 4*num_sl:4*num_sl+num_hl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_22_11_hl = (x2[:, 4*num_sl:4*num_sl+num_hl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_22_15_hl = (x2[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_22_15_hl = (x2[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_22_15_hl = (x2[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_22_51_hl = (x2[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_22_51_hl = (x2[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_22_51_hl = (x2[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        dx_22_55_hl = (x2[:, 4*num_sl+3*num_hl:, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_22_55_hl = (x2[:, 4*num_sl+3*num_hl:, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_22_55_hl = (x2[:, 4*num_sl+3*num_hl:, 3:4] + 1) * (2.5 - (-2.2)) / 2 + (-2.2)

        vx_22 = (x2[:, :, 4:5] + 1) * (4.5 - 0.3) / 2 + 0.3
        vy_22 = (x2[:, :, 5:6] + 1) * (4.5 - 0.3) / 2 + 0.3
        vz_22 = (x2[:, :, 6:7] + 1) * (2.2 - (-2)) / 2 + (-2)

        # calculate the collision area lower and upper bounds
        dist_diff2_11_sl = (-(torch.sqrt(((R1 - dx_22_11_sl) - dx_21_11_sl) ** 2 + ((R2 - dy_22_11_sl) - dy_21_11_sl) ** 2 +
                           (dz_22_11_sl - dz_21_11_sl) ** 2) - threshold_11) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_15_sl = (-(torch.sqrt(((R1 - dx_22_15_sl) - dx_21_15_sl) ** 2 + ((R2 - dy_22_15_sl) - dy_21_15_sl) ** 2 +
                           (dz_22_15_sl - dz_21_15_sl) ** 2) - threshold_15) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_51_sl = (-(torch.sqrt(((R1 - dx_22_51_sl) - dx_21_51_sl) ** 2 + ((R2 - dy_22_51_sl) - dy_21_51_sl) ** 2 +
                           (dz_22_51_sl - dz_21_51_sl) ** 2) - threshold_51) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_55_sl = (-(torch.sqrt(((R1 - dx_22_55_sl) - dx_21_55_sl) ** 2 + ((R2 - dy_22_55_sl) - dy_21_55_sl) ** 2 +
                           (dz_22_55_sl - dz_21_55_sl) ** 2) - threshold_55) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_11_hl = (-(torch.sqrt(((R1 - dx_22_11_hl) - dx_21_11_hl) ** 2 + ((R2 - dy_22_11_hl) - dy_21_11_hl) ** 2 +
                           (dz_22_11_hl - dz_21_11_hl) ** 2) - threshold_11) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_15_hl = (-(torch.sqrt(((R1 - dx_22_15_hl) - dx_21_15_hl) ** 2 + ((R2 - dy_22_15_hl) - dy_21_15_hl) ** 2 +
                           (dz_22_15_hl - dz_21_15_hl) ** 2) - threshold_15) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_51_hl = (-(torch.sqrt(((R1 - dx_22_51_hl) - dx_21_51_hl) ** 2 + ((R2 - dy_22_51_hl) - dy_21_51_hl) ** 2 +
                           (dz_22_51_hl - dz_21_51_hl) ** 2) - threshold_51) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_55_hl = (-(torch.sqrt(((R1 - dx_22_55_hl) - dx_21_55_hl) ** 2 + ((R2 - dy_22_55_hl) - dy_21_55_hl) ** 2 +
                           (dz_22_55_hl - dz_21_55_hl) ** 2) - threshold_55) * 5).squeeze().reshape(-1, 1).to(device)
        sigmoid2_11_sl = torch.sigmoid(dist_diff2_11_sl)
        sigmoid2_15_sl = torch.sigmoid(dist_diff2_15_sl)
        sigmoid2_51_sl = torch.sigmoid(dist_diff2_51_sl)
        sigmoid2_55_sl = torch.sigmoid(dist_diff2_55_sl)
        sigmoid2_11_hl = torch.sigmoid(dist_diff2_11_hl)
        sigmoid2_15_hl = torch.sigmoid(dist_diff2_15_hl)
        sigmoid2_51_hl = torch.sigmoid(dist_diff2_51_hl)
        sigmoid2_55_hl = torch.sigmoid(dist_diff2_55_hl)

        loss_instant2_11_sl = beta*sigmoid2_11_sl
        loss_instant2_15_sl = beta*sigmoid2_15_sl
        loss_instant2_51_sl = beta*sigmoid2_51_sl
        loss_instant2_55_sl = beta*sigmoid2_55_sl
        loss_instant2_11_hl = beta*sigmoid2_11_hl
        loss_instant2_15_hl = beta*sigmoid2_15_hl
        loss_instant2_51_hl = beta*sigmoid2_51_hl
        loss_instant2_55_hl = beta*sigmoid2_55_hl

        # calculate instantaneous loss
        loss_instant1 = torch.cat((loss_instant1_11_sl, loss_instant1_15_sl, loss_instant1_51_sl, loss_instant1_55_sl,
                                   loss_instant1_11_hl, loss_instant1_15_hl, loss_instant1_51_hl, loss_instant1_55_hl), dim=0)
        loss_instant2 = torch.cat((loss_instant2_11_sl, loss_instant2_15_sl, loss_instant2_51_sl, loss_instant2_55_sl,
                                   loss_instant2_11_hl, loss_instant2_15_hl, loss_instant2_51_hl, loss_instant2_55_hl), dim=0)
        loss_fun_1 = alpha * (100 * torch.tan(theta1) ** 2 + 100 * torch.tan(phi1) ** 2 + (thrust1 - gravity) ** 2 + loss_instant1)
        loss_fun_2 = alpha * (100 * torch.tan(theta2) ** 2 + 100 * torch.tan(phi2) ** 2 + (thrust2 - gravity) ** 2 + loss_instant2)

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * vx_11.squeeze() - lam11_2.squeeze() * vy_11.squeeze() - \
                lam11_3.squeeze() * vz_11.squeeze() - lam11_4.squeeze() * torch.tan(theta1).squeeze() * gravity + \
                lam11_5.squeeze() * torch.tan(phi1).squeeze() * gravity - lam11_6.squeeze() * (thrust1 - gravity).squeeze() - \
                lam12_1.squeeze() * vx_12.squeeze() - lam12_2.squeeze() * vy_12.squeeze() - \
                lam12_3.squeeze() * vz_12.squeeze() - lam12_4.squeeze() * torch.tan(theta2).squeeze() * gravity + \
                lam12_5.squeeze() * torch.tan(phi2).squeeze() * gravity - lam12_6.squeeze() * (thrust2 - gravity).squeeze() + \
                loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * vx_21.squeeze() - lam21_2.squeeze() * vy_21.squeeze() - \
                lam21_3.squeeze() * vz_21.squeeze() - lam21_4.squeeze() * torch.tan(theta1).squeeze() * gravity + \
                lam21_5.squeeze() * torch.tan(phi1).squeeze() * gravity - lam21_6.squeeze() * (thrust1 - gravity).squeeze() - \
                lam22_1.squeeze() * vx_22.squeeze() - lam22_2.squeeze() * vy_22.squeeze() - \
                lam22_3.squeeze() * vz_22.squeeze() - lam22_4.squeeze() * torch.tan(theta2).squeeze() * gravity + \
                lam22_5.squeeze() * torch.tan(phi2).squeeze() * gravity - lam22_6.squeeze() * (thrust2 - gravity).squeeze() + \
                loss_fun_2.squeeze()

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        diff_constraint_hom_1 = dvdt_1 + ham_1
        diff_constraint_hom_2 = dvdt_2 + ham_2
        diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        # supervised learning for values
        value1_difference = y1[:, :supervised_index] - alpha * groundtruth_values[:, :supervised_index]
        value2_difference = y2[:, :supervised_index] - alpha * groundtruth_values[:, supervised_index:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1[:supervised_index, :],
                                         lam11_2[:supervised_index, :],
                                         lam11_3[:supervised_index, :],
                                         lam11_4[:supervised_index, :],
                                         lam11_5[:supervised_index, :],
                                         lam11_6[:supervised_index, :],
                                         lam12_1[:supervised_index, :],
                                         lam12_2[:supervised_index, :],
                                         lam12_3[:supervised_index, :],
                                         lam12_4[:supervised_index, :],
                                         lam12_5[:supervised_index, :],
                                         lam12_6[:supervised_index, :]), dim=1)
        costate2_prediction = torch.cat((lam21_1[:supervised_index, :],
                                         lam21_2[:supervised_index, :],
                                         lam21_3[:supervised_index, :],
                                         lam21_4[:supervised_index, :],
                                         lam21_5[:supervised_index, :],
                                         lam21_6[:supervised_index, :],
                                         lam22_1[:supervised_index, :],
                                         lam22_2[:supervised_index, :],
                                         lam22_3[:supervised_index, :],
                                         lam22_4[:supervised_index, :],
                                         lam22_5[:supervised_index, :],
                                         lam22_6[:supervised_index, :]), dim=1)
        costate1_difference = costate1_prediction - alpha * groundtruth_costates[:, :supervised_index].squeeze()
        costate2_difference = costate2_prediction - alpha * groundtruth_costates[:, supervised_index:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # boundary condition check
        dirichlet_1 = y1[:, supervised_index:][dirichlet_mask] - alpha * source_boundary_values[:, :hji_index][dirichlet_mask]
        dirichlet_2 = y2[:, supervised_index:][dirichlet_mask] - alpha * source_boundary_values[:, hji_index:][dirichlet_mask]
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (weight1, weight2, weight3, weight4) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / weight1,
                'costates_difference': torch.abs(costates_difference).sum() / weight2,
                'dirichlet': torch.abs(dirichlet).sum() / weight3,  
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / weight4}
    return interaction_hji

import torch
import diff_operators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_narrow_HJI_supervised(dataset, Weight, alpha):
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
        lam11_1 = dvdx_1[:, :1] / ((95 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((39 - 31) / 2)  # lambda_11
        lam11_3 = dvdx_1[:, 2:3] / ((0.2 - (-0.2)) / 2)  # lambda_11
        lam11_4 = dvdx_1[:, 3:4] / ((29 - 18) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 4:5] / ((95 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 5:6] / ((39 - 31) / 2)  # lambda_12
        lam12_3 = dvdx_1[:, 6:7] / ((0.2 - (-0.2)) / 2)  # lambda_12
        lam12_4 = dvdx_1[:, 7:8] / ((29 - 18) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 4:5] / ((95 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 5:6] / ((39 - 31) / 2)  # lambda_21
        lam21_3 = dvdx_2[:, 6:7] / ((0.2 - (-0.2)) / 2)  # lambda_21
        lam21_4 = dvdx_2[:, 7:8] / ((29 - 18) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((95 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((39 - 31) / 2)  # lambda_22
        lam22_3 = dvdx_2[:, 2:3] / ((0.2 - (-0.2)) / 2)  # lambda_22
        lam22_4 = dvdx_2[:, 3:4] / ((29 - 18) / 2)  # lambda_22

        # supervised learning for values
        value1_difference = y1 - alpha * groundtruth_values[:, :y1.shape[1]]
        value2_difference = y2 - alpha * groundtruth_values[:, y2.shape[1]:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1, lam11_2, lam11_3, lam11_4,
                                         lam12_1, lam12_2, lam12_3, lam12_4), dim=1)
        costate2_prediction = torch.cat((lam21_1, lam21_2, lam21_3, lam21_4,
                                         lam22_1, lam22_2, lam22_3, lam22_4), dim=1)
        costate1_difference = costate1_prediction - alpha * groundtruth_costates[:, :y1.shape[1]].squeeze()
        costate2_difference = costate2_prediction - alpha * groundtruth_costates[:, y2.shape[1]:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # A factor of (weight1, weight2) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / weight1,
                'costates_difference': torch.abs(costates_difference).sum() / weight2}
    return interaction_hji


def initialize_narrow_HJI_hyrid(dataset, Weight, alpha):
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
        num_sl = supervised_index
        num_hl = hji_index

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
        lam11_1 = dvdx_1[:, :1] / ((95 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((39 - 31) / 2)  # lambda_11
        lam11_3 = dvdx_1[:, 2:3] / ((0.2 - (-0.2)) / 2)  # lambda_11
        lam11_4 = dvdx_1[:, 3:4] / ((29 - 18) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 4:5] / ((95 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 5:6] / ((39 - 31) / 2)  # lambda_12
        lam12_3 = dvdx_1[:, 6:7] / ((0.2 - (-0.2)) / 2)  # lambda_12
        lam12_4 = dvdx_1[:, 7:8] / ((29 - 18) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 4:5] / ((95 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 5:6] / ((39 - 31) / 2)  # lambda_21
        lam21_3 = dvdx_2[:, 6:7] / ((0.2 - (-0.2)) / 2)  # lambda_21
        lam21_4 = dvdx_2[:, 7:8] / ((29 - 18) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((95 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((39 - 31) / 2)  # lambda_22
        lam22_3 = dvdx_2[:, 2:3] / ((0.2 - (-0.2)) / 2)  # lambda_22
        lam22_4 = dvdx_2[:, 3:4] / ((29 - 18) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R = torch.tensor([70.], dtype=torch.float32).to(device)  # road length at y direction
        threshold_11 = torch.tensor([1.5], dtype=torch.float32).to(device)  # collision penalty threshold
        threshold_15 = torch.tensor([1.9], dtype=torch.float32).to(device)  # collision penalty threshold
        threshold_51 = torch.tensor([1.9], dtype=torch.float32).to(device)  # collision penalty threshold
        threshold_55 = torch.tensor([2.5], dtype=torch.float32).to(device)  # collision penalty threshold
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action
        # H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        u1 = 0.5 * lam11_4 * 1
        w1 = lam11_3 / 200 * 1

        # Agent 2's action
        u2 = 0.5 * lam22_4 * 1
        w2 = lam22_3 / 200 * 1

        # set up bounds for u1 and u2
        max_acc_u = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc_u = torch.tensor([-5.], dtype=torch.float32).to(device)
        max_acc_w = torch.tensor([1.], dtype=torch.float32).to(device)
        min_acc_w = torch.tensor([-1.], dtype=torch.float32).to(device)

        u1[torch.where(u1 > max_acc_u)] = max_acc_u
        u1[torch.where(u1 < min_acc_u)] = min_acc_u
        u2[torch.where(u2 > max_acc_u)] = max_acc_u
        u2[torch.where(u2 < min_acc_u)] = min_acc_u

        w1[torch.where(w1 > max_acc_w)] = max_acc_w
        w1[torch.where(w1 < min_acc_w)] = min_acc_w
        w2[torch.where(w2 > max_acc_w)] = max_acc_w
        w2[torch.where(w2 < min_acc_w)] = min_acc_w

        # unnormalize the state for agent 1
        dx_11_11_sl = (x1[:, :num_sl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_11_11_sl = (x1[:, :num_sl, 2:3] + 1) * (39 - 31) / 2 + 31
        
        dx_11_15_sl = (x1[:, num_sl:2*num_sl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_11_15_sl = (x1[:, num_sl:2*num_sl, 2:3] + 1) * (39 - 31) / 2 + 31

        dx_11_51_sl = (x1[:, 2*num_sl:3*num_sl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_11_51_sl = (x1[:, 2*num_sl:3*num_sl, 2:3] + 1) * (39 - 31) / 2 + 31

        dx_11_55_sl = (x1[:, 3*num_sl:4*num_sl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_11_55_sl = (x1[:, 3*num_sl:4*num_sl, 2:3] + 1) * (39 - 31) / 2 + 31

        dx_11_11_hl = (x1[:, 4*num_sl:4*num_sl+num_hl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_11_11_hl = (x1[:, 4*num_sl:4*num_sl+num_hl, 2:3] + 1) * (39 - 31) / 2 + 31
        
        dx_11_15_hl = (x1[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_11_15_hl = (x1[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 2:3] + 1) * (39 - 31) / 2 + 31

        dx_11_51_hl = (x1[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_11_51_hl = (x1[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 2:3] + 1) * (39 - 31) / 2 + 31

        dx_11_55_hl = (x1[:, 4*num_sl+3*num_hl:, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_11_55_hl = (x1[:, 4*num_sl+3*num_hl:, 2:3] + 1) * (39 - 31) / 2 + 31

        theta_11 = (x1[:, :, 3:4] + 1) * (0.2 - (-0.2)) / 2 + (-0.2)
        v_11 = (x1[:, :, 4:5] + 1) * (29 - 18) / 2 + 18

        # unnormalize the state for agent 2
        dx_12_11_sl = (x1[:, :num_sl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_12_11_sl = (x1[:, :num_sl, 6:7] + 1) * (39 - 31) / 2 + 31
        
        dx_12_15_sl = (x1[:, num_sl:2*num_sl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_12_15_sl = (x1[:, num_sl:2*num_sl, 6:7] + 1) * (39 - 31) / 2 + 31

        dx_12_51_sl = (x1[:, 2*num_sl:3*num_sl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_12_51_sl = (x1[:, 2*num_sl:3*num_sl, 6:7] + 1) * (39 - 31) / 2 + 31

        dx_12_55_sl = (x1[:, 3*num_sl:4*num_sl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_12_55_sl = (x1[:, 3*num_sl:4*num_sl, 6:7] + 1) * (39 - 31) / 2 + 31

        dx_12_11_hl = (x1[:, 4*num_sl:4*num_sl+num_hl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_12_11_hl = (x1[:, 4*num_sl:4*num_sl+num_hl, 6:7] + 1) * (39 - 31) / 2 + 31
        
        dx_12_15_hl = (x1[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_12_15_hl = (x1[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 6:7] + 1) * (39 - 31) / 2 + 31

        dx_12_51_hl = (x1[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_12_51_hl = (x1[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 6:7] + 1) * (39 - 31) / 2 + 31

        dx_12_55_hl = (x1[:, 4*num_sl+3*num_hl:, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_12_55_hl = (x1[:, 4*num_sl+3*num_hl:, 6:7] + 1) * (39 - 31) / 2 + 31

        theta_12 = (x1[:, :, 7:8] + 1) * (0.2 - (-0.2)) / 2 + (-0.2)
        v_12 = (x1[:, :, 8:9] + 1) * (29 - 18) / 2 + 18

        # calculate the collision area lower and upper bounds
        dist_diff1_11_sl = (-(torch.sqrt(((R - dx_12_11_sl) - dx_11_11_sl) ** 2 + (dy_12_11_sl - dy_11_11_sl) ** 2)
                            - threshold_11) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_15_sl = (-(torch.sqrt(((R - dx_12_15_sl) - dx_11_15_sl) ** 2 + (dy_12_15_sl - dy_11_15_sl) ** 2)
                            - threshold_15) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_51_sl = (-(torch.sqrt(((R - dx_12_51_sl) - dx_11_51_sl) ** 2 + (dy_12_51_sl - dy_11_51_sl) ** 2)
                            - threshold_51) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_55_sl = (-(torch.sqrt(((R - dx_12_55_sl) - dx_11_55_sl) ** 2 + (dy_12_55_sl - dy_11_55_sl) ** 2) 
                            - threshold_55) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_11_hl = (-(torch.sqrt(((R - dx_12_11_hl) - dx_11_11_hl) ** 2 + (dy_12_11_hl - dy_11_11_hl) ** 2)
                            - threshold_11) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_15_hl = (-(torch.sqrt(((R - dx_12_15_hl) - dx_11_15_hl) ** 2 + (dy_12_15_hl - dy_11_15_hl) ** 2)
                            - threshold_15) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_51_hl = (-(torch.sqrt(((R - dx_12_51_hl) - dx_11_51_hl) ** 2 + (dy_12_51_hl - dy_11_51_hl) ** 2)
                            - threshold_51) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff1_55_hl = (-(torch.sqrt(((R - dx_12_55_hl) - dx_11_55_hl) ** 2 + (dy_12_55_hl - dy_11_55_hl) ** 2)
                            - threshold_55) * 5).squeeze().reshape(-1, 1).to(device)
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
        dx_21_11_sl = (x2[:, :num_sl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_21_11_sl = (x2[:, :num_sl, 6:7] + 1) * (39 - 31) / 2 + 31
        
        dx_21_15_sl = (x2[:, num_sl:2*num_sl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_21_15_sl = (x2[:, num_sl:2*num_sl, 6:7] + 1) * (39 - 31) / 2 + 31

        dx_21_51_sl = (x2[:, 2*num_sl:3*num_sl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_21_51_sl = (x2[:, 2*num_sl:3*num_sl, 6:7] + 1) * (39 - 31) / 2 + 31

        dx_21_55_sl = (x2[:, 3*num_sl:4*num_sl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_21_55_sl = (x2[:, 3*num_sl:4*num_sl, 6:7] + 1) * (39 - 31) / 2 + 31

        dx_21_11_hl = (x2[:, 4*num_sl:4*num_sl+num_hl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_21_11_hl = (x2[:, 4*num_sl:4*num_sl+num_hl, 6:7] + 1) * (39 - 31) / 2 + 31
        
        dx_21_15_hl = (x2[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_21_15_hl = (x2[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 6:7] + 1) * (39 - 31) / 2 + 31

        dx_21_51_hl = (x2[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_21_51_hl = (x2[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 6:7] + 1) * (39 - 31) / 2 + 31

        dx_21_55_hl = (x2[:, 4*num_sl+3*num_hl:, 5:6] + 1) * (95 - 15) / 2 + 15
        dy_21_55_hl = (x2[:, 4*num_sl+3*num_hl:, 6:7] + 1) * (39 - 31) / 2 + 31

        theta_21 = (x2[:, :, 7:8] + 1) * (0.2 - (-0.2)) / 2 + (-0.2)
        v_21 = (x2[:, :, 8:9] + 1) * (29 - 18) / 2 + 18

        # unnormalize the state for agent 2
        dx_22_11_sl = (x2[:, :num_sl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_22_11_sl = (x2[:, :num_sl, 2:3] + 1) * (39 - 31) / 2 + 31
        
        dx_22_15_sl = (x2[:, num_sl:2*num_sl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_22_15_sl = (x2[:, num_sl:2*num_sl, 2:3] + 1) * (39 - 31) / 2 + 31

        dx_22_51_sl = (x2[:, 2*num_sl:3*num_sl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_22_51_sl = (x2[:, 2*num_sl:3*num_sl, 2:3] + 1) * (39 - 31) / 2 + 31

        dx_22_55_sl = (x2[:, 3*num_sl:4*num_sl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_22_55_sl = (x2[:, 3*num_sl:4*num_sl, 2:3] + 1) * (39 - 31) / 2 + 31

        dx_22_11_hl = (x2[:, 4*num_sl:4*num_sl+num_hl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_22_11_hl = (x2[:, 4*num_sl:4*num_sl+num_hl, 2:3] + 1) * (39 - 31) / 2 + 31
        
        dx_22_15_hl = (x2[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_22_15_hl = (x2[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 2:3] + 1) * (39 - 31) / 2 + 31

        dx_22_51_hl = (x2[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_22_51_hl = (x2[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 2:3] + 1) * (39 - 31) / 2 + 31

        dx_22_55_hl = (x2[:, 4*num_sl+3*num_hl:, 1:2] + 1) * (95 - 15) / 2 + 15
        dy_22_55_hl = (x2[:, 4*num_sl+3*num_hl:, 2:3] + 1) * (39 - 31) / 2 + 31

        theta_22 = (x2[:, :, 3:4] + 1) * (0.2 - (-0.2)) / 2 + (-0.2)
        v_22 = (x2[:, :, 4:5] + 1) * (29 - 18) / 2 + 18

        # calculate the collision area lower and upper bounds
        dist_diff2_11_sl = (-(torch.sqrt(((R - dx_22_11_sl) - dx_21_11_sl) ** 2 + (dy_22_11_sl - dy_21_11_sl) ** 2)
                            - threshold_11) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_15_sl = (-(torch.sqrt(((R - dx_22_15_sl) - dx_21_15_sl) ** 2 + (dy_22_15_sl - dy_21_15_sl) ** 2)
                            - threshold_15) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_51_sl = (-(torch.sqrt(((R - dx_22_51_sl) - dx_21_51_sl) ** 2 + (dy_22_51_sl - dy_21_51_sl) ** 2)
                            - threshold_51) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_55_sl = (-(torch.sqrt(((R - dx_22_55_sl) - dx_21_55_sl) ** 2 + (dy_22_55_sl - dy_21_55_sl) ** 2)
                            - threshold_55) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_11_hl = (-(torch.sqrt(((R - dx_22_11_hl) - dx_21_11_hl) ** 2 + (dy_22_11_hl - dy_21_11_hl) ** 2)
                            - threshold_11) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_15_hl = (-(torch.sqrt(((R - dx_22_15_hl) - dx_21_15_hl) ** 2 + (dy_22_15_hl - dy_21_15_hl) ** 2)
                            - threshold_15) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_51_hl = (-(torch.sqrt(((R - dx_22_51_hl) - dx_21_51_hl) ** 2 + (dy_22_51_hl - dy_21_51_hl) ** 2)
                            - threshold_51) * 5).squeeze().reshape(-1, 1).to(device)
        dist_diff2_55_hl = (-(torch.sqrt(((R - dx_22_55_hl) - dx_21_55_hl) ** 2 + (dy_22_55_hl - dy_21_55_hl) ** 2)
                            - threshold_55) * 5).squeeze().reshape(-1, 1).to(device)
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
        loss_fun_1 = alpha * (100 * w1 ** 2 + u1 ** 2 + loss_instant1)
        loss_fun_2 = alpha * (100 * w2 ** 2 + u2 ** 2 + loss_instant2)

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * v_11.squeeze() * torch.cos(theta_11.squeeze()) - \
                lam11_2.squeeze() * v_11.squeeze() * torch.sin(theta_11.squeeze()) - \
                lam11_3.squeeze() * w1.squeeze() - lam11_4.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v_12.squeeze() * torch.cos(theta_12.squeeze()) - \
                lam12_2.squeeze() * v_12.squeeze() * torch.sin(theta_12.squeeze()) - \
                lam12_3.squeeze() * w2.squeeze() - lam12_4.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v_21.squeeze() * torch.cos(theta_21.squeeze()) - \
                lam21_2.squeeze() * v_21.squeeze() * torch.sin(theta_21.squeeze()) - \
                lam21_3.squeeze() * w1.squeeze() - lam21_4.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v_22.squeeze() * torch.cos(theta_22.squeeze()) - \
                lam22_2.squeeze() * v_22.squeeze() * torch.sin(theta_22.squeeze()) - \
                lam22_3.squeeze() * w2.squeeze() - lam22_4.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

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
                                         lam12_1[:supervised_index, :],
                                         lam12_2[:supervised_index, :],
                                         lam12_3[:supervised_index, :],
                                         lam12_4[:supervised_index, :]), dim=1)
        costate2_prediction = torch.cat((lam21_1[:supervised_index, :],
                                         lam21_2[:supervised_index, :],
                                         lam21_3[:supervised_index, :],
                                         lam21_4[:supervised_index, :],
                                         lam22_1[:supervised_index, :],
                                         lam22_2[:supervised_index, :],
                                         lam22_3[:supervised_index, :],
                                         lam22_4[:supervised_index, :]), dim=1)
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

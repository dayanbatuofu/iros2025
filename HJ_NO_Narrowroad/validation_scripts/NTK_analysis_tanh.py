# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diff_operators
import modules_hno
import time
import torch
import numpy as np
import scipy.io as scio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_ntk(J1_list, J2_list):
    N = len(J1_list)
    D1, D2 = J1_list[0].shape[0], J2_list[0].shape[0]
    Ker = torch.zeros((D1, D2)).to(device)
    for k in range(N):
        J1 = torch.reshape(J1_list[k], shape=(D1, -1))
        J2 = torch.reshape(J2_list[k], shape=(D2, -1))

        K = torch.matmul(J1, J2.T)
        Ker = Ker + K
    return Ker

def compute_jacobian_w(v_pre, param):
    N = v_pre.shape[0]
    jac = torch.empty(0, param.shape[0], param.shape[1]).to(device)
    for n in range(N):
        jac_tmp = torch.autograd.grad(v_pre[n], param, create_graph=True)[0]
        with torch.no_grad():
            jac = torch.cat((jac, jac_tmp.unsqueeze(0)), dim=0)
    return jac

def compute_jacobian_b(v_pre, param):
    N = v_pre.shape[0]
    jac = torch.empty(0, param.shape[0]).to(device)
    for n in range(N):
        jac_tmp = torch.autograd.grad(v_pre[n], param, create_graph=True)[0]
        with torch.no_grad():
            jac = torch.cat((jac, jac_tmp.unsqueeze(0)), dim=0)
    return jac

def loss_fn(model_output, threshold, alpha):
    x = model_output['model_in']
    y = model_output['model_out']
    cut_index = x.shape[1] // 2
    supervised_index = cut_index // 2
    num_sl = supervised_index

    y1 = y[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
    y2 = y[:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
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
    threshold = torch.tensor([threshold], dtype=torch.float32).to(device)  # collision penalty threshold
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
    dx_11_sl = (x1[:, :num_sl, 1:2] + 1) * (95 - 15) / 2 + 15
    dy_11_sl = (x1[:, :num_sl, 2:3] + 1) * (39 - 31) / 2 + 31

    dx_11_hl = (x1[:, num_sl:, 1:2] + 1) * (95 - 15) / 2 + 15
    dy_11_hl = (x1[:, num_sl:, 2:3] + 1) * (39 - 31) / 2 + 31

    theta_11 = (x1[:, :, 3:4] + 1) * (0.2 - (-0.2)) / 2 + (-0.2)
    v_11 = (x1[:, :, 4:5] + 1) * (29 - 18) / 2 + 18

    # unnormalize the state for agent 2
    dx_12_sl = (x1[:, :num_sl, 5:6] + 1) * (95 - 15) / 2 + 15
    dy_12_sl = (x1[:, :num_sl, 6:7] + 1) * (39 - 31) / 2 + 31

    dx_12_hl = (x1[:, num_sl:, 5:6] + 1) * (95 - 15) / 2 + 15
    dy_12_hl = (x1[:, num_sl:, 6:7] + 1) * (39 - 31) / 2 + 31

    theta_12 = (x1[:, :, 7:8] + 1) * (0.2 - (-0.2)) / 2 + (-0.2)
    v_12 = (x1[:, :, 8:9] + 1) * (29 - 18) / 2 + 18

    # calculate the collision area lower and upper bounds
    dist_diff1_sl = (-(torch.sqrt(((R - dx_12_sl) - dx_11_sl) ** 2 + (dy_12_sl - dy_11_sl) ** 2)
                     - threshold) * 5).squeeze().reshape(-1, 1).to(device)
    dist_diff1_hl = (-(torch.sqrt(((R - dx_12_hl) - dx_11_hl) ** 2 + (dy_12_hl - dy_11_hl) ** 2)
                     - threshold) * 5).squeeze().reshape(-1, 1).to(device)
    sigmoid1_sl = torch.sigmoid(dist_diff1_sl)
    sigmoid1_hl = torch.sigmoid(dist_diff1_hl)

    loss_instant1_sl = beta * sigmoid1_sl
    loss_instant1_hl = beta * sigmoid1_hl

    # unnormalize the state for agent 1
    dx_21_sl = (x2[:, :num_sl, 5:6] + 1) * (95 - 15) / 2 + 15
    dy_21_sl = (x2[:, :num_sl, 6:7] + 1) * (39 - 31) / 2 + 31

    dx_21_hl = (x2[:, num_sl:, 5:6] + 1) * (95 - 15) / 2 + 15
    dy_21_hl = (x2[:, num_sl:, 6:7] + 1) * (39 - 31) / 2 + 31

    theta_21 = (x2[:, :, 7:8] + 1) * (0.2 - (-0.2)) / 2 + (-0.2)
    v_21 = (x2[:, :, 8:9] + 1) * (29 - 18) / 2 + 18

    # unnormalize the state for agent 2
    dx_22_sl = (x2[:, :num_sl, 1:2] + 1) * (95 - 15) / 2 + 15
    dy_22_sl = (x2[:, :num_sl, 2:3] + 1) * (39 - 31) / 2 + 31

    dx_22_hl = (x2[:, num_sl:, 1:2] + 1) * (95 - 15) / 2 + 15
    dy_22_hl = (x2[:, num_sl:, 2:3] + 1) * (39 - 31) / 2 + 31

    theta_22 = (x2[:, :, 3:4] + 1) * (0.2 - (-0.2)) / 2 + (-0.2)
    v_22 = (x2[:, :, 4:5] + 1) * (29 - 18) / 2 + 18

    # calculate the collision area lower and upper bounds
    dist_diff2_sl = (-(torch.sqrt(((R - dx_22_sl) - dx_21_sl) ** 2 + (dy_22_sl - dy_21_sl) ** 2)
                     - threshold) * 5).squeeze().reshape(-1, 1).to(device)
    dist_diff2_hl = (-(torch.sqrt(((R - dx_22_hl) - dx_21_hl) ** 2 + (dy_22_hl - dy_21_hl) ** 2)
                     - threshold) * 5).squeeze().reshape(-1, 1).to(device)
    sigmoid2_sl = torch.sigmoid(dist_diff2_sl)
    sigmoid2_hl = torch.sigmoid(dist_diff2_hl)

    loss_instant2_sl = beta * sigmoid2_sl
    loss_instant2_hl = beta * sigmoid2_hl

    # calculate instantaneous loss
    loss_instant1 = torch.cat((loss_instant1_sl,
                               loss_instant1_hl), dim=0)
    loss_instant2 = torch.cat((loss_instant2_sl,
                               loss_instant2_hl), dim=0)
    loss_fun_1 = 100 * w1 ** 2 + u1 ** 2 + loss_instant1
    loss_fun_2 = 100 * w2 ** 2 + u2 ** 2 + loss_instant2

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

    y_v = torch.cat((alpha*y1[:, :supervised_index],
                     alpha*y2[:, :supervised_index]), dim=1)

    # supervised learning for costates
    costate1_prediction = torch.cat((lam11_1[:supervised_index, :],
                                     lam11_2[:supervised_index, :],
                                     lam11_3[:supervised_index, :],
                                     lam11_4[:supervised_index, :],
                                     lam12_1[:supervised_index, :],
                                     lam12_2[:supervised_index, :],
                                     lam12_3[:supervised_index, :],
                                     lam12_4[:supervised_index, :]), dim=0)
    costate2_prediction = torch.cat((lam21_1[:supervised_index, :],
                                     lam21_2[:supervised_index, :],
                                     lam21_3[:supervised_index, :],
                                     lam21_4[:supervised_index, :],
                                     lam22_1[:supervised_index, :],
                                     lam22_2[:supervised_index, :],
                                     lam22_3[:supervised_index, :],
                                     lam22_4[:supervised_index, :]), dim=0)

    y_c = torch.cat((costate1_prediction,
                     costate2_prediction), dim=0)

    return {'v_pre_rr': diff_constraint_hom / 1000,
            'v_pre_vv': y_v.squeeze() / 40,
            'v_pre_cc': y_c.squeeze() / 500}

def effective_rank(ntk):
    # eigs, _ = torch.symeig(ntk)
    eigs, _ = torch.linalg.eigh(ntk)
    return torch.sum(eigs) / torch.max(eigs) if torch.max(eigs) != 0 else 0

if __name__ == '__main__':

    logging_root = './logs'
    N_neurons = 64
    layers = 5
    torch.manual_seed(0)

    for n_sample in range(2):
        cond_num_list = []
        erank_list = []
        for i in range(4):
            # policy = ['1_1', '1_2', '1_3', '1_4', '1_5', '2_2', '2_3', '2_4', '2_5', '3_3', '3_4', '3_5', '4_4', '4_5',
            #           '5_5']
            # param_type_P1 = ['theta_11', 'theta_12', 'theta_13', 'theta_14', 'theta_15', 'theta_22', 'theta_23',
            #                  'theta_24', 'theta_25', 'theta_33', 'theta_34', 'theta_35', 'theta_44', 'theta_45', 'theta_55']
            # param_type_P2 = ['theta_11', 'theta_21', 'theta_31', 'theta_41', 'theta_51', 'theta_22', 'theta_32',
            #                  'theta_42', 'theta_52', 'theta_33', 'theta_43', 'theta_53', 'theta_44', 'theta_54', 'theta_55']

            policy = ['1_1', '1_5', '5_1', '5_5']
            param_type_P1 = ['theta_11', 'theta_15', 'theta_51', 'theta_55']
            param_type_P2 = ['theta_11', 'theta_15', 'theta_51', 'theta_55']

            N_choice = i
            alpha = 1
            theta = (int(policy[N_choice][0]), int(policy[N_choice][2:]))

            theta1, theta2 = int(policy[N_choice][0]), int(policy[N_choice][2])
            threshold = 0.1 * (theta1 + theta2) + 0.05 * min(theta1, theta2) + 1.25
            count = 0

            print(N_choice, threshold)

            ckpt_path = './model/tanh/model_hno_tanh.pth'
            activation = 'tanh'

            # Initialize and load the model
            model = modules_hno.SingleBVPNet(in_features=9, out_features=64, type=activation, mode='mlp',
                                             final_layer_factor=1., hidden_features=64, num_hidden_layers=3)
            model.to(device)
            checkpoint = torch.load(ckpt_path)
            try:
                model_weights = checkpoint['model']
            except:
                model_weights = checkpoint
            model.load_state_dict(model_weights)
            model.eval()

            data_path = './train_data/narrowroad_param_fun_50.mat'
            Param_fun = scio.loadmat(data_path)
            param_fun_P1 = torch.tensor(Param_fun[str(param_type_P1[N_choice])], dtype=torch.float32)
            param_fun_P2 = torch.tensor(Param_fun[str(param_type_P2[N_choice])], dtype=torch.float32)
            param_fun = torch.cat((param_fun_P1, param_fun_P2), dim=0)

            "test data omits inevitable collisions"
            path = './train_data/data_train_narrowroad_' + str(policy[N_choice]) + '_1000.mat'
            # path = './test_data/data_test_narrowroad_' + str(policy[N_choice]) + '_600.mat'

            test_data = scio.loadmat(path)

            N_sample = 310

            t_test = torch.tensor(test_data['t'], dtype=torch.float32).flip(1)[:, N_sample*n_sample:N_sample*(n_sample+1)]
            X_test = torch.tensor(test_data['X'], dtype=torch.float32)[:, N_sample*n_sample:N_sample*(n_sample+1)]
            A_test = torch.tensor(test_data['A'], dtype=torch.float32)[:, N_sample*n_sample:N_sample*(n_sample+1)]
            V_test = torch.tensor(test_data['V'], dtype=torch.float32)[:, N_sample*n_sample:N_sample*(n_sample+1)]
            lb = torch.tensor([[15], [31], [-0.2], [18], [15], [31], [-0.2], [18]], dtype=torch.float32)
            ub = torch.tensor([[95], [39], [0.2], [29], [95], [39], [0.2], [29]], dtype=torch.float32)
            X_test = 2.0 * (X_test - lb) / (ub - lb) - 1.

            coords1_sl = X_test.T
            coords2_sl = torch.cat((coords1_sl[:, 4:], coords1_sl[:, :4]), dim=1)

            coords1_sl = torch.cat((t_test.T, coords1_sl), dim=1)
            coords2_sl = torch.cat((t_test.T, coords2_sl), dim=1)

            # uniformly sample domain and include coordinates for both agents
            coords_pinn = torch.load("data_sample.pt")
            index = coords_pinn.shape[0] // 2

            coords1_pinn = coords_pinn[:index, :][N_sample*n_sample:N_sample*(n_sample+1), :]
            coords2_pinn = coords_pinn[index:, :][N_sample*n_sample:N_sample*(n_sample+1), :]

            coords_1 = torch.cat((coords1_sl, coords1_pinn), dim=0)
            coords_2 = torch.cat((coords2_sl, coords2_pinn), dim=0)

            coords = torch.cat((coords_1, coords_2), dim=0).unsqueeze(0)
            param_num = coords.shape[1] // 2
            param_fun = param_fun.repeat(param_num, 1).unsqueeze(0)
            model_in = {'coords': coords.to(device),
                        'input_fun': param_fun.to(device)}

            model_output = model(model_in)
            losses = loss_fn(model_output, threshold, alpha)
            v_pre_rr = losses['v_pre_rr']
            v_pre_vv = losses['v_pre_vv']
            v_pre_cc = losses['v_pre_cc']

            parm = {}
            for name, parameters in model.named_parameters():
                parm[name] = parameters

            # compute trunk net
            weights1 = []
            bias1 = []
            for num in range(layers):
                weights1.append(parm['net.net.{}.0.weight'.format(num)])
                bias1.append(parm['net.net.{}.0.bias'.format(num)])

            J_r1 = []
            J_v1 = []
            J_c1 = []

            for i in range(layers):
                J_r1.append(compute_jacobian_w(v_pre_rr, weights1[i]))
                J_v1.append(compute_jacobian_w(v_pre_vv, weights1[i]))
                J_c1.append(compute_jacobian_w(v_pre_cc, weights1[i]))

            for i in range(layers):
                if i == layers - 1:
                    J_r1.append(torch.zeros(v_pre_rr.shape[0], 64).to(device))
                    J_v1.append(compute_jacobian_b(v_pre_vv, bias1[i]))
                    J_c1.append(torch.zeros(v_pre_cc.shape[0], 64).to(device))
                else:
                    J_r1.append(compute_jacobian_b(v_pre_rr, bias1[i]))
                    J_v1.append(compute_jacobian_b(v_pre_vv, bias1[i]))
                    J_c1.append(compute_jacobian_b(v_pre_cc, bias1[i]))

            K_rr1 = compute_ntk(J_r1, J_r1)
            K_rv1 = compute_ntk(J_r1, J_v1)
            K_rc1 = compute_ntk(J_r1, J_c1)
            K_vr1 = compute_ntk(J_v1, J_r1)
            K_vv1 = compute_ntk(J_v1, J_v1)
            K_vc1 = compute_ntk(J_v1, J_c1)
            K_cr1 = compute_ntk(J_c1, J_r1)
            K_cv1 = compute_ntk(J_c1, J_v1)
            K_cc1 = compute_ntk(J_c1, J_c1)

            K1 = torch.cat((torch.cat((K_rr1, K_rv1, K_rc1), dim=1),
                            torch.cat((K_vr1, K_vv1, K_vc1), dim=1),
                            torch.cat((K_cr1, K_cv1, K_cc1), dim=1)), dim=0)

            # compute branch net
            weights2 = []
            bias2 = []
            for num in range(layers):
                weights2.append(parm['net.branch_net.{}.0.weight'.format(num)])
                bias2.append(parm['net.branch_net.{}.0.bias'.format(num)])

            J_r2 = []
            J_v2 = []
            J_c2 = []

            for i in range(layers):
                J_r2.append(compute_jacobian_w(v_pre_rr, weights2[i]))
                J_v2.append(compute_jacobian_w(v_pre_vv, weights2[i]))
                J_c2.append(compute_jacobian_w(v_pre_cc, weights2[i]))

            for i in range(layers):
                if i == layers - 1:
                    J_r2.append(torch.zeros(v_pre_rr.shape[0], 64).to(device))
                    J_v2.append(compute_jacobian_b(v_pre_vv, bias2[i]))
                    J_c2.append(torch.zeros(v_pre_cc.shape[0], 64).to(device))
                else:
                    J_r2.append(compute_jacobian_b(v_pre_rr, bias2[i]))
                    J_v2.append(compute_jacobian_b(v_pre_vv, bias2[i]))
                    J_c2.append(compute_jacobian_b(v_pre_cc, bias2[i]))

            K_rr2 = compute_ntk(J_r2, J_r2)
            K_rv2 = compute_ntk(J_r2, J_v2)
            K_rc2 = compute_ntk(J_r2, J_c2)
            K_vr2 = compute_ntk(J_v2, J_r2)
            K_vv2 = compute_ntk(J_v2, J_v2)
            K_vc2 = compute_ntk(J_v2, J_c2)
            K_cr2 = compute_ntk(J_c2, J_r2)
            K_cv2 = compute_ntk(J_c2, J_v2)
            K_cc2 = compute_ntk(J_c2, J_c2)

            K2 = torch.cat((torch.cat((K_rr2, K_rv2, K_rc2), dim=1),
                            torch.cat((K_vr2, K_vv2, K_vc2), dim=1),
                            torch.cat((K_cr2, K_cv2, K_cc2), dim=1)), dim=0)

            K = K1*K2

            # eigvals, eigvec = torch.symeig(K)
            # U, S, V = torch.svd(K)  # Compute singular values manually
            # cond_num = S.max() / S.min()  # Condition number is max(S) / min(S)
            # erank = effective_rank(K)

            eigvals, eigvec = torch.linalg.eigh(K)
            cond_num = torch.linalg.cond(K)
            erank = effective_rank(K)

            cond_num_list.append(cond_num.item())
            erank_list.append(erank.item())

            print(f'Condition Number: {cond_num:.2f}')
            print(f'Effective Rank: {erank:.2f}')

        log_dir = 'NTK/'
        cond_num_file = 'condition_num_tanh' + str(n_sample) + '.txt'
        erank_file = 'erank_tanh' + str(n_sample) + '.txt'
        np.savetxt(os.path.join(log_dir, cond_num_file),
                   np.array(cond_num_list))
        np.savetxt(os.path.join(log_dir, erank_file),
                   np.array(erank_list))
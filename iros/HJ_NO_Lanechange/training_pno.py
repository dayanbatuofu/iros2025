'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import random
from examples.choose_problem_narrwo_road import system, problem, config
from scipy.integrate import solve_ivp
import scipy.io as scio
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(0)

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn, action_fn,
          traj_fn, bound_fn, value_fn, sampling_fn, summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False,
          use_lbfgs=False, loss_schedules=None, validation_fn=None, start_epoch=0):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.9, verbose=True, min_lr=1e-6,
                                                           patience=5000)

    # copy settings from Raissi et al. (2019) and here
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.9, verbose=True, min_lr=1e-6,
                                                               patience=5000)

    # Load the checkpoint if required
    if start_epoch > 0:
        # Load the model and start training from that point onwards
        model_path = os.path.join(model_dir, 'checkpoints', 'model_epoch_%04d.pth' % start_epoch)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.train()
        optim.load_state_dict(checkpoint['optimizer'])
        optim.param_groups[0]['lr'] = lr
        assert(start_epoch == checkpoint['epoch'])
    else:
        # Start training from scratch
        if os.path.exists(model_dir):
            val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
            if val == 'y':
                shutil.rmtree(model_dir)
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs - start_epoch) as pbar:
        train_losses = []
        bcs_losses_hji = []
        bcs_losses_costate = []
        losses_diff_vn = []
        losses_diff_cn = []
        values_diff = []
        HJI_weight = []
        LR = []
        costate_data = dict()
        costate_data.update({'coords_cn': 0,
                             'coords_vn': 0,
                             'costate_gt': 0,
                             'value_gt': 0,
                             'boundary_values_cn': 0,
                             'dirichlet_mask_cn': 0,
                             'input_fun': 0,
                             'num_cn': 0})

        for epoch in range(start_epoch, epochs):
            if not (epoch - 50000) % 50 and epoch and epoch >= 50000:
                # Saving the optimizer state is important to produce consistent results
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict()}
                torch.save(checkpoint,
                       os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                #            np.array(train_losses))
                # np.savetxt(os.path.join(checkpoints_dir, 'bcs_losses_hji_epoch_%04d.txt' % epoch),
                #            np.array(bcs_losses_hji))
                # np.savetxt(os.path.join(checkpoints_dir, 'bcs_losses_costate_epoch_%04d.txt' % epoch),
                #            np.array(bcs_losses_costate))
                # np.savetxt(os.path.join(checkpoints_dir, 'losses_diff_vn_epoch_%04d.txt' % epoch),
                #            np.array(losses_diff_vn))
                # np.savetxt(os.path.join(checkpoints_dir, 'losses_diff_cn_epoch_%04d.txt' % epoch),
                #            np.array(losses_diff_cn))
                # np.savetxt(os.path.join(checkpoints_dir, 'values_diff_epoch_%04d.txt' % epoch),
                #            np.array(values_diff))
                if validation_fn is not None:
                    validation_fn(model, checkpoints_dir, epoch)

            # self-supervised learning
            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.to(device) for key, value in model_input.items()}
                gt = {key: value.to(device) for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                counter = int(gt['counter'])
                counter_end = int(gt['counter_end'])

                if counter == 0:
                    model_output = model(model_input)
                    losses = loss_fn(model_output, gt)

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        if loss_name == 'weight':
                            if loss == 1:
                                hji_weight = 0
                            else:
                                hji_weight = loss
                            continue
                        if loss_name == 'dirichlet_vn':
                            bcs_loss_hji = loss.mean()
                            single_loss = bcs_loss_hji
                        if loss_name == 'dirichlet_cn':
                            bcs_loss_costate = loss.mean()
                            single_loss = bcs_loss_costate
                        if loss_name == 'costate_difference_vn':
                            loss_diff_vn = loss.mean()
                            single_loss = loss_diff_vn
                        if loss_name == 'costate_difference_cn':
                            loss_diff_cn = loss.mean()
                            single_loss = loss_diff_cn
                        if loss_name == 'value_difference':
                            value_diff = loss.mean()
                            single_loss = value_diff
                        else:
                            single_loss = loss.mean()

                        if loss_schedules is not None and loss_name in loss_schedules:
                            writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps),
                                              total_steps)
                            single_loss *= loss_schedules[loss_name](total_steps)

                        writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    bcs_losses_hji.append(bcs_loss_hji.item())
                    bcs_losses_costate.append(bcs_loss_costate.item())
                    losses_diff_vn.append(loss_diff_vn.item())
                    losses_diff_cn.append(loss_diff_cn.item())
                    values_diff.append(value_diff.item())
                    HJI_weight.append(hji_weight)
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not total_steps % steps_til_summary:
                        torch.save(model.state_dict(),
                                   os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                        optim.step()

                    scheduler.step(train_loss)

                    lr_scheduler = optim.state_dict()['param_groups'][0]['lr']
                    LR.append(lr_scheduler)

                else:
                    numrollout = int(gt['numrollout']) - 1
                    numcostate = int(gt['numcostate'])
                    numpoints = int(gt['numpoints'])
                    N_sample = numcostate*(numrollout + 1)

                    if counter == counter_end:
                        model_input['coords_cn'] = costate_data['coords_cn']
                        model_input['coords_vn'] = costate_data['coords_vn']
                        model_input['input_fun'] = costate_data['input_fun']
                        gt['costate_gt'] = costate_data['costate_gt']
                        gt['value_gt'] = costate_data['value_gt']
                        gt['num_cn'] = costate_data['num_cn']

                    elif not (counter - 1) % 10 or counter == 1:

                        # dynamical sampling for initial state applied to costate net
                        if counter == 1:
                            coords_data_save = {'coords': model_input['coords_cn'].squeeze().detach().cpu().numpy()}
                            save_path = 'cn_data_tanh/coords_cn_data_' + str(counter) + '.mat'
                            scio.savemat(save_path, coords_data_save)

                        if not counter == 1:
                            coords_vn_tmp = model_input['coords_vn']
                            input_fun_tmp = model_input['input_fun']
                            model_input['coords_vn'] = costate_data['coords_vn']
                            model_input['coords_cn'] = costate_data['coords_cn']
                            model_input['input_fun'] = costate_data['input_fun']
                            gt['num_cn'] = costate_data['num_cn']
                            model_output = model(model_input)
                            model_input['coords_cn'] = sampling_fn(model_output, gt, counter)['coords_cn']
                            model_input['coords_vn'] = coords_vn_tmp
                            model_input['input_fun'] = input_fun_tmp

                        coords1_update_11 = model_input['coords_cn'][:, :numcostate, :].squeeze(0)
                        coords2_update_11 = model_input['coords_cn'][:, numcostate:, :].squeeze(0)

                        coords1_mask_update_11 = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
                        coords2_mask_update_11 = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)

                        for num in range(numrollout):  # generate closed-loop trajectories at t=3.0s
                            model_output = model(model_input)
                            action = action_fn(model_output)
                            with torch.no_grad():
                                coords_new = traj_fn(model_output, gt, action, num)
                                model_input['coords_cn'] = coords_new['coords_cn']
                                coords1_next_11 = coords_new['coords_cn'][:, :numcostate, :].squeeze(0)
                                coords2_next_11 = coords_new['coords_cn'][:, numcostate:, :].squeeze(0)
                                coords1_update_11 = torch.cat((coords1_update_11, coords1_next_11), dim=0)
                                coords2_update_11 = torch.cat((coords2_update_11, coords2_next_11), dim=0)
                                
                                coords1_mask_next_11 = coords_new['coords_mask'][:, :numcostate, :]
                                coords2_mask_next_11 = coords_new['coords_mask'][:, numcostate:, :]
                                coords1_mask_update_11 = torch.cat((coords1_mask_update_11, coords1_mask_next_11), dim=1)
                                coords2_mask_update_11 = torch.cat((coords2_mask_update_11, coords2_mask_next_11), dim=1)
                            # gc.collect()
                            # torch.cuda.empty_cache()

                        # add boundary points
                        coords1_cn = coords1_update_11
                        coords2_cn = coords2_update_11
                        coords_cn = torch.cat((coords1_cn, coords2_cn), dim=0)

                        coords1_mask = coords1_mask_update_11
                        coords2_mask = coords2_mask_update_11
                        coords_mask = torch.cat((coords1_mask, coords2_mask), dim=1)

                        # record the remaining state for each type
                        num_11 = coords1_mask_update_11.sum().unsqueeze(0)

                        print('remaining data:', int(num_11))

                        gt['num_cn'] = torch.cat((num_11, torch.tensor([0]).to(device)), dim=0)
                        gt_update = bound_fn(coords1_cn, coords2_cn)

                        costate = gt_update['boundary_values_cn']
                        cn_index = costate.shape[0] // 2
                        costate1 = costate[:cn_index]
                        costate2 = costate[cn_index:]
                        state = coords_cn[:cn_index, 1:-1]
                        dirichlet_mask_cn = gt_update['dirichlet_mask_cn']

                        X_pred_T = state[dirichlet_mask_cn].detach().cpu().numpy().reshape(numcostate, 8).T
                        A1_pred_T = costate1[dirichlet_mask_cn].detach().cpu().numpy().reshape(numcostate, 8).T
                        A2_pred_T = costate2[dirichlet_mask_cn].detach().cpu().numpy().reshape(numcostate, 8).T
                        A_pred_T = np.vstack((A1_pred_T, A2_pred_T))
                        data_ode = {}
                        start_time = time.time()

                        """
                        use for trajectory verification
                        """
                        p_num = numcostate*(numrollout+1)
                        dx1_tmp = ((state[:p_num, 0:1] + 1) * (90 - 15) / 2 + 15).reshape(31, numcostate).T
                        dy1_tmp = ((state[:p_num, 1:2] + 1) * (38 - 32) / 2 + 32).reshape(31, numcostate).T
                        theta1_tmp = ((state[:p_num, 2:3] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)).reshape(31, numcostate).T
                        v1_tmp = ((state[:p_num, 3:4] + 1) * (25 - 18) / 2 + 18).reshape(31, numcostate).T
                        dx2_tmp = ((state[:p_num, 4:5] + 1) * (90 - 15) / 2 + 15).reshape(31, numcostate).T
                        dy2_tmp = ((state[:p_num, 5:6] + 1) * (38 - 32) / 2 + 32).reshape(31, numcostate).T
                        theta2_tmp = ((state[:p_num, 6:7] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)).reshape(31, numcostate).T
                        v2_tmp = ((state[:p_num, 7:8] + 1) * (25 - 18) / 2 + 18).reshape(31, numcostate).T
                        dx1_tmp = dx1_tmp.detach().cpu().numpy()
                        dy1_tmp = dy1_tmp.detach().cpu().numpy()
                        theta1_tmp = theta1_tmp.detach().cpu().numpy()
                        v1_tmp = v1_tmp.detach().cpu().numpy()
                        dx2_tmp = dx2_tmp.detach().cpu().numpy()
                        dy2_tmp = dy2_tmp.detach().cpu().numpy()
                        theta2_tmp = theta2_tmp.detach().cpu().numpy()
                        v2_tmp = v2_tmp.detach().cpu().numpy()

                        theta1_tmp_cos = v1_tmp*np.cos(theta1_tmp)
                        theta1_tmp_sin = v1_tmp*np.sin(theta1_tmp)
                        theta2_tmp_cos = v2_tmp*np.cos(theta2_tmp)
                        theta2_tmp_sin = v2_tmp*np.sin(theta2_tmp)

                        for idx in range(0, numcostate):
                            if 0 <= idx < numcostate:
                                mu1, mu2 = 1, 1

                            dx1 = (X_pred_T[0, idx] + 1) * (90 - 15) / 2 + 15
                            dy1 = (X_pred_T[1, idx] + 1) * (38 - 32) / 2 + 32
                            theta1 = (X_pred_T[2, idx] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
                            v1 = (X_pred_T[3, idx] + 1) * (25 - 18) / 2 + 18
                            dx2 = (X_pred_T[4, idx] + 1) * (90 - 15) / 2 + 15
                            dy2 = (X_pred_T[5, idx] + 1) * (38 - 32) / 2 + 32
                            theta2 = (X_pred_T[6, idx] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
                            v2 = (X_pred_T[7, idx] + 1) * (25 - 18) / 2 + 18

                            xT = np.vstack((dx1, dy1, theta1, v1, dx2, dy2, theta2, v2)).reshape(-1, 1)
                            yT = A_pred_T[:, idx].reshape(-1, 1)

                            X_aug = np.vstack((xT, yT)).reshape(-1)
                            t_eval = np.linspace(0.0, 3.0, num=numrollout+1)
                            t_span = np.array([t_eval[-1], 0.0])
                            t_eval = t_eval[::-1]

                            # solve final value problem
                            SOL = solve_ivp(problem.v_dynamics, t_span, X_aug, method='RK45',
                                            t_eval=t_eval, args=(model, mu1, mu2), rtol=1e-03)
                            # SOL = solve_ivp(problem.v_dynamics, t_span, X_aug, method='DOP853',
                            #                 t_eval=t_eval, args=(model, theta1, theta2), rtol=1e-03)
                            A_ivp = SOL.y
                            t_ivp = SOL.t.reshape(1, -1)
                            if 'A' in data_ode.keys():
                                data_ode['A'] = np.hstack((data_ode['A'], np.flip(A_ivp[8:], axis=1)))
                                data_ode['t'] = np.hstack((data_ode['t'], np.flip(t_ivp, axis=1)))
                                data_ode['X'] = np.hstack((data_ode['X'], np.flip(A_ivp[:8], axis=1)))
                            else:
                                data_ode['A'] = np.flip(A_ivp[8:], axis=1)
                                data_ode['t'] = np.flip(t_ivp, axis=1)
                                data_ode['X'] = np.flip(A_ivp[:8], axis=1)

                        """
                        use for trajectory verification
                        """
                        dx1_test = data_ode['X'][0, :].reshape(numcostate, 31)
                        dy1_test = data_ode['X'][1, :].reshape(numcostate, 31)
                        theta1_test = data_ode['X'][2, :].reshape(numcostate, 31)
                        v1_test = data_ode['X'][3, :].reshape(numcostate, 31)
                        dx2_test = data_ode['X'][4, :].reshape(numcostate, 31)
                        dy2_test = data_ode['X'][5, :].reshape(numcostate, 31)
                        theta2_test = data_ode['X'][6, :].reshape(numcostate, 31)
                        v2_test = data_ode['X'][7, :].reshape(numcostate, 31)

                        final_time = time.time() - start_time
                        print('ode cost time:', final_time)

                        data_ode.update({'t0': data_ode['t']})
                        idx0 = np.nonzero(np.equal(data_ode.pop('t0'), 0))[1]
                        A1_11 = np.empty((0, 8))
                        A2_11 = np.empty((0, 8))
                        for idx in range(numrollout+1):
                            A1_11 = np.vstack((A1_11, data_ode['A'][:8, idx0 + idx][:, :numcostate].T))
                            A2_11 = np.vstack((A2_11, data_ode['A'][8:, idx0 + idx][:, :numcostate].T))

                        A1 = torch.tensor(A1_11, dtype=torch.float32).unsqueeze(0).to(device)
                        A2 = torch.tensor(A2_11, dtype=torch.float32).unsqueeze(0).to(device)

                        X1_11 = np.empty((0, 4))
                        X2_11 = np.empty((0, 4))
                        T_11 = np.empty((0, 1))
                        for idx in range(numrollout+1):
                            X1_11 = np.vstack((X1_11, data_ode['X'][:4, idx0 + idx][:, :numcostate].T))
                            X2_11 = np.vstack((X2_11, data_ode['X'][4:, idx0 + idx][:, :numcostate].T))
                            T_11 = np.vstack((T_11, data_ode['t'][:, idx0 + idx][:, :numcostate].T))

                        X1 = torch.tensor(X1_11, dtype=torch.float32).to(device)
                        X2 = torch.tensor(X2_11, dtype=torch.float32).to(device)
                        T = torch.flip(torch.tensor(T_11, dtype=torch.float32).to(device), [0])
                        label1 = torch.zeros(N_sample, 1).to(device)
                        label2 = torch.ones(N_sample, 1).to(device)

                        coords1_cn = torch.cat((2.0 * (X1[:, 0:1] - 15) / (90 - 15) - 1,
                                                2.0 * (X1[:, 1:2] - 32) / (38 - 32) - 1,
                                                2.0 * (X1[:, 2:3] - (-0.15)) / (0.18 - (-0.15)) - 1,
                                                2.0 * (X1[:, 3:4] - 18) / (25 - 18) - 1,
                                                2.0 * (X2[:, 0:1] - 15) / (90 - 15) - 1,
                                                2.0 * (X2[:, 1:2] - 32) / (38 - 32) - 1,
                                                2.0 * (X2[:, 2:3] - (-0.15)) / (0.18 - (-0.15)) - 1,
                                                2.0 * (X2[:, 3:4] - 18) / (25 - 18) - 1), dim=1)
                        coords2_cn = torch.cat((coords1_cn[:, 4:], coords1_cn[:, :4]), dim=1)
                        coords1_cn = torch.cat((T, coords1_cn, label1), dim=1)
                        coords2_cn = torch.cat((T, coords2_cn, label2), dim=1)
                        coords_cn = torch.cat((coords1_cn, coords2_cn), dim=0)

                        gt['costate_gt'] = torch.cat((A1, A2), dim=1)
                        model_input['coords_cn'] = coords_cn.unsqueeze(0)
                        V_cn = value_fn(model_input, gt)
                        gt['value_gt'] = torch.cat((V_cn[:, :N_sample, :],
                                                    V_cn[:, N_sample:, :]), dim=1)

                        # remove the state beyond the space
                        N_shape = int(2*(num_11))
                        model_input['coords_cn'] = model_input['coords_cn'][torch.cat([coords_mask]*10, dim=2)].reshape(N_shape, 10).unsqueeze(0)
                        gt['costate_gt'] = gt['costate_gt'][torch.cat([coords_mask]*8, dim=2)].reshape(N_shape, 8).unsqueeze(0)
                        gt['value_gt'] = gt['value_gt'][coords_mask].reshape(N_shape, 1).unsqueeze(0)

                        inputfun1_add_11 = model_input['input_fun'][:, 0, :].repeat(1, N_sample, 1).squeeze(0)
                        inputfun2_add_11 = model_input['input_fun'][:, numpoints, :].repeat(1, N_sample, 1).squeeze(0)
                        inputfun1_add = inputfun1_add_11
                        inputfun2_add = inputfun2_add_11

                        cut_index = coords_mask.shape[1]//2
                        input_mask1 = torch.cat([coords_mask]*30, dim=2)[:, :cut_index, :]
                        input_mask2 = torch.cat([coords_mask]*30, dim=2)[:, cut_index:, :]
                        inputfun1_add = inputfun1_add.unsqueeze(0)[input_mask1].reshape(N_shape//2, 30)
                        inputfun2_add = inputfun2_add.unsqueeze(0)[input_mask2].reshape(N_shape//2, 30)

                        inputfun1_pre = model_input['input_fun'][:, :numpoints, :].squeeze(0)
                        inputfun2_pre = model_input['input_fun'][:, numpoints:, :].squeeze(0)
                        inputfun_vn = torch.cat((inputfun1_add, inputfun1_pre,
                                                 inputfun2_add, inputfun2_pre), dim=0)
                        model_input['input_fun'] = inputfun_vn.unsqueeze(0)

                        N_sample_new = model_input['coords_cn'].shape[1] // 2

                        coords1_cn = model_input['coords_cn'][:, :N_sample_new, :-1].squeeze(0)
                        coords2_cn = model_input['coords_cn'][:, N_sample_new:, :-1].squeeze(0)
                        coords1_vn = model_input['coords_vn'][:, :numpoints, :].squeeze(0)
                        coords2_vn = model_input['coords_vn'][:, numpoints:, :].squeeze(0)
                        coords_vn = torch.cat((coords1_cn, coords1_vn,
                                               coords2_cn, coords2_vn), dim=0)
                        model_input['coords_vn'] = coords_vn.unsqueeze(0)

                        costate_data['coords_cn'] = model_input['coords_cn']
                        costate_data['coords_vn'] = model_input['coords_vn']
                        costate_data['input_fun'] = model_input['input_fun']
                        costate_data['costate_gt'] = gt['costate_gt']
                        costate_data['value_gt'] = gt['value_gt']
                        costate_data['num_cn'] = gt['num_cn']

                        # don't need this term
                        # costate_data['boundary_values_cn'] = gt['boundary_values_cn']
                        # costate_data['dirichlet_mask_cn'] = gt['dirichlet_mask_cn']
                    else:
                        model_input['coords_cn'] = costate_data['coords_cn']
                        model_input['coords_vn'] = costate_data['coords_vn']
                        model_input['input_fun'] = costate_data['input_fun']
                        gt['costate_gt'] = costate_data['costate_gt']
                        gt['value_gt'] = costate_data['value_gt']
                        gt['num_cn'] = costate_data['num_cn']

                    num_gradient = 1  # 5000

                    for _ in range(num_gradient):
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)

                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            if loss_name == 'weight':
                                if loss == 1:
                                    hji_weight = 0
                                else:
                                    hji_weight = loss
                                continue
                            if loss_name == 'dirichlet_vn':
                                bcs_loss_hji = loss.mean()
                                single_loss = bcs_loss_hji
                            if loss_name == 'dirichlet_cn':
                                bcs_loss_costate = loss.mean()
                                single_loss = bcs_loss_costate
                            if loss_name == 'costate_difference_vn':
                                loss_diff_vn = loss.mean()
                                single_loss = loss_diff_vn
                            if loss_name == 'costate_difference_cn':
                                loss_diff_cn = loss.mean()
                                single_loss = loss_diff_cn
                            if loss_name == 'value_difference':
                                value_diff = loss.mean()
                                single_loss = value_diff
                            else:
                                single_loss = loss.mean()

                            if loss_schedules is not None and loss_name in loss_schedules:
                                writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps),
                                                  total_steps)
                                single_loss *= loss_schedules[loss_name](total_steps)

                            writer.add_scalar(loss_name, single_loss, total_steps)
                            train_loss += single_loss

                        train_losses.append(train_loss.item())
                        bcs_losses_hji.append(bcs_loss_hji.item())
                        bcs_losses_costate.append(bcs_loss_costate.item())
                        losses_diff_vn.append(loss_diff_vn.item())
                        losses_diff_cn.append(loss_diff_cn.item())
                        values_diff.append(value_diff.item())
                        HJI_weight.append(hji_weight)
                        writer.add_scalar("total_train_loss", train_loss, total_steps)

                        # if not total_steps % steps_til_summary:
                        if not total_steps % 10:
                            torch.save(model.state_dict(),
                                       os.path.join(checkpoints_dir, 'model_current.pth'))
                            # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                        if not use_lbfgs:
                            optim.zero_grad()
                            train_loss.backward()

                            if clip_grad:
                                if isinstance(clip_grad, bool):
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                            optim.step()

                        scheduler.step(train_loss)

                        lr_scheduler = optim.state_dict()['param_groups'][0]['lr']
                        LR.append(lr_scheduler)

            pbar.update(1)

            if counter == 0:
                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.3f, bcs loss hji %0.2f, bcs loss costate %0.2f, loss diff_cn %0.2f, hji weight %0.2f, lr %0.6f"
                                % (epoch, train_loss, bcs_loss_hji, bcs_loss_costate, loss_diff_cn, hji_weight, lr_scheduler))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

            else:
                if not total_steps % 10:
                    tqdm.write("Epoch %d, Total loss %0.3f, bcs loss hji %0.2f, bcs loss costate %0.2f, loss diff_cn %0.2f, hji weight %0.2f, lr %0.6f"
                        % (epoch, train_loss, bcs_loss_hji, bcs_loss_costate, loss_diff_cn, hji_weight, lr_scheduler))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

            total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'bcs_losses_hji_final.txt'),
                   np.array(bcs_losses_hji))
        np.savetxt(os.path.join(checkpoints_dir, 'bcs_losses_costate_final.txt'),
                   np.array(bcs_losses_costate))
        np.savetxt(os.path.join(checkpoints_dir, 'losses_diff_vn_final.txt'),
                   np.array(losses_diff_vn))
        np.savetxt(os.path.join(checkpoints_dir, 'losses_diff_cn_final.txt'),
                   np.array(losses_diff_cn))
        np.savetxt(os.path.join(checkpoints_dir, 'values_diff_final.txt'),
                   np.array(values_diff))
        np.savetxt(os.path.join(checkpoints_dir, 'hji_weight_final.txt'),
                   np.array(HJI_weight))
        np.savetxt(os.path.join(checkpoints_dir, 'learning rate.txt'),
                   np.array(LR))

class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)

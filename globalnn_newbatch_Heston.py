'''dM = dPhi*dW+Psi_2*(dW^2-dt)'''

# import matplotlib.pyplot as plt
from generals import *
import numpy as np
import os
import torch
import torch.nn as nn
import time

def value2nn(features, multiplier, old):
    if old == True:
        conti = model_conti_old(features)
        Psi = model_mg_old(features)
    else:
        conti = model_conti(features)
        Psi = model_mg(features)
    dM = torch.sum(Psi*multiplier, 1, False) 
    return conti[:, 0], dM

def model2nn_dict(pathname, action, old):
    if old == True:
        model_conti_old.load_state_dict(model_conti.state_dict())
        model_mg_old.load_state_dict(model_mg.state_dict())
    else:
        if action == 'save':
            torch.save(model_conti.state_dict(), pathname+'_Conti')
            torch.save(model_mg.state_dict(), pathname+'_Mg')
        if action == 'load':
            model_conti.load_state_dict(torch.load(pathname+'_Conti'))
            model_mg.load_state_dict(torch.load(pathname+'_Mg'))


def stopping(X, Exe, N, updates, valuenn, old, **kwargs):    
    goal = torch.zeros((N, kwargs['total_step']),  device=my_device)
    cash_flow, upper_bound = (torch.clone(Exe[:, -1]) for _ in range(2))

    if updates != 0:
        for i in range(kwargs['N_step']-1,-1,-1): #i=num_stpe-1,...,1
            for j in range(kwargs['sub_step']-1,-1,-1): 
                step = i*kwargs['sub_step']+j
                cash_flow *= kwargs['discount']
                goal[:, step] = cash_flow
                conti, mg_pred = valuenn(X[:, step, 0:2*kwargs['N_stock']+1], 
                                         X[:, step, 2*kwargs['N_stock']+1::], old)
                mg_pred = mg_pred.detach()
                upper_bound = upper_bound*kwargs['discount'] - mg_pred
                cash_flow -= mg_pred
            if i > 0:
                cash_flow, upper_bound = decision_backward(
                    conti.detach() , Exe[:, i], cash_flow, upper_bound)
            else:
                cash_flow, upper_bound = decision_backward(
                    np.inf , Exe[:, i], cash_flow, upper_bound)
            print('Lower and upper bound at t = {}, sub = {} is {} and {}.'
                    .format(i, j, torch.mean(cash_flow), torch.mean(upper_bound)))
    else:
        for i in range(kwargs['total_step']-1, -1, -1):
            goal[:, i] = cash_flow*(kwargs['discount']**(kwargs['total_step']-i))
    return goal.reshape((N*kwargs['total_step'])), torch.mean(upper_bound)-torch.mean(cash_flow)                       


def Training(X_val, Exe_val, pathname, S_mean, S_std, V_mean, V_std, valuenn, modelnn_dict, **kwargs):  
    step_batch = np.tile(np.arange(kwargs['total_step'])*dt,(kwargs['batch_size'],)
            ).reshape((kwargs['batch_size'], kwargs['total_step'], 1))
    step_batch = torch.from_numpy(step_batch).float().to(my_device)
    Val_loss, Diff_val = [], []
    updates, patience = 0, 0
    while patience < kwargs['patience']:
        modelnn_dict(pathname, 'load', True)
        for _ in range(kwargs['N_batch']):
            Stock, Vol, dW_S, dW_V, Exe, _, _, _, _ = paths_Heston(
                my_device, False, kwargs['total_step'], kwargs['batch_size'], 
                kwargs['S0_batch'], kwargs['V0_batch'], S_mean, S_std, V_mean, V_std, **kwargs)
            X = torch.cat((step_batch, Stock[:,:-1,:], Vol[:,:-1,:], dW_S, dW_S**2-1,
                           dW_V, dW_V**2-1), dim=-1)
            with torch.no_grad():
                Y, _ = stopping(X, Exe, kwargs['batch_size'], updates, valuenn, True, **kwargs)
            # True indicates we use the old (just trained in the previous step) network to 
            # gengerate target values, to ensure for different batches target values are from
            # the same network, not the one getting trained 
            X = X.reshape(kwargs['batch_size']*kwargs['total_step'], 1+6*kwargs['N_stock'])
            batch_loss = []
            for _ in range(kwargs['N_epoch']):               
                conti, dM = valuenn(X[:, 0:2*kwargs['N_stock']+1], X[:, 2*kwargs['N_stock']+1::], False)
                cash_flow = conti + dM
                opt.zero_grad()
                loss = loss_f(cash_flow, Y) 
                loss.backward()
                opt.step()
                batch_loss.append(loss.detach())
            print('batch loss: {}.'.format(batch_loss))
        updates+=1
        with torch.no_grad():
            Y_val, diff_val = stopping(X_val, Exe_val, kwargs['N_val'], updates, valuenn, False, **kwargs)
            # we use the just trained one to check the stopping criteria
            Diff_val.append(diff_val)
            feature = X_val.reshape(kwargs['N_val']*kwargs['total_step'], 1+6*kwargs['N_stock'])
            conti, dM = valuenn(feature[:, 0:2*kwargs['N_stock']+1], feature[:, 2*kwargs['N_stock']+1::], False)
            cash_flow = conti + dM
            loss = loss_f(cash_flow, Y_val)
            Val_loss.append(loss.detach())
            print('validation loss: {}.'.format(loss))

        if Val_loss[-1] == min(Val_loss) or Diff_val[-1] == min(Diff_val):
            modelnn_dict(pathname, 'save', False)
            patience = 0
        else:
            patience += 1
         
    return updates


def Testing(pathname, device, S_mean, S_std, V_mean, V_std, valuenn, modelnn_dict, **kwargs):
    M_test, cash_flow_test = (torch.zeros((kwargs['N_test']), device=device) for _ in range(2))
    Stock, Vol, dW_S, dW_V, exe_now, _, _, _, _ = paths_Heston(
        device, False, kwargs['sub_step'], kwargs['N_test'], kwargs['S0_test'],
        kwargs['V0_test'], S_mean[0: kwargs['sub_step'], :], S_std[0: kwargs['sub_step'], :], 
        V_mean[0: kwargs['sub_step'], :], V_std[0: kwargs['sub_step'], :], **kwargs)  
    upper_bound_test =  torch.clone(exe_now[:, 0])
    exe_now[:, 0] = -np.inf
    modelnn_dict(pathname, 'load', False)
    for i in range(0, kwargs['N_step']):   
        for j in range(kwargs['sub_step']):    
            step_train = torch.ones((kwargs['N_test'], 1), device = my_device)*kwargs['dt']*(i*kwargs['sub_step']+j)
            with torch.no_grad():
                conti, M_incre = valuenn(torch.cat((step_train, Stock[:, j, :], Vol[:, j, :]),dim=1), 
                                        torch.cat((dW_S[:, j, :], dW_S[:, j, :]**2-1, 
                                                   dW_V[:, j, :], dW_V[:, j, :]**2-1), dim=1), False)
            if j == 0:
                cash_flow_test,upper_bound_test=decision_forward(
                    i, conti.detach(), kwargs['discount']**kwargs['sub_step'], exe_now[:, 0],
                    cash_flow_test, upper_bound_test, M_test)
            M_test += M_incre.detach()*(kwargs['discount']**(i*kwargs['sub_step']+j))
            print('The lower and upper bound at step {} are {} and {}'.format(
                i, torch.mean(cash_flow_test), torch.mean(upper_bound_test)))
        if i < kwargs['N_step']-1:
            Stock, Vol, dW_S, dW_V, exe_now, _, _, _, _= paths_Heston(
                device, False, kwargs['sub_step'], kwargs['N_test'], Stock[:, -1, :], 
                Vol[:, -1, :], S_mean[(i+1)*kwargs['sub_step']:(i+2)*kwargs['sub_step'], :], 
                S_std[(i+1)*kwargs['sub_step']:(i+2)*kwargs['sub_step'], :],
                V_mean[(i+1)*kwargs['sub_step']:(i+2)*kwargs['sub_step'], :], 
                V_std[(i+1)*kwargs['sub_step']:(i+2)*kwargs['sub_step'], :],  **kwargs)    
    cash_flow_test, upper_bound_test = decision_forward(
        kwargs['N_step'], -np.inf, kwargs['discount']**kwargs['sub_step'], exe_now[:, 1], 
        cash_flow_test, upper_bound_test, M_test)
    print('The lower and upper bound at step {} are {} and {}'.format(
        i+1, torch.mean(cash_flow_test), torch.mean(upper_bound_test)))
    return torch.mean(cash_flow_test).item(), torch.mean(upper_bound_test).item()
    

# parameters for 1D American Option
np.random.seed(92)
torch.manual_seed(92)
generator = torch.Generator().manual_seed(torch.seed())
use_cuda = torch.cuda.is_available()
my_device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.cuda.memory._record_memory_history()

S0, V0, T = 100, 0.01, 1
my_option = {'N_step': 10, 'N_stock': 1, 'strike': 100, 'r': torch.tensor(0.1, device = my_device), 'xi': 0.2, 'lam': 2, 
             'sigma2': 0.01, 'rho1': torch.tensor(-0.3, device = my_device), 'option_type': "put", 
              'option_name': 'Heston'}

my_training = {'N_val': int(1e4), 'N_test': int(1e5), 'batch_size': int(1e4), 
               'N_neuron_1': [50, 50, 50], 'N_neuron_2': [50, 50, 50], 'patience': 5, 
               'N_batch': 5, 'N_epoch': 10}

Num_Training = 1
lrs = [0.01]

loss_f = nn.MSELoss()
for sub_steps in [3]:
    my_training['sub_step'] = sub_steps
    my_option, my_training = prep_kwargs_Heston(S0, V0, T, my_device, 'Global_NewBatch', my_option, my_training)
    dt = T/(my_training['total_step'])

    for lr in lrs:
        my_training['lr'] = lr
        path_root = ''
        path = "e4e5_{}_{}{}_{}p{}e{}b_{}sub".format(
            lr, my_training['N_neuron_1'], my_training['N_neuron_2'],  
            my_training['patience'], my_training['N_epoch'], my_training['N_batch'],
            my_training['sub_step']).replace(",", "")
        N_varibels = num_free_variable(my_option['N_stock']+1,  my_training['N_neuron_1'], 1) \
            + num_free_variable( my_option['N_stock']+1, my_training['N_neuron_2'],  2*my_option['N_stock'])
        try:
            os.mkdir(path_root+path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)
        LB, UB, Train_time, Test_time, Updates = (np.zeros((Num_Training,)) for _ in range(5))
        for ite in range(Num_Training):
            print('Trial {}'.format(ite+1))
            model_loc = path+'/Trial_'+str(ite+1)
            model_loc = path+'/Trial_'+str(ite+1)
            Stock, Vol, dW_S, dW_V, Exe, S_mean, S_std, V_mean, V_std = paths_Heston(
                my_device, True, my_training['total_step'], my_training['N_val'], 
                my_training['S0_val'], my_training['V0_val'], None, None, 
                None, None, **my_option, **my_training)
            StandCoef_Heston('write', my_option['N_stock'], my_training['total_step'], path_root, 
                    model_loc, my_device, S_mean, S_std, V_mean, V_std )
            
            step_train = np.tile(np.arange(my_training['total_step'])*dt,(my_training['N_val'],)
                    ).reshape((my_training['N_val'], my_training['total_step'], 1))
            step_train = torch.from_numpy(step_train).float().to(my_device)
            X = torch.cat((step_train, Stock[:,:-1,:], Vol[:,:-1,:], dW_S, dW_S**2-1, dW_V, dW_V**2-1), dim=-1)
            del step_train, Stock, Vol, dW_S, dW_V
            torch.cuda.empty_cache()
            # S_mean, S_std, V_mean, V_std  = StandCoef_Heston('read', my_option['N_stock'], my_training['total_step'], 
            #     path_root, model_loc, my_device) 
            
            start1 = time.time()
            model_conti = Network(2*my_option['N_stock']+1, my_training['N_neuron_1'], 1)
            model_mg = Network(2*my_option['N_stock']+1, my_training['N_neuron_2'], 4*my_option['N_stock'])
            model_conti_old = Network(2*my_option['N_stock']+1, my_training['N_neuron_1'], 1)
            model_mg_old = Network(2*my_option['N_stock']+1, my_training['N_neuron_2'], 4*my_option['N_stock'])
            if use_cuda:
                model_conti.cuda()
                model_mg.cuda()
                model_conti_old.cuda()
                model_mg_old.cuda()
            opt = torch.optim.Adam(list(model_conti.parameters()) + 
                                list(model_mg.parameters()), lr)
            Updates[ite] = Training(
                X, Exe, path_root+model_loc, S_mean, S_std, V_mean, V_std, value2nn, model2nn_dict, **my_option, **my_training)
            end1 = time.time()
            LB[ite], UB[ite] = Testing(path_root+model_loc, my_device, S_mean, S_std, V_mean, V_std,
                                        value2nn, model2nn_dict, **my_option, **my_training)
        
            Train_time[ite] = end1-start1
            Test_time[ite] = time.time()-end1
            my_training['N_path'] = Updates*my_training['batch_size']*my_training['N_batch']+my_training['N_val']
        write_results(LB, UB, Train_time, Test_time, Updates, path_root, path, 
                        N_varibels, 'Global_NewBatch', **my_training)
        
# torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
# N_points = int(1e4)
# model_plot = Network(my_option['N_stock'], my_training['N_neuron'], 1)
# w_loss = 0.5
# plot_conti_deriv(my_training, T, my_option['N_step'], 
#                  my_option['N_stock'], model_plot, N_points, path_root, path, 
#                  'Trial_1', threeD = False)

# plot_deriv(path_root, my_training, N_points, my_option['N_stock'], 
#            my_option['N_step'], w_losses, model_plot, 'Trial_1')
'''
    This file corresponds to the Method I with warm-start in the paper where we generate
    all paths before training.
    can choose to train one or two networks at each time by setting TwoNN = False/True
'''
# import matplotlib.pyplot as plt
from generals import *
import numpy as np
import os
import torch
import torch.nn as nn
import time
# import matplotlib
# matplotlib.use('TKAgg')
# there is a package called torchplot, which can directly plot troch.tensor, but I cannot change the backend and the default one doesn't work
def value1nn(features, multiplier):
    outputs = model(features)
    dM = torch.sum(outputs[:, 1::]*multiplier, 1, False) 
    return outputs[:, 0], dM

def value2nn(features, multiplier):
    conti = model_conti(features)
    Psi = model_mg(features)
    dM = torch.sum(Psi*multiplier, 1, False) 
    return conti[:, 0], dM

def model2nn_dict(exe_step, sub_step, pathname, action):
    if action == 'save':
        torch.save(model_conti.state_dict(), pathname+'_Time' +
                    str(exe_step)+'_Sub'+str(sub_step)+'_Conti')
        torch.save(model_mg.state_dict(), pathname+'_Time' +
                    str(exe_step)+'_Sub'+str(sub_step)+'_Mg')
    if action == 'load':
        model_conti.load_state_dict(torch.load(
                pathname+'_Time'+str(exe_step)+'_Sub'+str(sub_step)+'_Conti'))
        model_mg.load_state_dict(torch.load(
                pathname+'_Time'+str(exe_step)+'_Sub'+str(sub_step)+'_Mg'))
    
def model1nn_dict(exe_step, sub_step, pathname, action):
    if action == 'save':
        torch.save(model.state_dict(), pathname+'_Time' +
                    str(exe_step)+'_Sub'+str(sub_step))
    if action == 'load':
        model.load_state_dict(torch.load(
                pathname+'_Time'+str(exe_step)+'_Sub'+str(sub_step)))
        
def decision_backward(ex_conti, ex_now, cash_flow, upper_bound):
    # always substract martingale increments, 
    mask = ((ex_conti<ex_now) & (ex_now>0))
    cash_flow[mask] = ex_now[mask]
    upper_bound[(ex_now>=upper_bound)]=ex_now[(ex_now>=upper_bound)]

    return cash_flow, upper_bound,

def decision_forward(ite, ex_conti, discount, exe_now, cash_flow, upper_bound, 
                     m_test):
    # cash_flow, m_test and upper_bound are always discounted back at 0
    # substep has been considered in discounting factor, discount here is discount_global*sub_step
    exe_now_at0 = (exe_now)*(discount**ite)-m_test
    upper_bound[upper_bound < exe_now_at0] = exe_now_at0[upper_bound < exe_now_at0]
    mask = ((ex_conti<exe_now) & (cash_flow==0))
    cash_flow[mask] = exe_now_at0[mask]
    
    return cash_flow, upper_bound
        
def train_model(valuenn, modelnn_dict, current_step, current_sub, X, Y, pathname, **kwargs):
    epoch, patience_now = 0, 0
    best_mse = 10**8
    while patience_now < kwargs['patience'] and (epoch < kwargs['max_epoch']):
        # training_loss = []
        train_feature, train_label, val_feature, val_label = random_split(
            X, Y, kwargs['N_train'], generator)
        for i in range(kwargs['N_batch']):
            ind_l, ind_r = i*kwargs['batch_size'], (i+1)*kwargs['batch_size']
            conti, dM = valuenn(train_feature[ind_l: ind_r, 0: kwargs['N_stock']],
                                  train_feature[ind_l: ind_r, kwargs['N_stock']::])
            cash_flow = conti + dM
            opt.zero_grad()
            loss = loss_f(cash_flow, train_label[ind_l: ind_r]) 
            loss.backward()
            opt.step()
            # training_loss.append(loss.detach())
        epoch += 1
        conti, dM = valuenn(val_feature[:, 0: kwargs['N_stock']],
                            val_feature[:, kwargs['N_stock']::])
        cash_flow = conti + dM
        loss = loss_f(cash_flow, val_label)
        # print('The valdation loss: {}.'.format(loss))

        val_loss = loss.detach()
        if val_loss < best_mse:
            best_mse = val_loss
            modelnn_dict(current_step, current_sub, pathname, 'save')
            patience_now = 0
        else:
            patience_now += 1
    modelnn_dict(current_step, current_sub, pathname, 'load')
    conti, dM = valuenn(X[:, 0: kwargs['N_stock']], X[:, kwargs['N_stock']::])        
    return conti.detach(), dM.detach(), epoch


def Training(Stock, dW, Exe, pathname, valuenn, modelnn_dict, **kwargs):
    cash_flow_train, upper_bound_train = (torch.clone(Exe[:, -1]) for _ in range(2))    
    Epochs = np.empty(kwargs['total_step'])
    for i in range(kwargs['N_step']-1, -1, -1):
        for j in range(kwargs['sub_step']-1, -1, -1):
            step = i*kwargs['sub_step']+j
            cash_flow_train *= kwargs['discount']
            X = torch.cat((Stock[:,step,:], dW[:, step, :], dW[:, step, :]**2-1), dim=1)
            conti, mg_pred, Epochs[step] = train_model(
                valuenn, modelnn_dict, i, j, X, cash_flow_train, pathname, **kwargs)
            upper_bound_train = upper_bound_train*kwargs['discount']-mg_pred
            cash_flow_train -= mg_pred
        # the update of upper bounds can be neglected as it does not contribute the training
        cash_flow_train, upper_bound_train = decision_backward(
            conti, Exe[:, i], cash_flow_train, upper_bound_train)
        # print('Lower and upper bound at t = {}, sub = {} is {} and {}.'
        #         .format(i, j, torch.mean(cash_flow_train),
        #                 torch.mean(upper_bound_train)))
    return Epochs


def Testing(pathname, device, S_mean, S_std, valuenn, modelnn_dict, **kwargs):
    M_test, cash_flow_test, upper_bound_test = (
        torch.zeros((kwargs['N_test']), device=device) for _ in range(3))
    Stock, dW, exe_now, _, _ = paths(
        device, False, kwargs['sub_step'], kwargs['N_test'], kwargs['S0_test'], 
        S_mean[0: kwargs['sub_step'], :], S_std[0: kwargs['sub_step'], :], **kwargs)    
    exe_now[:, 0] = -np.inf
    for i in range(0, kwargs['N_step']):  
        for j in range(kwargs['sub_step']):
            modelnn_dict(i, j, pathname, 'load')
            dWs = torch.cat((dW[:, j, :], dW[:, j, :]**2-1), dim=1)
            with torch.no_grad():
                conti, M_incre = valuenn(Stock[:, j], dWs) 
            if j == 0:
                cash_flow_test,upper_bound_test=decision_forward(
                    i, conti.detach(), kwargs['discount']**kwargs['sub_step'], exe_now[:, 0],
                    cash_flow_test, upper_bound_test, M_test)
            M_test += M_incre.detach()*(kwargs['discount']**(i*kwargs['sub_step']+j))
            # print('The lower and upper bound at step {} are {} and {}'.format(
            #     i, torch.mean(cash_flow_test), torch.mean(upper_bound_test)))
        if i < kwargs['N_step']-1:
            Stock, dW, exe_now, _, _ = paths(
                device, False, kwargs['sub_step'], kwargs['N_test'], Stock[:, -1, :], 
                S_mean[(i+1)*kwargs['sub_step']:(i+2)*kwargs['sub_step'], :], 
                S_std[(i+1)*kwargs['sub_step']:(i+2)*kwargs['sub_step'], :], **kwargs)    
    cash_flow_test, upper_bound_test = decision_forward(
        kwargs['N_step'], -np.inf, kwargs['discount']**kwargs['sub_step'], exe_now[:, 1], 
        cash_flow_test, upper_bound_test, M_test)
    # print('The lower and upper bound at step {} are {} and {}'.format(
    #     i+1, torch.mean(cash_flow_test), torch.mean(upper_bound_test)))
    return torch.mean(cash_flow_test).item(), torch.mean(upper_bound_test).item()


# parameters for 1D American Option
torch.manual_seed(92)
generator = torch.Generator().manual_seed(torch.seed())
use_cuda = torch.cuda.is_available()
my_device = torch.device("cuda:0" if use_cuda else "cpu")

# # 1D Put
# S0, r, div, sigma, T = 36, 0.06, 0, 0.2, 1
# my_option = {'N_step': 50, 'N_stock': 1, 'strike': 40, 'option_type': "put", 
#               'option_name': 'Put_1D50S'}

# 5D Max-Call
S0, r, div, sigma, T = 100, 0.05, 0.1, 0.2, 3
my_option = {'N_step': 9, 'N_stock': 5, 'strike': 100, 'option_type': "max_call",
            'option_name': 'MaxCall_5D9S'}

my_training = {'N_path': int(1e5), 'N_test': int(1e6), 'batch_size': int(1e4), 
               'N_neuron_1': [50, 25], 'N_neuron_2': [50, 50], 'val': 0.1,
               'patience': 5, 'max_epoch':100, 'TwoNN': True}
for sub_steps in [2]:
    my_training['sub_step'] = sub_steps
    my_option, my_training = prep_kwargs(S0, r, div, sigma, T, my_device, 'Multi', my_option, my_training)

    Num_Training = 10
    lrs = [0.005]
    loss_f = nn.MSELoss() #nn.L1Loss()
    for lr in lrs:
        my_training['lr'] = lr
        if my_training['TwoNN'] == True:
            path_root = ''
            path = "{}_{}{}_{}p_{}sub".format(
                lr, my_training['N_neuron_1'], my_training['N_neuron_2'], 
                my_training['patience'], 
                my_training['sub_step']).replace(",", "")
            N_varibels = num_free_variable(my_option['N_stock'],  my_training['N_neuron_1'], 1) \
                + num_free_variable( my_option['N_stock'],  my_training['N_neuron_2'], 2*my_option['N_stock'])
        else:
            path_root = ''
            path = "{}_{}_{}p_{}sub".format(
                lr, my_training['N_neuron_1'], my_training['patience'], 
                my_training['sub_step']).replace(",", "")
            N_varibels = num_free_variable(
        my_option['N_stock'],  my_training['N_neuron_1'], 2*my_option['N_stock']+1) 
        try:
            os.mkdir(path_root+path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)
    
        Epochs = np.zeros((Num_Training, my_training['total_step']))
        LB, UB, Train_time, Test_time = (np.zeros((Num_Training,)) for _ in range(4))
        for ite in range(Num_Training):
            print('Trial {}'.format(ite+1))
            model_loc = path+'/Trial_'+str(ite+1)
            Stock, dW, Exe, S_mean, S_std = paths(
                my_device, True, my_training['total_step'], my_training['N_path'], 
                my_training['S0_train'], None, None, **my_option, **my_training)
            StandCoef('write', my_option['N_stock'], my_training['total_step'], path_root, 
                    model_loc, my_device, S_mean, S_std)
            # S_mean, S_std = StandCoef('read', my_option['N_stock'], my_training['total_step'], 
            #     path_root, model_loc, my_device) 
            start1 = time.time()
            if my_training['TwoNN'] == True:
                model_conti = Network(my_option['N_stock'], my_training['N_neuron_1'], 1)
                model_mg = Network(my_option['N_stock'], my_training['N_neuron_2'], 2*my_option['N_stock'])
                if use_cuda:
                    model_conti.cuda()
                    model_mg.cuda()
                opt = torch.optim.Adam(list(model_conti.parameters()) + 
                            list(model_mg.parameters()), lr)
                Epochs[ite, :] = Training(Stock, dW, Exe, path_root+model_loc, value2nn, 
                                        model2nn_dict, **my_option, **my_training)
                end1 = time.time()
                LB[ite], UB[ite] = Testing(path_root+model_loc, my_device, S_mean, S_std, 
                                        value2nn, model2nn_dict, **my_option, **my_training)
            else:
                model = Network(my_option['N_stock'], my_training['N_neuron_1'], 2*my_option['N_stock']+1)
                if use_cuda:
                    model.cuda()
                opt = torch.optim.Adam(model.parameters(), lr)           
                Epochs[ite, :] = Training(Stock, dW, Exe, path_root+model_loc, value1nn, 
                                        model1nn_dict, **my_option, **my_training) 
                end1 = time.time()
                LB[ite], UB[ite] = Testing(path_root+model_loc, my_device, S_mean, S_std, 
                                        value1nn, model1nn_dict, **my_option, **my_training)      
            
            Train_time[ite] = end1-start1           
            Test_time[ite] = time.time()-end1

        write_results(LB, UB, Train_time, Test_time, np.mean(Epochs,axis=0), path_root, path, 
                        N_varibels*my_training['total_step'], 'Multi', **my_training)

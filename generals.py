'''Here we have all functions that are shared among all different versions of the main training and testing processes'''

import torch
import numpy as np
from torch.autograd import grad
import torch.nn as nn
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TKAgg')
import csv
import os.path

def payoffs(option_type, x, kappa):    
    if option_type == 'put':
       # the payoff function is (K-S_{t})^{+}, it can only be a 1-D case
       y = kappa-torch.squeeze(x, -1)

       y [y <= 0] = 0
       
    if option_type == 'max_call':
        x_max, _ = torch.max(x, -1)
        y = x_max - kappa
        y[y <= 0] = 0

    if option_type == 'geo_put':
        x_prod = torch.prod(x, -1)
        y = kappa-x_prod.reshape((-1, 1))
        y[y <= 0] = 0

    return y

class Network(nn.Module):
    def __init__(self, n_in, n_neu, n_out):
        super(Network, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(n_in, n_neu[0]), 
                                     nn.Softplus(),])
        for i in range(len(n_neu)-1):
            self.layers.append(nn.Linear(n_neu[i],n_neu[i+1]))
            self.layers.append(nn.Softplus())
        self.layers.append(nn.Linear(n_neu[-1], n_out))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x
        
#step_num and step_size have been treated due to substep, with 4 substep and n exericising points(not inclusing 0), step_num=4*n 
def paths(device, first, steps, N, S0, S_mean, S_std, N_stock, drift, sigma_dt, generator = None, **kwargs):
    S = torch.zeros((N, steps+1, N_stock), device=device)
    dW = torch.randn(size=(int(N), steps, N_stock), generator = generator, device = device)
    S[:, 0, :] = S0
    lnS = torch.log(S0)
    for step in range(steps):
        lnS += drift+sigma_dt*dW[:, step, :]
        S[:, step+1, :] = torch.exp(lnS)
    Exe_mask = np.arange(0, steps+1, kwargs['sub_step'])
    Exe = payoffs(kwargs['option_type'], S[:, Exe_mask, :], kwargs['strike'])
    if first == True:
        S_mean, S_std = S[:,:-1,:].mean(dim = 0), S[:,:-1,:].std(dim = 0)
        S_std[0,:] = 1
    S[:,:-1,:] = (S[:,:-1,:]-S_mean)/S_std
    return S, dW, Exe, S_mean, S_std

def paths_Heston(device, first, steps, N, S0, V0, S_mean, S_std, V_mean, V_std, N_stock, r, sigma2, lam_dt, xi_dt, dt, rho1, rho2, generator = None, **kwargs):
    S, V = (torch.zeros((N, steps+1, N_stock), device=device) for _ in range(2))
    dW_V = torch.randn(size=(int(N), steps, N_stock), generator = generator, device = device)
    dW_S = rho1*dW_V + rho2*torch.randn(size=(int(N), steps, N_stock), generator = generator, device = device)
    S[:, 0, :], V[:, 0, :] = S0, V0
    lnS = torch.log(S0)
    for i in range(steps):
        V[:, i+1, :] = V[:, i, :] + lam_dt*(sigma2 - V[:, i, :]) + xi_dt*torch.sqrt(V[:, i, :])*dW_V[:, i, :]
        V[:, i+1, :] [V[:, i+1, :] < 0] = 0
        lnS +=  (r - V[:, i, :]/2)*dt + torch.sqrt(V[:, i, :])*torch.sqrt(dt)*dW_S[:, i, :]
        S[:, i+1, :] = torch.exp(lnS)
    Exe_mask = np.arange(0, steps+1, kwargs['sub_step'])
    Exe = payoffs(kwargs['option_type'], S[:, Exe_mask, :], kwargs['strike'])
    if first == True:
        S_mean, S_std = S[:,:-1,:].mean(dim = 0), S[:,:-1,:].std(dim = 0)
        V_mean, V_std = V[:,:-1,:].mean(dim = 0), V[:,:-1,:].std(dim = 0)
        S_std[0,:], V_std[0,:] = 1, 1
    S[:,:-1,:] = (S[:,:-1,:]-S_mean)/S_std
    V[:,:-1,:] = (V[:,:-1,:]-V_mean)/V_std
    return S, V, dW_S, dW_V, Exe, S_mean, S_std, V_mean, V_std

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

def num_free_variable(inputs, neuron, outputs):
  layer = len(neuron)
  num_var = (inputs+1)*neuron[0] #the plus one is for the bias
  for i in range(layer-1):
    num_var += (neuron[i]+1)*neuron[i+1]
  num_var += (neuron[-1]+1)*outputs
  return num_var

def prep_kwargs(S0, r, div, sigma, T, my_device, version, my_option, my_training):
    my_training['total_step'] = my_training['sub_step']*my_option['N_step']
    dt = torch.tensor(T/my_training['total_step'], device = my_device)
    my_option['discount'] = torch.exp(-r*dt)
    my_option['drift'] = (r-div-sigma**2/2)*dt
    my_option['sigma_dt'] = sigma*torch.sqrt(dt)
    
    my_training['S0_test'] = S0*torch.ones((my_training['N_test'], my_option['N_stock']),
                                            device=my_device)
    if version == 'Multi':
        my_training['S0_train'] = S0*torch.ones((my_training['N_path'], my_option['N_stock']),
                                            device=my_device)
        my_training['N_train'] = int(my_training['N_path']*(1-my_training['val']))
        my_training['N_batch'] = int(my_training['N_train']/my_training['batch_size'])    
    
    if version == 'Global':
        my_training['S0_train'] = S0*torch.ones((my_training['N_path'], my_option['N_stock']),
                                            device=my_device)
        my_training['N_train'] = int(my_training['N_path']*(1-my_training['val'])*my_training['total_step'])
        my_training['N_batch'] = int(my_training['N_train']/my_training['batch_size'])
        my_option ['dt'] = dt
    
    if version == 'Global_NewBatch':
        my_training['S0_val'] = S0*torch.ones((my_training['N_val'], my_option['N_stock']),
                                    device=my_device)
        my_training['S0_batch'] = S0*torch.ones((my_training['batch_size'], my_option['N_stock']),
                                    device=my_device)
        my_option ['dt'] = dt
    return my_option, my_training

def prep_kwargs_Heston(S0, V0, T, my_device, version, my_option, my_training):
    my_training['total_step'] = my_training['sub_step']*my_option['N_step']
    dt = torch.tensor(T/my_training['total_step'], device = my_device)
    my_option ['dt'] = dt
    my_option['discount'] = torch.exp(-my_option['r']*dt)
    my_option['lam_dt'] = my_option['lam']*dt
    my_option['xi_dt'] = my_option['xi']*torch.sqrt(dt)
    my_option['rho2'] = torch.sqrt(1-my_option['rho1']**2) 
    
    my_training['S0_test'] = S0*torch.ones((my_training['N_test'], my_option['N_stock']),
                                            device=my_device)
    my_training['V0_test'] = V0*torch.ones((my_training['N_test'], my_option['N_stock']),
                                            device=my_device)
    if version == 'Multi':
        my_training['S0_train'] = S0*torch.ones((my_training['N_path'], my_option['N_stock']),
                                            device=my_device)
        my_training['V0_train'] = V0*torch.ones((my_training['N_path'], my_option['N_stock']),
                                            device=my_device)
        my_training['N_train'] = int(my_training['N_path']*(1-my_training['val']))
        my_training['N_batch'] = int(my_training['N_train']/my_training['batch_size'])    
    
    if version == 'Global':
        my_training['S0_train'] = S0*torch.ones((my_training['N_path'], my_option['N_stock']),
                                            device=my_device)
        my_training['V0_train'] = V0*torch.ones((my_training['N_path'], my_option['N_stock']),
                                            device=my_device)
        my_training['N_train'] = int(my_training['N_path']*(1-my_training['val'])*my_training['total_step'])
        my_training['N_batch'] = int(my_training['N_train']/my_training['batch_size'])
    
    if version == 'Global_NewBatch':
        my_training['S0_val'] = S0*torch.ones((my_training['N_val'], my_option['N_stock']),
                                    device=my_device)
        my_training['S0_batch'] = S0*torch.ones((my_training['batch_size'], my_option['N_stock']),
                                    device=my_device)
        my_training['V0_val'] = V0*torch.ones((my_training['N_val'], my_option['N_stock']),
                                    device=my_device)
        my_training['V0_batch'] = V0*torch.ones((my_training['batch_size'], my_option['N_stock']),
                                    device=my_device)
    return my_option, my_training

def plot_conti_deriv(para_training, T, N_step, N_stock, model_plot, N_points,
                      path_root, path, name, threeD = False):
    with open(path_root+'Normalisation.csv') as f:
        for row in f:
            if row.split(',')[0]==path+name:
                parameter=row.split(',')[1:N_step*N_stock*2+1]
        parameters = np.array(list(map(float,parameter)))
        std_save = parameters[N_step*N_stock: 2*N_step*N_stock]
        S_std = torch.from_numpy(
            std_save.reshape((N_step, N_stock))
            ).float()
    Stock = np.linspace(-3, 3, N_points) #98% of the data should be included.
    dt = T/(N_step*para_training['sub_step'])
    #the x-axis is based on the stock, the y-axis is based on the time in a 3D plot
    X1, Y1 = np.meshgrid(Stock, (np.arange(N_step)+1)*dt*para_training['sub_step']) 
    #X1, Y1 are for 3D continuation values
    X2, Y2 = np.meshgrid(Stock, (np.arange(N_step*para_training['sub_step'])+1)*dt)
    #X2, Y2 are for 3D martingale functions
    t_step = np.tile(np.arange(para_training['sub_step'])*dt*N_step/T, (N_points,1)) 
    X = np.zeros((para_training['sub_step'], N_points, 2))
    for i in range(para_training['sub_step']):
        X[i,:,:] = np.stack((X1[0,:], t_step[:,i]), axis=1)
    Z1, Z2 = np.zeros((N_step, N_points)), np.zeros((N_step*para_training['sub_step'], N_points))
            
    for i in range(N_step):
        for j in range(para_training['sub_step']):
            model_plot.load_state_dict(torch.load(path_root+path+'/'+name+'_Time'
                                                + str(i)+'_Sub'+str(j)))
            features_plot = torch.from_numpy(Stock.reshape((-1, 1))).float()
            features_plot.requires_grad = True
            conti_plot = model_plot(features_plot)[:, 0]
            delta_plot = grad(conti_plot, features_plot,
                              grad_outputs=torch.ones(conti_plot.shape))[0]
            Z1[i,:] = conti_plot.detach().numpy()
            Z2[i*para_training['sub_step']+j,:] = (delta_plot/S_std[i*para_training['sub_step']+j, :]).numpy().reshape((-1,))
    if threeD == True:   
        fig = plt.figure(figsize=(10,10))
        ax11 = fig.add_subplot(221)
        ax12 = fig.add_subplot(222)
    else:
        fig = plt.figure(figsize=(10,5))
        ax11 = fig.add_subplot(121)       
        ax12 = fig.add_subplot(122)
        
    ax11.plot(Stock,Z1[0,:], label='step=0')
    ax12.plot(Stock,Z2[0,:], label='step=0')
    for i in range(1,N_step-1,1):   
        ax11.plot(Stock,Z1[i,:])

    for i in range(1,N_step*para_training['sub_step']-1,1):   
        ax12.plot(Stock,Z2[i,:])

    ax11.plot(Stock,Z1[-1,:], label='step=49')
    ax12.plot(Stock,Z2[-1,:], label='step=49')

    ax11.set_xlabel('Standardised Stock Price')
    ax11.set_ylabel(r'$\Phi(S_t)$') 
    ax12.set_xlabel('Standardised Stock Price')
    ax12.set_ylabel(r'$\frac{\partial \Phi}{\partial s}$') 
    ax11.legend()
    ax12.legend()
    if threeD == True:
        ax21 = fig.add_subplot(223,projection='3d')
        surf = ax21.plot_surface(X1, Y1, Z1, cmap=cm.coolwarm)
        ax22 = fig.add_subplot(224, projection='3d')
        surf = ax22.plot_surface(X2, Y2, Z2, cmap=cm.coolwarm)
        ax21.view_init(10, 45)
        ax21.set_xlabel('Standardised Stock Price')
        ax21.set_ylabel('Time')
        ax21.set_zlabel(r'$\Phi(S_t)$') 
        ax22.view_init(10, 45)
        ax22.set_xlabel('Standardised Stock Price')
        ax22.set_ylabel('Time')
        ax22.set_zlabel(r'$\Psi(S_t)$')  

    plt.savefig(path_root+path+name+'.png')
    plt.show()
 

def plot_deriv(path_root, para_training, N_points, N_stock, N_step, w_losses, 
               model_plot, name):
    path = "e5e3e4_{}_{}_{}p_{}sub".format(
            para_training['lr'], para_training['N_neuron'],
            para_training['patience'], para_training['sub_step']).replace(",", "")
    fig, axs = plt.subplots(2, 2, figsize = (8, 8))
    fig.text(0.5, 0.04, 'Standardised Stock Price', ha='center')
    fig.text(0.04, 0.5, 'First Derivative of the continuation value function '+
            r'$\frac{\partial \Phi}{\partial s}$', va='center', rotation='vertical')
    axs = axs.ravel()
    N_plots_deriv = len(w_losses)
    
    for k in range(N_plots_deriv):  
        w_loss = w_losses[k]
        path_plot = path+'_{}l1'.format(w_loss)
        
        with open(path_root+'Normalisation.csv') as f:
            for row in f:
                if row.split(',')[0]==path_plot+name:
                    parameter=row.split(',')[1:N_step*N_stock*2+1]
            parameters = np.array(list(map(float,parameter)))
            std_save = parameters[N_step*N_stock: 2*N_step*N_stock]
            S_std = torch.from_numpy(
                std_save.reshape((N_step, N_stock))
                ).float()
        Stock = np.linspace(-3, 3, N_points) #98% of the data should be included.
        for i in range(para_training['sub_step']):
            Z= np.zeros((N_step*para_training['sub_step'], N_points))    
            for i in range(N_step):
                for j in range(para_training['sub_step']):
                    model_plot.load_state_dict(
                        torch.load(path_root + path_plot+'/'+name+'_Time'
                                    + str(i)+'_Sub'+str(j)))
                    features_plot = torch.from_numpy(Stock.reshape((-1, 1))).float()
                    features_plot.requires_grad = True
                    conti_plot = model_plot(features_plot)[:, 0]
                    delta_plot = grad(conti_plot, features_plot,
                                    grad_outputs=torch.ones(conti_plot.shape))[0]
                    Z[i*para_training['sub_step']+j,:] = (delta_plot/S_std[i*para_training['sub_step']+j, :]).numpy().reshape((-1,))
        
        
        axs[k].plot(Stock,Z[0,:], label='step=0')
        for i in range(1,N_step*para_training['sub_step']-1,1):   
            axs[k].plot(Stock,Z[i,:])
        axs[k].plot(Stock,Z[-1,:], label='step=49')

        axs[k].legend()
        axs[k].set_title('Weight = '+ str(w_loss))
    plt.savefig(path_root+'Derivatives'+path+name+'.png')
    plt.show()    

def StandCoef(action, N_stock, total_step, path_root, pathname, 
              my_device, S_train_mean = None, S_train_std = None):
    if action == 'write':
        mean_save = S_train_mean.cpu().numpy().reshape((total_step*N_stock,))
        std_save = S_train_std.cpu().numpy().reshape((total_step*N_stock,))
        parameters = list(np.concatenate((mean_save, std_save)))
        parameters.insert(0, pathname)
        parameters.append('end')
        with open(path_root+'Normalisation.csv', 'a', newline='') as f:  # 'a' means append
            result = csv.writer(f, delimiter=',')
            result.writerow(parameters)
    
    if action == 'read':
        with open(path_root+'Normalisation.csv') as f:
            for row in f:
                if row.split(',')[0]==pathname:
                    parameter=row.split(',')[1: total_step*N_stock*2+1]
        parameters = np.array(list(map(float,parameter)))
        mean_save = parameters[0: total_step*N_stock]
        std_save = parameters[total_step*N_stock: 2*total_step*N_stock]
        S_train_mean = torch.from_numpy(
            mean_save.reshape((total_step, N_stock))).float().to(my_device)
        S_train_std = torch.from_numpy(
            std_save.reshape(( total_step, N_stock))).float().to(my_device)
        return S_train_mean, S_train_std

def StandCoef_Heston(action, N_stock, total_step, path_root, pathname, 
              my_device, S_train_mean = None, S_train_std = None,
              V_train_mean = None, V_train_std = None):
    if action == 'write':
        S_mean_save = S_train_mean.cpu().numpy().reshape((total_step*N_stock,))
        S_std_save = S_train_std.cpu().numpy().reshape((total_step*N_stock,))
        V_mean_save = V_train_mean.cpu().numpy().reshape((total_step*N_stock,))
        V_std_save = V_train_std.cpu().numpy().reshape((total_step*N_stock,))
        parameters = list(np.concatenate((S_mean_save, S_std_save, V_mean_save, V_std_save)))
        parameters.insert(0, pathname)
        parameters.append('end')
        with open(path_root+'Normalisation.csv', 'a', newline='') as f:  # 'a' means append
            result = csv.writer(f, delimiter=',')
            result.writerow(parameters)
    
    if action == 'read':
        with open(path_root+'Normalisation.csv') as f:
            for row in f:
                if row.split(',')[0]==pathname:
                    parameter=row.split(',')[1: total_step*N_stock*4+1]
        parameters = np.array(list(map(float,parameter)))
        S_mean_save = parameters[0: total_step*N_stock]
        S_std_save = parameters[total_step*N_stock: 2*total_step*N_stock]
        V_mean_save = parameters[2*total_step*N_stock: 3*total_step*N_stock]
        V_std_save = parameters[3*total_step*N_stock: 4*total_step*N_stock]
        S_train_mean = torch.from_numpy(
            S_mean_save.reshape((total_step, N_stock))).float().to(my_device)
        S_train_std = torch.from_numpy(
            S_std_save.reshape(( total_step, N_stock))).float().to(my_device)
        V_train_mean = torch.from_numpy(
            V_mean_save.reshape((total_step, N_stock))).float().to(my_device)
        V_train_std = torch.from_numpy(
            V_std_save.reshape(( total_step, N_stock))).float().to(my_device)
        return S_train_mean, S_train_std, V_train_mean, V_train_std
    
def write_results(LB, UB, Training_Time, Test_time, Epochs, path_root, path, 
                  n_variables, version, **kwargs):
    Diff = UB-LB
    if os.path.isfile(path_root+'Price.csv') == False:
        with open(path_root+'Price.csv', 'w') as f:  # 'a' means append
            result = csv.writer(f, delimiter=',')
            result.writerow(['path_total', 'path_batch', 'path_val', 'lr', 'NN1', 'NN2', 
                             'n_variables', 'patience', 'n_epoch', 'n_batch', 'subsep', 
                             'Train Time', 'LB mean', 'LB std', 'UB mean', 'UB std', 
                             'D_mean', 'D_std'])
    
    if version == 'Multi':
        with open(path_root+'Price.csv', 'a', newline='') as f:  # 'a' means append
            result = csv.writer(f, delimiter=',')
            result.writerow(["{:e}".format(kwargs['N_path']), 
                                "{:e}".format(kwargs['batch_size']),
                                "{:e}".format(kwargs['N_path']*kwargs['val']), kwargs['lr'], 
                                kwargs['N_neuron_1'], kwargs['N_neuron_2'],
                                n_variables, kwargs['patience'], '', '', 
                                kwargs['sub_step'], 
                                int(np.mean(Training_Time)),
                                "%.4f" % np.mean(LB), "%.4f" % np.std(LB), 
                                "%.4f" % np.mean(UB), "%.4f" % np.std(UB), 
                                "%.4f" % np.mean(Diff), "%.4f" % np.std(Diff)])
    if version == 'Global':
        with open(path_root+'Price.csv', 'a', newline='') as f:  # 'a' means append
            result = csv.writer(f, delimiter=',')
            result.writerow(["{:e}".format(kwargs['N_path']), 
                                "{:e}".format(kwargs['batch_size']),
                                "{:e}".format(kwargs['N_path']*kwargs['val']), kwargs['lr'],
                                kwargs['N_neuron_1'], kwargs['N_neuron_2'],
                                n_variables, kwargs['patience'], kwargs['N_epoch'], '', 
                                kwargs['sub_step'], 
                                int(np.mean(Training_Time)),
                                "%.4f" % np.mean(LB), "%.4f" % np.std(LB), 
                                "%.4f" % np.mean(UB), "%.4f" % np.std(UB), 
                                "%.4f" % np.mean(Diff), "%.4f" % np.std(Diff)])
    if version == 'Global_NewBatch':
        with open(path_root+'Price.csv', 'a', newline='') as f:  # 'a' means append
            result = csv.writer(f, delimiter=',')
            result.writerow(["{:e}".format(np.mean(kwargs['N_path'])), 
                                "{:e}".format(kwargs['batch_size']),
                                "{:e}".format(kwargs['N_val']), kwargs['lr'], 
                                kwargs['N_neuron_1'], kwargs['N_neuron_2'],
                                n_variables, kwargs['patience'], kwargs['N_epoch'], 
                                kwargs['N_batch'], kwargs['sub_step'], 
                                int(np.mean(Training_Time)),
                                "%.4f" % np.mean(LB), "%.4f" % np.std(LB), 
                                "%.4f" % np.mean(UB), "%.4f" % np.std(UB), 
                                "%.4f" % np.mean(Diff), "%.4f" % np.std(Diff)])
    with open(path_root+'Price_detailed.csv', 'a', newline='') as f:  # 'a' means append
        result = csv.writer(f, delimiter=',')
        result.writerow([path, n_variables, 'free variables'])
        result.writerow(['Training Time', 'Test Time', 'LB_mean', 'LB_std',
                        'LB_opt', 'UB_mean', 'UB_std', 'UB_opt', 'D_mean',
                         'D_std', 'D_opt'])
        result.writerow([np.mean(Training_Time), np.mean(Test_time),
                         np.mean(LB), np.std(LB), np.max(LB),
                         np.mean(UB), np.std(UB), np.min(UB),
                        np.mean(Diff), np.std(Diff), np.min(Diff)])
        result.writerow(['The following are raw data of', 'Training Time',
                        'LB', 'UB', 'Diff', 'epochs at each exercise point'])
        result.writerow(Training_Time)
        result.writerow(LB)
        result.writerow(UB)
        result.writerow(Diff)
        result.writerow(Epochs)
    if version == 'Global_NewBatch':
        with open(path_root+'Price_detailed.csv', 'a', newline='') as f:  # 'a' means append
            result = csv.writer(f, delimiter=',')
            result.writerow(kwargs['N_path'])

def write_results_with_time(LB, UB, Training_Time, path_root, n_variables, version, **kwargs):
    Diff = UB-LB
    if os.path.isfile(path_root+'Price.csv') == False:
        with open(path_root+'Price.csv', 'w') as f:  # 'a' means append
            result = csv.writer(f, delimiter=',')
            result.writerow(['path_total', 'path_batch', 'path_test', 'lr', 'NN1', 'NN2', 
                             'n_variables', 'n_epoch', 'n_batch', 'subsep', 
                             'Training Time', 'LB', 'UB', 'Diff'])
    if version == 'Global':
        with open(path_root+'Price.csv', 'a', newline='') as f:  # 'a' means append
            result = csv.writer(f, delimiter=',')
            result.writerow(["{:e}".format(kwargs['N_path']), 
                            "{:e}".format(kwargs['batch_size']),
                            "{:e}".format(kwargs['N_test']),
                            kwargs['lr'], kwargs['N_neuron_1'], kwargs['N_neuron_2'],
                            n_variables, kwargs['N_epoch'], '', kwargs['sub_step']])
            result.writerow(Training_Time)
            result.writerow(LB)
            result.writerow(UB)
            result.writerow(Diff)
    if version == 'Global_NewBatch':
        with open(path_root+'Price.csv', 'a', newline='') as f:  # 'a' means append
            result = csv.writer(f, delimiter=',')
            result.writerow(["{:e}".format(np.mean(kwargs['N_path'])), 
                            "{:e}".format(kwargs['batch_size']),
                            "{:e}".format(kwargs['N_test']),
                            kwargs['lr'], kwargs['N_neuron_1'], kwargs['N_neuron_2'],
                            n_variables, kwargs['N_epoch'], 
                            kwargs['N_batch'], kwargs['sub_step']])
            result.writerow(Training_Time)
            result.writerow(LB)
            result.writerow(UB)
            result.writerow(Diff)

# I need pytorch 1.8 to work with my cuda10.2, which odes not have the built-in function torch.utils.data.random_split(dataset, lengths)
# I added the source code of that function first, but relised I don't need that many functions then wrote my own split function

def random_split(feature, label, train_size, generator):
    # Randomly split a dataset into non-overlapping new datasets of given lengths.
    whole_size = feature.shape[0]
    indices = torch.randperm(whole_size, generator=generator).tolist()  
    return feature[indices[0 : train_size]], label[indices[0 : train_size]], feature[indices[train_size::]], label[indices[train_size::]]


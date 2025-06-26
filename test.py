import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from pypower.loadcase import loadcase
from pypower.ppoption import ppoption
import time
import os
import scipy.io as scio

from env_37bus import IEEE37bus
from safenet import PrimalDualTrain,SafeNet

use_cuda=torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

parser=argparse.ArgumentParser(description='Primal-dual learning')
parser.add_argument('--env_name',default='case37',help='Please choose (case37)')
parser.add_argument('--train_flag',default=False,help='Training or not')
parser.add_argument('--trial_name',default='case37Adam_alpha_01_genco_1',help='The name to save pth and results')
parser.add_argument('--lineargrad',default=True,help='run linear primal dual based on pg and qg')
args=parser.parse_args()

## load the IEEE 37-bus case
case_name=args.env_name
casefile='test_'+case_name+'_ieee.py'
datacase=loadcase(casefile)


## load the environment
ppopt=ppoption()
ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=False, ENFORCE_Q_LIMS=False)
if args.env_name=='case37':
    env=IEEE37bus(datacase,ppopt)

## each controllable power injection has an agent
num_agent=env.gen_num
obs_dim=1
action_dim=1
hidden_dim=64

agent_list=[]
for i in range(num_agent):
    net=SafeNet(env=env,obs_dim=obs_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
    net.eval()
    agent=PrimalDualTrain(safenet=net)
    agent_list.append(agent)

## load the pretrain model
for i in range(num_agent):
    net_dict=torch.load('model_pth/{}_safenet_{}.pth'.format(args.trial_name,i))
    agent_list[i].safenet.load_state_dict(net_dict)

num_steps=96*50-1
mu_lower=[0]*36
mu_upper=[0]*36
###################################################################################
t_lower=np.zeros((36,),dtype=np.float32)
t_upper=np.zeros((36,),dtype=np.float32)

## start the real-time OPF #####
env.is_train=False
env.reset_demand()
v,pd,qd=env.reset(0)

v_rec = np.zeros((env.bus_num, num_steps), dtype=np.float32)
totalcost_rec=[]
demand_rec=[]
env.reset_demand()
print('Calculating the designed control!!')
t_start1=time.time()
for step in range(num_steps):
    action = []
    for i in range(num_agent):
        action_agent = agent_list[i].safenet.get_action(np.asarray([v[i]]), i, mu_upper, mu_lower,t_upper,t_lower,pd[i],qd[i])
        action.append(action_agent)

    action = np.asarray(action)
    next_v, VMag, totalcost,demand,pd,qd = env.step_realtime(action,step)
    # next_v, VMag, totalcost = env.step_Reward(action)

    v = np.copy(next_v)

    ## record the iteration
    v_rec[:, step] = VMag
    totalcost_rec.append(totalcost)
    demand_rec.append(demand)
print('Time for designed control is:',time.time()-t_start1)
# save the proposed controlled solution
file_name_control='loadshape/testing_data_driven_control_{}.mat'.format(args.trial_name)
scio.savemat(file_name_control,{'totalcost':totalcost_rec,
                                'v_rec':v_rec})


################################################################################################################
## Below is for the linear control method under time-varying setting
print('Calculating the linear control!!')
t_start2=time.time()
v,pd,qd=env.reset(0)
v_rec_linear=np.zeros((env.bus_num, num_steps), dtype=np.float32)
totalcost_rec_linear=[]
mu_upper_ = [0] * 36
mu_lower_ = [0] * 36
for step in range(num_steps):
    cost_linear, VMag,mu_upper_, mu_lower_ = env.primal_dual_real_time(mu_upper_, mu_lower_,step)
    totalcost_rec_linear.append(cost_linear)
    v_rec_linear[:,step]=VMag
print('Time for primal-dual control is:',time.time()-t_start2)
# save the primal-dual control
file_name_PD='loadshape/testing_primal_dual_control.mat'
# file_name_PD='loadshape/testing_no_control.mat'
scio.savemat(file_name_PD,{'totalcost':totalcost_rec_linear,
                           'v_rec':v_rec_linear})


#################################################################################################################
print('Solve the actual OPF solution!')
## Below is to solve the actual OPF solution
# if there is already solved solutions, just load it
file_name_test='loadshape/testing_OPF_solution_genco_1.mat'
if os.path.exists(file_name_test):
    OPFsolution=scio.loadmat(file_name_test)
    v_rec_actual=OPFsolution['v_rec'][0]
    totalcost_rec_actual=OPFsolution['totalcost'][0]
else:
    v_rec_actual = np.zeros((env.bus_num, num_steps), dtype=np.float32)
    t_start3 = time.time()
    totalcost_rec_actual = []
    for step in range(num_steps):
        cost_actual, VMag_actual = env.cal_OPF_solution(step)
        totalcost_rec_actual.append(cost_actual)
        v_rec_actual[:, step] = VMag_actual
        if step % 100 == 0:
            print('Solve OPF at: ', step)
    print('Time for Matpower is:', time.time() - t_start3)
    scio.savemat(file_name_test,{'totalcost':totalcost_rec_actual,
                                 'v_rec':v_rec_actual})


#################################################################################################################
#################################################################################################################
## Below is plot the figures
plt.figure(1,figsize=(18,6))
timeLine=list(range(v_rec.shape[1]))
injection_bus=env.injection_bus
for i in range(v_rec.shape[0]):
    if i in injection_bus:
        plt.plot(timeLine, v_rec[i, :], linestyle='--', label='injection_{}'.format(i + 1))
    else:
        plt.plot(timeLine, v_rec[i, :])
plt.xlabel('Time from 00:00 to 23:59')
plt.ylabel('Volatge')
plt.title('The Changes of Voltages in a Day')
plt.grid(True)
plt.savefig('figures/test_figures/{}_training_voltage.png'.format(args.trial_name))
plt.show()

plt.figure(2,figsize=(18,6))
timeLine=list(range(v_rec_linear.shape[1]))
injection_bus=env.injection_bus
for i in range(v_rec.shape[0]):
    if i in injection_bus:
        plt.plot(timeLine, v_rec_linear[i, :], linestyle='--', label='injection_{}'.format(i + 1))
    else:
        plt.plot(timeLine, v_rec_linear[i, :])
plt.xlabel('Time from 00:00 to 23:59')
plt.ylabel('Volatge')
plt.title('The Changes of Voltages in a Day (Linear)')
plt.grid(True)
plt.savefig('figures/test_figures/{}_voltage_linear.png'.format(args.trial_name))
plt.show()

plt.figure(3,figsize=(18,6))
timeLine=list(range(len(totalcost_rec_linear)))
plt.plot(timeLine,totalcost_rec,c='red',label='Designed Local Dynamics')
plt.plot(timeLine,totalcost_rec_linear,label='Primal-Dual on Pg and Qg')
plt.plot(timeLine,totalcost_rec_actual,c='green',label='Actual_Cost')
plt.xlabel('Iteration')
plt.ylabel('Total Cost')
plt.title('Total Cost in Real-Time')
plt.grid(True)
plt.legend()
plt.savefig('figures/test_figures/{}_totalcost.png'.format(args.trial_name))
plt.show()

plt.figure(4)
plt.plot(timeLine,demand_rec,c='red',label='load shape')
plt.ylabel('the overall power demands')
plt.xlabel('load change from 00:00 to 23:59')
plt.title('Load Shape')
plt.grid(True)
plt.legend()
plt.savefig('figures/test_figures/{}_loadshape.png'.format(args.trial_name))
plt.show()

## calculate the tracking error
tracking_error_learn=np.abs(np.array(totalcost_rec)-np.array(totalcost_rec_actual))
tracking_error_primaldual=np.abs(np.array(totalcost_rec_linear)-np.array(totalcost_rec_actual))
timeLine=list(range(len(tracking_error_learn)))
plt.figure(5)
plt.plot(timeLine,tracking_error_learn,c='blue',label='Learned Dynamics')
plt.plot(timeLine,tracking_error_primaldual,c='green',linestyle='dotted',label='Primal-Dual')
plt.xlabel('Time from 00:00 to 23:59')
plt.ylabel('Errors')
plt.title('The Tracking Errors')
plt.grid(True)
plt.legend()
plt.savefig('figures/test_figures/tracking_errors.png'.format(args.trial_name))
plt.show()

## The actual voltage
plt.figure(6)
timeLine=list(range(v_rec_actual.shape[1]))
injection_bus=env.injection_bus
for i in range(v_rec_actual.shape[0]):
    if i in injection_bus:
        plt.plot(timeLine, v_rec_actual[i, :], linestyle='--', label='injection_{}'.format(i + 1))
    else:
        plt.plot(timeLine, v_rec_actual[i, :])
plt.xlabel('Time from 00:00 to 23:59')
plt.ylabel('The Actual Volatge')
plt.title('The Changes of Actual Voltages in a Day')
plt.grid(True)
plt.show()


print('The tracking error by learning dynamics is:',np.mean(tracking_error_learn))
print('The tracking error by primal-dual dynamics is:',np.mean(tracking_error_primaldual))
import time

import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from pypower.loadcase import loadcase
from pypower.ppoption import ppoption

from env_37bus import IEEE37bus
from safenet import PrimalDualTrain,SafeNet,AuxiliaryT,PrimalTrainT

use_cuda=torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

parser=argparse.ArgumentParser(description='Primal-dual learning')
parser.add_argument('--env_name',default='case37',help='Please choose (case37)')
parser.add_argument('--train_flag',default=True,help='Training or not')
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
    agent=PrimalDualTrain(safenet=net)
    agent_list.append(agent)

## load the pre_train model if not training
if args.train_flag!=True:
    for i in range(num_agent):
        net_dict=torch.load('model_pth/{}_safenet_{}.pth'.format(args.trial_name,i))
        agent_list[i].safenet.load_state_dict(net_dict)


## Set the training parameter
num_epochs=50
num_episodes=5760
num_steps=50
batch_size=32

## The dual variables
mu_lower=np.zeros((36,),dtype=np.float32)
mu_upper=np.zeros((36,),dtype=np.float32)
mu_lower_batch=np.tile(mu_lower,(batch_size,1))
mu_upper_batch=np.tile(mu_upper,(batch_size,1))
mu_lower_rec=[]
lr_mu=1e2


## The auxiliary variable for chance constraint
alpha=0.1
auxiliary_t=AuxiliaryT(bus_num=env.bus_num,alpha=alpha)
agent_t=PrimalTrainT(auxiliary_t=auxiliary_t)
t_lower,t_upper=agent_t.auxiliary_t.get_instance_t()

VUpper=[1.05]*36
VLower=[0.95]*36

# construct for batch training
batch_vgen=np.zeros((batch_size,env.gen_num),dtype=np.float32)
batch_VMag=np.zeros((batch_size,env.bus_num),dtype=np.float32)
batch_pg=np.zeros((batch_size,env.gen_num),dtype=np.float32)
batch_qg=np.zeros((batch_size,env.gen_num),dtype=np.float32)
batch_pd=np.zeros((batch_size,env.gen_num),dtype=np.float32)
batch_qd=np.zeros((batch_size,env.gen_num),dtype=np.float32)
ZEROS=[0]*batch_size
for epoch in range(num_epochs):
    v_rec_episodes=np.zeros((env.bus_num,num_episodes),dtype=np.float32)
    # reset demanded loads before each epoch
    env.reset_demand()
    t_start=time.time()
    for episode in range(num_episodes):
        v,load_p,load_q = env.reset(episode)

        v_rec = np.zeros((env.bus_num, num_steps), dtype=np.float32)
        totalcost_rec = []
        for step in range(num_steps):
            action = []
            # if step==100:
            #     print(1)
            for i in range(num_agent):
                action_agent = agent_list[i].safenet.get_action(np.asarray([v[i]]), i, mu_upper_batch, mu_lower_batch,t_upper,t_lower,load_p[i],load_q[i])
                action.append(action_agent)

            action = np.asarray(action)
            next_v, VMag, totalcost,pg,qg = env.step_Reward(action)

            v = np.copy(next_v)

            ## record the iteration
            v_rec[:, step] = VMag
            totalcost_rec.append(totalcost)

        ## record the episode
        v_rec_episodes[:, episode] = VMag

        # construct the minibatch data for training
        batch_vgen[episode%batch_size,:]=next_v
        batch_VMag[episode%batch_size,:]=VMag
        batch_pg[episode%batch_size,:]=pg
        batch_qg[episode%batch_size,:]=qg
        batch_pd[episode%batch_size,:]=load_p
        batch_qd[episode%batch_size,:]=load_q

        if (episode+1)%batch_size==0 and args.train_flag == True:
            t_lower = np.ones((36,), dtype=np.float32) * 0.0005
            t_upper = np.ones((36,), dtype=np.float32) * 0.0005

            ## The dual update
            # The average formulation for dual update
            lr_mu_ = lr_mu #/ np.sqrt(epoch*num_epochs+episode + 1)
            for j in range(VMag.shape[0]):
                grad_mu_lower = np.mean(np.max((t_lower[j] + VLower[j] ** 2 - batch_VMag[:, j] ** 2, ZEROS), axis=0) - alpha * t_lower[j])
                grad_mu_upper = np.mean(np.max((t_upper[j] + batch_VMag[:, j] ** 2 - VUpper[j] ** 2, ZEROS), axis=0) - alpha * t_upper[j])

                mu_upper[j] = max(mu_upper[j] + lr_mu_ * (grad_mu_upper - 0.0 * mu_upper[j]), 0)
                mu_lower[j] = max(mu_lower[j] + lr_mu_ * (grad_mu_lower - 0.0 * mu_lower[j]), 0)

            mu_lower_batch = np.tile(mu_lower, (batch_size, 1))
            mu_upper_batch = np.tile(mu_upper, (batch_size, 1))

            # record the dual variables
            mu_lower_rec.append(mu_lower.copy())

            # ## The t_lower and t_upper update
            # agent_t.primaltrain_t_step(batch_VMag,mu_upper_batch,mu_lower_batch)
            #
            # t_lower, t_upper = agent_t.auxiliary_t.get_instance_t()
            ## The primal update
            for i in range(num_agent):
                ## The primal train step
                agent_list[i].primaltrain_step(batch_vgen[:,i], batch_VMag,i, mu_upper_batch, mu_lower_batch,t_upper,t_lower,batch_pg[:,i],batch_qg[:,i],batch_pd[:,i],batch_qd[:,i])

        if episode%batch_size==0:
            print('The training epoch: {}, episode: {}'.format(epoch,episode))
    print('Episode time:{} at epoch {}'.format(time.time()-t_start,epoch))

    if epoch%10==0 and epoch!=0:
        for i in range(num_agent):
            torch.save(agent_list[i].safenet.state_dict(),'model_pth/{}_safenet_{}_epoch_{}.pth'.format(args.trial_name,i,epoch),_use_new_zipfile_serialization=False)

if args.train_flag==True:
    ## save the model
    for i in range(num_agent):
        torch.save(agent_list[i].safenet.state_dict(),'model_pth/{}_safenet_{}.pth'.format(args.trial_name,i),_use_new_zipfile_serialization=False)

if args.lineargrad:
    _ = env.reset(episode)
    totalcost_linear=[]
    mu_upper_ = [0] * 36
    mu_lower_ = [0] * 36
    for i in range(num_steps):
        cost_linear,mu_upper_,mu_lower_=env.primal_dual_linear(mu_upper_,mu_lower_)
        totalcost_linear.append(cost_linear)

plt.figure(1)
timeLine=list(range(v_rec.shape[1]))
injection_bus=env.injection_bus
for i in range(v_rec.shape[0]):
    if i in injection_bus:
        plt.plot(timeLine, v_rec[i, :], linestyle='--', label='injection_{}'.format(i + 1))
    else:
        plt.plot(timeLine, v_rec[i, :])
plt.xlabel('Iteration')
plt.ylabel('Volatge')
plt.title('Voltage Convergence')
plt.grid(True)
plt.savefig('figures/{}_voltage_convergence.png'.format(args.trial_name))
plt.show()

plt.figure(2)
timeLine=list(range(len(totalcost_rec)))
plt.plot(timeLine,totalcost_rec,c='red',label='primal on DNN')
if args.lineargrad:
    plt.plot(timeLine,totalcost_linear,label='primal on pg and qg')
plt.xlabel('Iteration')
plt.ylabel('Total Cost')
plt.title('Total Cost Convergence')
plt.grid(True)
plt.legend()
plt.savefig('figures/{}_totalcost.png'.format(args.trial_name))
plt.show()

if args.train_flag==True:
    plt.figure(3)
    timeLine2 = list(range(v_rec_episodes.shape[1]))
    for i in range(v_rec_episodes.shape[0]):
        if i in injection_bus:
            plt.plot(timeLine2, v_rec_episodes[i, :], linestyle='--')
        else:
            plt.plot(timeLine2, v_rec_episodes[i, :])
    plt.xlabel('Training Episode')
    plt.ylabel('Voltage')
    plt.title('Training Result')
    plt.grid(True)
    plt.savefig('figures/{}_training_voltage.png'.format(args.trial_name))
    plt.show()

    # plot the dual variable
    plt.figure(4)
    mu_lower_rec=np.asarray(mu_lower_rec)
    timeLine=list(range(mu_lower_rec.shape[0]))
    for i in range(mu_lower_rec.shape[1]):
        plt.plot(timeLine,mu_lower_rec[:,i])
    plt.xlabel('training episode')
    plt.ylabel('dual variables')
    plt.grid(True)
    plt.savefig('figures/{}_dual_variables.png'.format(args.trial_name))
    plt.show()
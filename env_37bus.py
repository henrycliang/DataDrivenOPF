import gym
import numpy as np
from pypower.api import runopf,runpf
# from pypower.runpf import runpf
import pandas as pd
import scipy.io as scio
import os

from utils import LinearGradient,interpolinate

class IEEE37bus(gym.Env):
    def __init__(self,mpc,ppopt):
        mpc['bus'][:,11]=1.2
        mpc['bus'][:,12]=0.8
        # mpc['bus'][:,2]=mpc['bus'][:,2]*1.1
        # mpc['bus'][:, 3] = mpc['bus'][:, 3] * 1.1
        self.injection_bus = (mpc['gen'][1:, 0] - 1).astype(int)
        ## reset the load demand
        mpc['bus'][self.injection_bus,2]=mpc['bus'][self.injection_bus,2]*2.0
        mpc['bus'][self.injection_bus,3]=mpc['bus'][self.injection_bus,3]*2.0
        self.ppopt=ppopt
        self.mpc=mpc
        self.gen=mpc['gen']
        self.bus=mpc['bus']
        self.gen_num=self.gen.shape[0]-1
        self.bus_num=self.bus.shape[0]

        ## set the basis for the power model
        self.Vbase = 4.8  # in kV, if you want to change it to V, you need to multiply with 10e3
        self.Sbase = ((self.Vbase * 1000) ** 2) / 1e3  ## in kVA, if you want to change to W, you need to multiply 10e3

        ## get the pg and qg bounds
        self.Pgbound_upper = np.array([mpc['gen'][1:, 8]]) * 1e6 / self.Sbase
        self.Pgbound_lower = np.array([mpc['gen'][1:, 9]]) * 1e6 / self.Sbase
        self.Qgbound_upper = np.array([mpc['gen'][1:, 3]]) * 1e6 / self.Sbase
        self.Qgbound_lower = np.array([mpc['gen'][1:, 4]]) * 1e6 / self.Sbase

        ## set the initial load
        self.load_p = self.mpc['bus'][self.injection_bus, 2].copy()
        self.load_q = self.mpc['bus'][self.injection_bus, 3].copy()

        ## set the initial point
        self.pg = (self.gen[1:, 8] + self.gen[1:, 9]) * 1e6 / self.Sbase
        self.qg = (self.gen[1:, 3] + self.gen[1:, 4]) * 1e6 / self.Sbase

        self.pg_ref = (self.gen[1:, 8] + self.gen[1:, 9]) * 1e6 / self.Sbase * 0
        self.qg_ref = (self.gen[1:, 3] + self.gen[1:, 4]) * 1e6 / self.Sbase * 0

        ## the step size of primal and dual variables
        self.step_pq = 0.9*6*8e-2

        ## the coefficients of OPF objective function
        self.gencost_a = mpc['gencost'][1:self.gen_num+1, 4].copy()*10/3
        self.gencost_b = mpc['gencost'][1:self.gen_num+1, 5]
        self.gencost_c = mpc['gencost'][1:self.gen_num+1, 6]

        ## calculate the gradient mat
        LinearMpc=LinearGradient(mpc)
        self.gradientMat_p=LinearMpc.gradientMat_p
        self.gradientMat_q=LinearMpc.gradientMat_q

        self.plot_totalcost=True
        self.is_train=True
        # print(1)

    def project(self,pg,qg,pgbound_upper,pgbound_lower,qgbound_upper,qgbound_lower):
        pg_new=np.minimum(np.maximum(pg,pgbound_lower),pgbound_upper)
        qg_new=np.minimum(np.maximum(qg,qgbound_lower),qgbound_upper)
        return pg_new,qg_new

    def step_Reward(self,action):
        action_pg=action[:,0]
        action_qg=action[:,1]

        ## update pg and qg
        for i in range(self.gen_num):
            self.pg[i] = self.pg[i] - self.step_pq * (
                        2 * self.gencost_a[i] * (self.pg[i] - self.pg_ref[i]) - action_pg[i])
            self.qg[i] = self.qg[i] - self.step_pq * (
                        2 * self.gencost_a[i] * (self.qg[i] - self.qg_ref[i]) - action_qg[i])
            self.pg[i], self.qg[i] = self.project(self.pg[i], self.qg[i], self.Pgbound_upper[0, i],
                                                  self.Pgbound_lower[0, i], self.Qgbound_upper[0, i],
                                                  self.Qgbound_lower[0, i])

        ### Solve the power flow using the new power injection
        self.mpc['gen'][1:, 1] = self.pg * self.Sbase / 1e6
        self.mpc['gen'][1:, 2] = self.qg * self.Sbase / 1e6
        # solve the power flow
        result = runpf(self.mpc, self.ppopt)
        if result[0]['success']:
            VMag = result[0]['bus'][:, 7]
        else:
            print('Warning! The power flow fails to converge!!')

        VGen=VMag[self.injection_bus]

        # return the totalcost
        total_cost=0
        if self.plot_totalcost:
            for i in range(self.gen_num):
                total_cost+=self.gencost_a[i]*(self.pg[i]-self.pg_ref[i])**2+self.gencost_a[i]*(self.qg[i]-self.qg_ref[i])**2
        return VGen,VMag,total_cost,self.pg.copy(),self.qg.copy()


    def step_realtime(self,action,step):
        action_pg = action[:, 0]
        action_qg = action[:, 1]

        ###########################################################################
        # The below is for real-time loads
        self.mpc['bus'][self.injection_bus, 2] = self.new_load_p[step]
        self.mpc['bus'][self.injection_bus, 3] = self.new_load_q[step]

        ## return the t+1 time step loads for computing of action u(t+1)
        load_p=self.new_load_p[step+1]
        load_q=self.new_load_q[step+1]

        # Use for load shape calculation
        demand=np.sum(load_p)


        # ## for step-wise load shape
        # if step <= 200:
        #     self.mpc['bus'][self.injection_bus, 2] = self.load_p * 0.9
        #     self.mpc['bus'][self.injection_bus, 3] = self.load_q * 0.9
        # elif 200 < step <= 400:
        #     self.mpc['bus'][self.injection_bus, 2] = self.load_p * 1.1
        #     self.mpc['bus'][self.injection_bus, 3] = self.load_q * 1.1
        # else:
        #     self.mpc['bus'][self.injection_bus, 2] = self.load_p * 1.0
        #     self.mpc['bus'][self.injection_bus, 3] = self.load_q * 1.0
        ###########################################################################
        # print(self.pg, self.qg)
        ## update pg and qg
        for i in range(self.gen_num):
            self.pg[i] = self.pg[i] - self.step_pq * (
                    2 * self.gencost_a[i] * (self.pg[i] - self.pg_ref[i]) - action_pg[i])
            self.qg[i] = self.qg[i] - self.step_pq * (
                    2 * self.gencost_a[i] * (self.qg[i] - self.qg_ref[i]) - action_qg[i])
            self.pg[i], self.qg[i] = self.project(self.pg[i], self.qg[i], self.Pgbound_upper[0, i],
                                                  self.Pgbound_lower[0, i], self.Qgbound_upper[0, i],
                                                  self.Qgbound_lower[0, i])

        ### Solve the power flow using the new power injection
        self.mpc['gen'][1:, 1] = self.pg * self.Sbase / 1e6
        self.mpc['gen'][1:, 2] = self.qg * self.Sbase / 1e6

        # solve the power flow
        result = runpf(self.mpc, self.ppopt)
        if result[0]['success']:
            VMag = result[0]['bus'][:, 7]
        else:
            print('Warning! The power flow fails to converge!!')

        VGen = VMag[self.injection_bus]

        # return the totalcost
        total_cost = 0
        if self.plot_totalcost:
            for i in range(self.gen_num):
                total_cost += self.gencost_a[i] * (self.pg[i] - self.pg_ref[i]) ** 2 + self.gencost_a[i] * (
                            self.qg[i] - self.qg_ref[i]) ** 2
        return VGen, VMag, total_cost, demand, 50 * load_p, 50 * load_q  # self.demand[step].copy()


    def reset(self,i):
        ## reset the initial point
        # sampling_value = np.random.uniform(low=0.8, high=1.0, size=(2 * 5,))
        self.pg = (self.gen[1:, 8] + self.gen[1:, 9]) * 1e6 / self.Sbase *0#* sampling_value[:5]
        self.qg = (self.gen[1:, 3] + self.gen[1:, 4]) * 1e6 / self.Sbase *0#* sampling_value[5:]

        ## reset the state
        ### Solve the power flow using the reset power injection
        self.mpc['gen'][1:, 1] = self.pg * self.Sbase / 1e6
        self.mpc['gen'][1:, 2] = self.qg * self.Sbase / 1e6

        ###########################################################################
        # The below is for real-time loads

        self.mpc['bus'][self.injection_bus,2]=self.new_load_p[i]
        self.mpc['bus'][self.injection_bus,3]=self.new_load_q[i]
        load_p = self.mpc['bus'][self.injection_bus, 2].copy()
        load_q = self.mpc['bus'][self.injection_bus, 3].copy()
        ###########################################################################

        # solve the power flow
        result = runpf(self.mpc, self.ppopt)
        if result[0]['success']:
            VMag = result[0]['bus'][:, 7]
        else:
            print('Warning! The power flow fails to converge!!')

        VGen=VMag[self.injection_bus]
        return VGen,50*load_p,50*load_q

    def reset_demand(self):
        ## The below is for real-time loads
        loadshape = pd.read_csv('loadshape/CAISO-netdemand-20240909.csv')
        demand = loadshape.iloc[4]
        demand = demand[193:]
        demand = np.asarray(demand)-24000
        demand = interpolinate(demand)
        if self.is_train==True:
            # the load profile on day 1
            loadshape1=pd.read_csv('loadshape/CAISO-netdemand-20240908.csv')
            demand1=loadshape1.iloc[4]
            demand1=demand1[193:]
            demand1=np.asarray(demand1)-22000
            demand1=interpolinate(demand1,num=20)
            max_value1=np.max(demand1)
            scale_demand1=demand1/max_value1

            # the load profile on day 2
            loadshape2=pd.read_csv('loadshape/CAISO-netdemand-20240905.csv')
            demand2=loadshape2.iloc[3]
            demand2=demand2[193:]
            demand2=np.asarray(demand2)-24000
            demand2=interpolinate(demand2,num=20)
            max_value2=np.max(demand2)
            scale_demand2=demand2/max_value2

            # the load profile on day 3
            loadshape3=pd.read_csv('loadshape/CAISO-netdemand-20240903.csv')
            demand3=loadshape3.iloc[4]
            demand3=demand3[193:]
            demand3=np.asarray(demand3)-20000
            demand3=interpolinate(demand3,num=20)
            max_value3=np.max(demand3)
            scale_demand3=demand3/max_value3

            ## concatenate all the training data
            scale_demand_train=np.hstack((scale_demand1,scale_demand2,scale_demand3))

            np.random.shuffle(scale_demand_train)

            # scale each bus's load
            load_p = self.load_p
            load_q = self.load_q
            scale_p = 0.08
            scale_q = 0.08
            delta_p_allbus = 1 / np.sqrt(load_p) * scale_p
            delta_q_allbus = 1 / np.sqrt(load_q) * scale_q

            new_load_p = np.zeros((len(scale_demand_train), len(delta_p_allbus)), dtype=np.float32)
            new_load_q = np.zeros((len(scale_demand_train), len(delta_q_allbus)), dtype=np.float32)
            for i in range(len(scale_demand_train)):
                new_load_p[i, :] = load_p * (scale_demand_train[i] + delta_p_allbus * np.random.normal(1, 0.05, size=(self.gen_num,)))
                new_load_q[i, :] = load_q * (scale_demand_train[i] + delta_q_allbus * np.random.normal(1, 0.05, size=(self.gen_num,)))

            ## save the load profile, or load the existing profile
            file_name = 'loadshape/training_load_profiles.mat'
            if os.path.exists(file_name):
                load_profiles=scio.loadmat(file_name)
                shuffle_index=np.arange(len(load_profiles['load_p']))
                np.random.shuffle(shuffle_index) # for shuffle the dataset
                self.new_load_p=load_profiles['load_p'][shuffle_index]
                self.new_load_q=load_profiles['load_q'][shuffle_index]
            else:
                self.new_load_p = new_load_p
                self.new_load_q = new_load_q
                scio.savemat(file_name, {'load_p': new_load_p,
                                         'load_q': new_load_q})

        ## This else is for testing
        else:
            max_value=np.max(demand)
            scale_demand=demand/max_value

            load_p = self.load_p
            load_q = self.load_q

            scale_p = 0.08
            scale_q = 0.08
            delta_p_allbus = 1 / np.sqrt(load_p) * scale_p
            delta_q_allbus = 1 / np.sqrt(load_q) * scale_q

            new_load_p = np.zeros((len(scale_demand), len(delta_p_allbus)), dtype=np.float32)
            new_load_q = np.zeros((len(scale_demand), len(delta_q_allbus)), dtype=np.float32)
            for i in range(len(scale_demand)):
                if i==700:
                    scale_demand[i]=scale_demand[i]*1.35
                if i==900:
                    scale_demand[i]=scale_demand[i]*1.45
                if i==1000:
                    scale_demand[i]=scale_demand[i]*1.45
                if i==1200:
                    scale_demand[i]=scale_demand[i]*1.25
                if i==2200:
                    scale_demand[i]=scale_demand[i]*1.1
                new_load_p[i, :] = load_p * (scale_demand[i] + delta_p_allbus * np.random.normal(1, 0.05, size=(self.gen_num,)))
                new_load_q[i, :] = load_q * (scale_demand[i] + delta_q_allbus * np.random.normal(1, 0.05, size=(self.gen_num,)))

                # new_load_p[i, :] = load_p * (scale_demand[i] + delta_p_allbus )
                # new_load_q[i, :] = load_q * (scale_demand[i] + delta_q_allbus )
            # ##------------------------------------------------------------------------------
            # # only for figure plot
            # load_test_statistics_file='loadshape/testing_load_statistics.mat'
            # demand_all = loadshape.iloc[1]
            # demand_all = demand_all[193:]
            # demand_all = np.asarray(demand_all) - 24000
            # demand_all = interpolinate(demand_all)
            # scale_demand_all = demand_all / max_value+(np.mean(delta_p_allbus)+np.mean(delta_q_allbus))* np.random.normal(1, 0.05,size=(4800,))
            # scio.savemat(load_test_statistics_file,{'load_p':load_p,
            #                                         'load_q':load_q,
            #                                         'demand':scale_demand_all})
            # ##------------------------------------------------------------------------------
            ## save the load profile, or load the existing profile
            file_name_test='loadshape/testing_load_profiles.mat'
            if os.path.exists(file_name_test):
                load_profiles_test = scio.loadmat(file_name_test)
                self.new_load_p = load_profiles_test['load_p']
                self.new_load_q = load_profiles_test['load_q']
            else:
                self.new_load_p = new_load_p
                self.new_load_q = new_load_q
                scio.savemat(file_name_test, {'load_p': new_load_p,
                                              'load_q': new_load_q})


    def primal_dual_linear(self,mu_upper,mu_lower):
        step_pq=3e-2
        step_mu=0.2e4
        VUpper = [1.05] * 36
        VLower = [0.95] * 36
        ## The primal update
        for j in range(self.gen_num):
            action_pg=0
            action_qg=0
            for i in range(self.bus_num):
                action_pg += (mu_upper[i] - mu_lower[i]) * self.gradientMat_p[j, i].cpu().numpy()
                action_qg += (mu_upper[i] - mu_lower[i]) * self.gradientMat_q[j, i].cpu().numpy()
            self.pg[j]=self.pg[j]-step_pq*(2 * self.gencost_a[j] * (self.pg[j] - self.pg_ref[j])+action_pg)
            self.qg[j]=self.qg[j]-step_pq*(2 * self.gencost_a[j] * (self.qg[j] - self.qg_ref[j])+action_qg)
            self.pg[j], self.qg[j] = self.project(self.pg[j], self.qg[j], self.Pgbound_upper[0, j],
                                                  self.Pgbound_lower[0, j], self.Qgbound_upper[0, j],
                                                  self.Qgbound_lower[0, j])
        ### Solve the power flow using the new power injection
        self.mpc['gen'][1:, 1] = self.pg * self.Sbase / 1e6
        self.mpc['gen'][1:, 2] = self.qg * self.Sbase / 1e6
        # solve the power flow
        result = runpf(self.mpc, self.ppopt)
        if result[0]['success']:
            VMag = result[0]['bus'][:, 7]
        else:
            print('Warning! The power flow fails to converge!!')

        ## The dual update
        VMagSquare = np.square(VMag)
        for j in range(self.bus_num):
            mu_upper[j] = max(mu_upper[j] + step_mu * (VMagSquare[j] - VUpper[j] ** 2 - 0.0 * mu_upper[j]), 0)  # 0.000001
            mu_lower[j] = max(mu_lower[j] + step_mu * (VLower[j] ** 2 - VMagSquare[j] - 0.0 * mu_lower[j]), 0)

        # return the totalcost
        total_cost = 0
        if self.plot_totalcost:
            for i in range(self.gen_num):
                total_cost += self.gencost_a[i] * (self.pg[i] - self.pg_ref[i]) ** 2 + self.gencost_a[i] * (self.qg[i] - self.qg_ref[i]) ** 2

        return total_cost,mu_upper,mu_lower

    def primal_dual_real_time(self,mu_upper,mu_lower,step):
        step_pq=6.0e-2
        step_mu=0.2e4
        VUpper = [1.05] * 36
        VLower = [0.95] * 36

        ###########################################################################
        # The below is for real-time loads
        self.mpc['bus'][self.injection_bus, 2] = self.new_load_p[step]
        self.mpc['bus'][self.injection_bus, 3] = self.new_load_q[step]

        # ## for step-wise load shape
        # if step <= 200:
        #     self.mpc['bus'][self.injection_bus, 2] = self.load_p * 0.9
        #     self.mpc['bus'][self.injection_bus, 3] = self.load_q * 0.9
        # elif 200 < step <= 400:
        #     self.mpc['bus'][self.injection_bus, 2] = self.load_p * 1.1
        #     self.mpc['bus'][self.injection_bus, 3] = self.load_q * 1.1
        # else:
        #     self.mpc['bus'][self.injection_bus, 2] = self.load_p * 1.0
        #     self.mpc['bus'][self.injection_bus, 3] = self.load_q * 1.0

        ## The primal update
        for j in range(self.gen_num):
            action_pg=0
            action_qg=0
            for i in range(self.bus_num):
                action_pg += (mu_upper[i] - mu_lower[i]) * self.gradientMat_p[j, i].cpu().numpy()
                action_qg += (mu_upper[i] - mu_lower[i]) * self.gradientMat_q[j, i].cpu().numpy()
            self.pg[j]=self.pg[j]-step_pq*(2 * self.gencost_a[j] * (self.pg[j] - self.pg_ref[j])+action_pg)#*0
            self.qg[j]=self.qg[j]-step_pq*(2 * self.gencost_a[j] * (self.qg[j] - self.qg_ref[j])+action_qg)#*0
            self.pg[j], self.qg[j] = self.project(self.pg[j], self.qg[j], self.Pgbound_upper[0, j],
                                                  self.Pgbound_lower[0, j], self.Qgbound_upper[0, j],
                                                  self.Qgbound_lower[0, j])
        ### Solve the power flow using the new power injection
        self.mpc['gen'][1:, 1] = self.pg * self.Sbase / 1e6
        self.mpc['gen'][1:, 2] = self.qg * self.Sbase / 1e6
        # solve the power flow
        result = runpf(self.mpc, self.ppopt)
        if result[0]['success']:
            VMag = result[0]['bus'][:, 7]
        else:
            print('Warning! The power flow fails to converge!!')

        ## The dual update
        VMagSquare = np.square(VMag)
        for j in range(self.bus_num):
            mu_upper[j] = max(mu_upper[j] + step_mu * (VMagSquare[j] - VUpper[j] ** 2 - 0.0 * mu_upper[j]), 0)  # 0.000001
            mu_lower[j] = max(mu_lower[j] + step_mu * (VLower[j] ** 2 - VMagSquare[j] - 0.0 * mu_lower[j]), 0)

        # return the totalcost
        total_cost = 0
        if self.plot_totalcost:
            for i in range(self.gen_num):
                total_cost += self.gencost_a[i] * (self.pg[i] - self.pg_ref[i]) ** 2 + self.gencost_a[i] * (self.qg[i] - self.qg_ref[i]) ** 2

        return total_cost,VMag,mu_upper,mu_lower

    def cal_OPF_solution(self,step):
        ###########################################################################
        # set the voltage upper and lower bound to 1.05 and 0.95 p.u.
        self.mpc['bus'][0,11]=1.0
        self.mpc['bus'][0,12]=1.0
        self.mpc['bus'][1:,11]=1.05
        self.mpc['bus'][1:,12]=0.95
        # The below is for real-time loads
        self.mpc['bus'][self.injection_bus, 2] = self.new_load_p[step]
        self.mpc['bus'][self.injection_bus, 3] = self.new_load_q[step]

        # # set the generator
        # self.mpc['gen'][1:, 1] = self.pg * self.Sbase / 1e6*0
        # self.mpc['gen'][1:, 2] = self.qg * self.Sbase / 1e6*0

        ## set the gencost (coefficient)
        k=10./3*2
        self.mpc['gencost'][1:self.gen_num+1,4]=self.gencost_a*k#*10/3
        self.mpc['gencost'][self.gen_num+2:,4]=self.gencost_a*k#*10/3

        self.mpc['gencost'][1:self.gen_num+1,5]=-2*self.pg_ref*self.Sbase/1e6*self.gencost_a*k#*10/3
        self.mpc['gencost'][self.gen_num+2:,5]=-2*self.qg_ref*self.Sbase/1e6*self.gencost_a*k#*10/3

        self.mpc['gencost'][1:self.gen_num+1,6]=(self.pg_ref*self.Sbase/1e6)**2*self.gencost_a*k#*10/3
        self.mpc['gencost'][self.gen_num+2:,6]=(self.qg_ref*self.Sbase/1e6)**2*self.gencost_a*k#*10/3

        # solve the OPF
        result=runopf(self.mpc,self.ppopt)

        pg=result['gen'][1:,1]*1e6/self.Sbase
        qg=result['gen'][1:,2]*1e6/self.Sbase

        # calculate the totalcost
        totalcost=0
        for i in range(self.gen_num):
            totalcost+=self.gencost_a[i]*(pg[i]-self.pg_ref[i])**2+self.gencost_a[i]*(qg[i]-self.qg_ref[i])**2

        VMag=result['bus'][:,7]
        return totalcost,VMag

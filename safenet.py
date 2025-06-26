import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.autograd import Function

def sample_sphere(ndim=3):
    vec=np.random.randn(ndim,1) # create a random normal column vector
    vec/=np.linalg.norm(vec,axis=0)
    return vec

class PrimalDualTrain:
    def __init__(self,safenet,lr=1e-3):
        use_cuda=torch.cuda.is_available()
        self.device=torch.device("cuda" if use_cuda else "cpu")
        self.safenet=safenet

        self.optimizer=optim.Adam(safenet.parameters(),lr=lr)


    def primaltrain_step(self,VGen,VMag,i,mu_upper,mu_lower,t_upper,t_lower,pg,qg,pd,qd):
        VGen=torch.FloatTensor(VGen).unsqueeze(1).to(self.device)
        pg=torch.FloatTensor(pg).unsqueeze(1).to(self.device)
        qg=torch.FloatTensor(qg).unsqueeze(1).to(self.device)
        pd = torch.FloatTensor(np.asarray(pd)).unsqueeze(1).to(self.device)
        qd = torch.FloatTensor(np.asarray(qd)).unsqueeze(1).to(self.device)
        VMag=torch.FloatTensor(VMag).to(self.device)
        loss=self.safenet(VGen,VMag,i,mu_upper,mu_lower,t_upper,t_lower,pg,qg,pd,qd).mean()

        ## use optimizer to backward gradient
        self.optimizer.zero_grad()
        loss.backward()

        ## The primal update, we use the optimizer.step() as the primal update
        self.optimizer.step()


class PrimalTrainT:
    def __init__(self,auxiliary_t,lr=5e-5):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.auxiliary_t=auxiliary_t

        self.optimizer=optim.Adam(auxiliary_t.parameters(),lr=lr)

    def primaltrain_t_step(self,VMag,mu_upper,mu_lower):
        VMag=torch.FloatTensor(VMag).to(self.device)
        mu_upper=torch.FloatTensor(mu_upper).to(self.device)
        mu_lower=torch.FloatTensor(mu_lower).to(self.device)
        loss=self.auxiliary_t(VMag,mu_lower,mu_upper).sum(axis=1).mean()

        self.optimizer.zero_grad()
        loss.backward()

        ## The t_lower and t_upper update, we use the optimizer.step() as the update
        self.optimizer.step()

        ## restrict t_upper and t_lower into positive numbers
        self.auxiliary_t.clip_t()




class PrimalGradientT(Function):
    @staticmethod
    def forward(ctx, NN_input,VMag,mu_lower,mu_upper,t_lower,t_upper,alpha):
        ctx.save_for_backward(NN_input,VMag,mu_lower,mu_upper,t_lower,t_upper,alpha)
        return NN_input

    @staticmethod
    def backward(ctx,grad_output):
        NN_input,VMag,mu_lower,mu_upper,t_lower,t_upper,alpha=ctx.saved_tensors
        mask_lower=torch.zeros_like(mu_lower)
        mask_lower[t_lower+0.95**2-VMag**2>=0]=1
        grad_t_lower=(mask_lower-alpha)*mu_lower

        # below is for upper bound
        mask_upper=torch.zeros_like(mu_upper)
        mask_upper[t_upper+VMag**2-1.05**2>=0]=1
        grad_t_upper=(mask_upper-alpha)*mu_upper
        grad_t_lower_upper=torch.cat((grad_t_lower,grad_t_upper),dim=1)
        grad_output=grad_output*grad_t_lower_upper
        return grad_output,None,None,None,None,None,None

class AuxiliaryT(nn.Module):
    def __init__(self,bus_num,alpha):
        super(AuxiliaryT,self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.alpha=torch.FloatTensor([alpha])

        self.t_lower=torch.nn.Parameter(torch.zeros(bus_num,1),requires_grad=True)
        self.t_upper=torch.nn.Parameter(torch.zeros(bus_num,1),requires_grad=True)

        self.primal_t_layer=PrimalGradientT.apply

    def forward(self,VMag,mu_lower,mu_upper):
        pseudo_input=torch.ones_like(VMag)
        t_lower_out=pseudo_input*torch.transpose(self.t_lower,0,1)
        t_upper_out=pseudo_input*torch.transpose(self.t_upper,0,1)
        t_out=torch.cat((t_lower_out,t_upper_out),dim=1)
        output=self.primal_t_layer(t_out,VMag,mu_lower,mu_upper,torch.squeeze(self.t_lower.detach()),torch.squeeze(self.t_upper.detach()), self.alpha)

        return output

    def get_instance_t(self):
        t_lower=self.t_lower.detach().cpu().numpy()
        t_upper=self.t_upper.detach().cpu().numpy()
        return np.squeeze(t_lower.T),np.squeeze(t_upper.T)

    def clip_t(self):
        self.t_lower.clamp(min=0)
        self.t_upper.clamp(min=0)




## This class is used to calculate the weights' gradient
class PrimalDualGradient(Function):
    @staticmethod
    def forward(ctx, NN_input,pg,qg,pg_ref,qg_ref,mu_upper,mu_lower,gradientMat_p,gradientMat_q,v,t_upper,t_lower):
        ctx.save_for_backward(NN_input,pg,qg,pg_ref,qg_ref,mu_upper,mu_lower,gradientMat_p,gradientMat_q,v,t_upper,t_lower)
        ## Need to define the forward
        return NN_input

    @staticmethod
    def backward(ctx,grad_output):
        NN_input,pg,qg,pg_ref,qg_ref,mu_upper,mu_lower,gradientMat_p,gradientMat_q,v,t_upper,t_lower=ctx.saved_tensors
        # below is for the lower bound
        mask_lower=torch.zeros_like(mu_lower)
        mask_lower[t_lower + 0.95**2 - v**2 >= 0] = 1
        indic_lower=mu_lower*mask_lower
        penalty_p_lower=torch.mm(-indic_lower,gradientMat_p)
        penalty_q_lower=torch.mm(-indic_lower,gradientMat_q)

        # below is for the upper bound
        mask_upper=torch.zeros_like(mu_upper)
        mask_upper[t_upper+v**2-1.05**2>=0]=1
        indic_upper=mu_upper*mask_upper
        penalty_p_upper=torch.mm(indic_upper,gradientMat_p)
        penalty_q_upper=torch.mm(indic_upper,gradientMat_q)

        grad_p=(2*(pg-pg_ref)+penalty_p_upper+penalty_p_lower)*0.5  ##0.5 is 1/2, 2 is the second-order derivative of the obj
        grad_q=(2*(qg-qg_ref)+penalty_q_upper+penalty_q_lower)*0.5
        pq=torch.cat((grad_p,grad_q),dim=1)
        grad_output=grad_output*pq
        return grad_output,None,None,None,None,None,None,None,None,None,None,None



class SafeNet(nn.Module):
    def __init__(self,env,obs_dim,action_dim,hidden_dim,scale=4,scale_pq=4,scale_bias=0.4):
        super(SafeNet,self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # the env
        self.env=env

        self.hidden_dim=hidden_dim

        ## define the primal-dual gradient layer
        self.primalduallayer=PrimalDualGradient.apply

        ## the parameter for pd
        self.linear1_pd = nn.Linear(obs_dim, hidden_dim)
        self.linear2_pd = nn.Linear(hidden_dim,hidden_dim)
        self.linear3_pd = nn.Linear(hidden_dim, action_dim)

        ## the parameter for qd
        self.linear1_qd = nn.Linear(obs_dim, hidden_dim)
        self.linear2_qd = nn.Linear(hidden_dim,hidden_dim)
        self.linear3_qd = nn.Linear(hidden_dim, action_dim)

        self.k_p=torch.nn.Parameter(torch.rand(action_dim,action_dim)*4,requires_grad=True)
        self.k_q=torch.nn.Parameter(torch.rand(action_dim,action_dim)*4,requires_grad=True)

    def forward(self,state,VMag,i,mu_upper,mu_lower,t_upper,t_lower,pg,qg,pd,qd):

        x_pd=F.relu(self.linear1_pd(pd))
        x_pd=F.relu(self.linear2_pd(x_pd))
        x_pd=self.linear3_pd(x_pd)

        # k_p = 4

        v_pg = -torch.square(self.k_p) * (state-1.9)
        # v_pg=-2*state
        action_pg = x_pd + v_pg


        ### here we seperate the weights for pg and qg
        x_qd=F.relu(self.linear1_qd(qd))
        x_qd=F.relu(self.linear2_qd(x_qd))
        x_qd=self.linear3_qd(x_qd)

        # k_q = 4
        v_qg = -torch.square(self.k_q) * (state-1.9)
        # v_qg=-2*state
        action_qg = x_qd + v_qg


        action = torch.cat((action_pg, action_qg), dim=1)

        ## Below is used for redefining backward
        pg_ref = torch.FloatTensor(np.asarray([self.env.pg_ref[i].copy()])).unsqueeze(0).to(self.device)
        pg_ref=pg_ref.repeat(state.shape[0],1)
        qg_ref = torch.FloatTensor(np.asarray([self.env.qg_ref[i].copy()])).unsqueeze(0).to(self.device)
        qg_ref=qg_ref.repeat(state.shape[0],1)

        mu_upper=torch.FloatTensor(mu_upper).to(self.device)
        mu_lower=torch.FloatTensor(mu_lower).to(self.device)
        t_upper=torch.FloatTensor(t_upper).unsqueeze(0).to(self.device)
        t_lower=torch.FloatTensor(t_lower).unsqueeze(0).to(self.device)
        gradientMat_p=torch.FloatTensor(self.env.gradientMat_p[i,:]).unsqueeze(1).to(self.device)
        gradientMat_q=torch.FloatTensor(self.env.gradientMat_q[i,:]).unsqueeze(1).to(self.device)
        action = self.primalduallayer(action, pg, qg, pg_ref, qg_ref,mu_upper,mu_lower,gradientMat_p,gradientMat_q,VMag,t_upper,t_lower)
        return action

    def get_action(self,v,i,mu_upper,mu_lower,t_upper,t_lower,pd,qd):
        v = torch.FloatTensor(v).unsqueeze(0).to(self.device)
        pd = torch.FloatTensor(np.asarray([pd])).unsqueeze(1).to(self.device)
        qd = torch.FloatTensor(np.asarray([qd])).unsqueeze(1).to(self.device)
        pg = torch.FloatTensor(np.asarray([self.env.pg[i]])).unsqueeze(0).to(self.device)
        qg = torch.FloatTensor(np.asarray([self.env.qg[i]])).unsqueeze(0).to(self.device)
        batch_VMag=np.ones_like(mu_upper)
        batch_VMag=torch.FloatTensor(batch_VMag).to(self.device)
        action=self.forward(v,batch_VMag,i,mu_upper,mu_lower,t_upper,t_lower,pg,qg,pd,qd)
        return action.detach().cpu().numpy()[0]

    def cal_loss(self,v,i,mu_upper,mu_lower):
        # v = torch.FloatTensor(v).unsqueeze(0).to(self.device)
        action = self.forward(v, i,mu_upper,mu_lower)
        return action
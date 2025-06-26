import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rc('font',family='Times New Roman')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.default'] = 'it'



from matplotlib import rcParams
config = {
    # "font.family":'serif',
    # "font.size": 14,
    "mathtext.fontset":'stix',
    # "font.serif": ['SimSun'],
}
rcParams.update(config)
# # 启用 LaTeX
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['STCAIYUN']


Solution_control=scio.loadmat('loadshape/testing_data_driven_control_case37Adam_alpha_01_genco_1.mat')
Solution_PD=scio.loadmat('loadshape/testing_primal_dual_control.mat')
Solution_OPF=scio.loadmat('loadshape/testing_OPF_solution_genco_1.mat')
Solution_NoCtrl=scio.loadmat('loadshape/testing_no_control.mat')

v_control=Solution_control['v_rec']
v_PD=Solution_PD['v_rec']
v_OPF=Solution_OPF['v_rec']
v_NoCtrl=Solution_NoCtrl['v_rec']

obj_control=Solution_control['totalcost'][0]
obj_PD=Solution_PD['totalcost'][0]
obj_OPF=Solution_OPF['totalcost'][0]

fig1,ax1=plt.subplots(1,1,figsize=(8,4))
timeLine=list(range(v_OPF.shape[1]))
observed_bus_index=[1,9,13,23,27]
# observed_bus_index=range(36)
ax1.set_xticks(ticks=[0,600,1200,1800,2400,3000,3600,4200,4800],labels=['16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00'],fontproperties='Times New Roman',fontsize=14)
ax1.set_xlim(0,4800)
ax1.set_ylim(0.93,1.0)
ax1.tick_params(axis='y',labelsize=14)
ax1.tick_params(axis='x',pad=5)
for i in observed_bus_index:
    # plt.plot(timeLine,v_OPF[i,:],linewidth=1.0)
    ax1.plot(timeLine, v_control[i, :], linewidth=1.0,label='Node {}'.format(i+1))
plt.grid(ls='--')
ax1.set_xlabel('Time',fontsize=14)
ax1.set_ylabel('Voltage magnitude (p.u.)',fontsize=14)
## insert the mini-box
axins1=inset_axes(ax1,width="40%",height="30%",bbox_to_anchor=(0.1,0.1,1,1),
                  loc='upper left',bbox_transform=ax1.transAxes)
axins1.plot(timeLine,v_control[23,:],linewidth=1.0,c='red')
axins1.set_position([0.6, 0.6, 0.2, 0.2])
xlim0=2190#timeLine[zone_left]-(timeLine[zone_right]-timeLine[zone_left])*x_ratio
xlim1=2250#timeLine[zone_right]+(timeLine[zone_right]-timeLine[zone_left])*x_ratio
ylim0=0.9475#np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1=0.9525#np.max(y)+(np.max(y)-np.min(y))*y_ratio
axins1.set_xlim(xlim0,xlim1)
axins1.set_ylim(ylim0,ylim1)
axins1.set_xticks(ticks=[2200,2220,2240],labels=['19:40','19:42','19:44'])
axins1.grid(ls='--')
mark_inset(ax1, axins1, loc1=3, loc2=4, fc="none", ec='k', lw=1)
ax1.legend(loc='lower right',fontsize=14)
plt.savefig('figures/test_figures/testing_voltage_control.png',bbox_inches='tight')
plt.show()

fig2,ax2=plt.subplots(1,1,figsize=(8,4))
timeLine=list(range(v_OPF.shape[1]))
ax2.set_xticks(ticks=[0,600,1200,1800,2400,3000,3600,4200,4800],labels=['16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00'],fontproperties='Times New Roman',fontsize=14)
ax2.set_xlim(0,4800)
ax2.set_ylim(0.93,1.0)
ax2.tick_params(axis='y',labelsize=14)
ax2.tick_params(axis='x',pad=5)
# observed_bus_index=range(36)
for i in observed_bus_index:
    # plt.plot(timeLine,v_OPF[i,:],linewidth=1.0)
    ax2.plot(timeLine, v_PD[i, :], linewidth=1.0,label='Node {}'.format(i+1))
plt.grid(ls='--')
ax2.set_xlabel('Time',fontsize=14)
ax2.set_ylabel('Voltage magnitude (p.u.)',fontsize=14)
## insert the mini-box
axins2=inset_axes(ax2,width="40%",height="30%",bbox_to_anchor=(0.1,0.1,1,1),
                  loc='upper left',bbox_transform=ax2.transAxes)
axins2.plot(timeLine,v_PD[23,:],linewidth=1.0,c='red')
axins2.set_position([0.6, 0.6, 0.2, 0.2])
xlim0=2190#timeLine[zone_left]-(timeLine[zone_right]-timeLine[zone_left])*x_ratio
xlim1=2250#timeLine[zone_right]+(timeLine[zone_right]-timeLine[zone_left])*x_ratio
ylim0=0.9475#np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1=0.9525#np.max(y)+(np.max(y)-np.min(y))*y_ratio
axins2.set_xlim(xlim0,xlim1)
axins2.set_ylim(ylim0,ylim1)
axins2.set_xticks(ticks=[2200,2220,2240],labels=['19:40','19:42','19:44'])
axins2.grid(ls='--')
mark_inset(ax2, axins2, loc1=3, loc2=4, fc="none", ec='k', lw=1)
ax2.legend(loc='lower right',fontsize=14)
plt.savefig('figures/test_figures/testing_voltage_PD.png',bbox_inches='tight')
plt.show()


fig3,ax3=plt.subplots(1,1,figsize=(8,4))
timeLine=list(range(v_OPF.shape[1]))
ax3.set_xticks(ticks=[0,600,1200,1800,2400,3000,3600,4200,4800],labels=['16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00'],fontproperties='Times New Roman',fontsize=14)
ax3.set_xlim(0,4800)
ax3.set_ylim(0.93,1.0)
ax3.tick_params(axis='y',labelsize=14)
ax3.tick_params(axis='x',pad=5)
# observed_bus_index=range(36)
for i in observed_bus_index:
    # plt.plot(timeLine,v_OPF[i,:],linewidth=1.0)
    ax3.plot(timeLine, v_NoCtrl[i, :], linewidth=1.0,label='Node {}'.format(i+1))
plt.grid(ls='--')
ax3.set_xlabel('Time',fontsize=14)
ax3.set_ylabel('Voltage magnitude (p.u.)',fontsize=14)
## insert the mini-box
axins3=inset_axes(ax3,width="60%",height="30%",bbox_to_anchor=(0.1,0.1,1,1),
                  loc='upper left',bbox_transform=ax3.transAxes)
axins3.plot(timeLine,v_NoCtrl[23,:],linewidth=1.0,c='red')
axins3.plot(timeLine,v_NoCtrl[13,:],linewidth=1.0,c='green')
axins3.set_position([0.6, 0.6, 0.2, 0.2])
xlim0=900#timeLine[zone_left]-(timeLine[zone_right]-timeLine[zone_left])*x_ratio
xlim1=3900#timeLine[zone_right]+(timeLine[zone_right]-timeLine[zone_left])*x_ratio
ylim0=0.88#np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1=0.935#np.max(y)+(np.max(y)-np.min(y))*y_ratio
axins3.set_xlim(xlim0,xlim1)
axins3.set_ylim(ylim0,ylim1)
axins3.set_xticks(ticks=[1200,1800,2400,3000,3600],labels=['18:00','19:00','20:00','21:00','22:00'])
axins3.grid(ls='--')
# mark_inset(ax2, axins2, loc1=3, loc2=4, fc="none", ec='k', lw=1)
ax3.legend(loc='lower right',fontsize=14)
plt.savefig('figures/test_figures/testing_voltage_NoControl.pdf',bbox_inches='tight')
plt.show()

# plt.figure(3,figsize=(8,4))
# timeLine=list(range(v_OPF.shape[1]))
# # observed_bus_index=[1,27,23,9,13]
# # observed_bus_index=range(36)
# plt.xticks(ticks=[0,600,1200,1800,2400,3000,3600,4200,4800],labels=['16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00'],fontproperties='Times New Roman',fontsize=10)
# plt.xlim(0,4800)
# plt.ylim(0.93,1.0)
# for i in observed_bus_index:
#     plt.plot(timeLine,v_NoCtrl[i,:],linewidth=1.0,label='Node {}'.format(i+1))
#     # plt.plot(timeLine, v_OPF[i, :], linewidth=1.0,linestyle='dotted')
# plt.grid(ls='--')
# plt.xlabel('Time',fontsize=10)
# plt.ylabel('Voltage magnitude (p.u.)',fontsize=10)
# plt.legend(loc='lower right')
# plt.savefig('figures/test_figures/testing_voltage_NoControl.pdf',bbox_inches='tight')
# plt.show()

tracking_error_control=np.abs(np.array(obj_control)-np.array(obj_OPF))
tracking_error_PD=np.abs(np.array(obj_PD)-np.array(obj_OPF))

# relative_track_control=tracking_error_control/obj_OPF
fig=plt.figure(4,figsize=(8,4))
ax=fig.add_subplot(111)
plt.xticks(ticks=[0,600,1200,1800,2400,3000,3600,4200,4800],labels=['16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00'],fontproperties='Times New Roman',fontsize=14)
ax.set_xlim(0,4800)
ax.set_ylim(-20,460)
ax.tick_params(axis='y',colors='blue',labelsize=14)
# ax.spines['bottom'].set_color('blue')
# plt.plot(timeLine,obj_OPF,linewidth=0.35,c='orange')
# plt.plot(timeLine,obj_OPF,linewidth=1.0)
ax.plot(timeLine,obj_OPF,linewidth=1.0,c='green',label='Optimal',linestyle='dotted')
ax.plot(timeLine,obj_PD,linewidth=0.8,c='blue',label='Primal-dual')
# plt.plot(timeLine,relative_track_control,linewidth=1.0,c='blue')
# plt.plot(timeLine,tracking_error_PD,linewidth=0.35,c='green')
ax.grid(ls='--')
ax2=ax.twinx()
ax2.plot(timeLine,tracking_error_PD,linewidth=0.5,c='brown',label='Abs-gap')
ax2.set_ylim(-10,230)
# ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))
ax.set_xlabel('Time',fontsize=14)
ax.set_ylabel("OPF objective",c='blue',fontsize=14)
ax2.set_ylabel(r'$\left|f(x^{t})-f(x^{*,t})\right|$',c='brown',fontsize=14)
ax2.tick_params(axis='y',colors='brown',labelsize=14)
ax2.spines['right'].set_color('brown')
ax2.spines['left'].set_color('blue')
fig.legend(loc='lower right',bbox_to_anchor=(0.999,0.3), bbox_transform=ax.transAxes,fontsize=15)
axins4=inset_axes(ax,width="40%",height="30%",bbox_to_anchor=(0.1,0.1,1,1),
                  loc='upper right',bbox_transform=ax.transAxes)
axins4.plot(timeLine,obj_PD,linewidth=1.5,c='blue')
axins4.plot(timeLine,obj_OPF,linewidth=1.5,c='green',linestyle='dotted')
axins4.set_position([0.6, 0.6, 0.2, 0.2])
xlim0=2190#timeLine[zone_left]-(timeLine[zone_right]-timeLine[zone_left])*x_ratio
xlim1=2250#timeLine[zone_right]+(timeLine[zone_right]-timeLine[zone_left])*x_ratio
ylim0=308#np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1=410#np.max(y)+(np.max(y)-np.min(y))*y_ratio
axins4.set_xlim(xlim0,xlim1)
axins4.set_ylim(ylim0,ylim1)
# axins4.tick_params(axis='y',colors='blue')
# axins4.spines['left'].set_color('blue')
axins4.set_xticks(ticks=[2200,2220,2240],labels=['19:40','19:42',''])
axins4.grid(ls='--')
mark_inset(ax, axins4, loc1=2, loc2=3, fc="none", ec='k', lw=1)
plt.savefig('figures/test_figures/testing_tracking_PD.png',bbox_inches='tight')
plt.show()

# relative_track_control=tracking_error_control/obj_OPF
fig=plt.figure(5,figsize=(8,4))
ax=fig.add_subplot(111)
plt.xticks(ticks=[0,600,1200,1800,2400,3000,3600,4200,4800],labels=['16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00'],fontproperties='Times New Roman',fontsize=14)
ax.set_xlim(0,4800)
ax.set_ylim(-20,460)
ax.tick_params(axis='y',colors='blue',labelsize=14)
# plt.plot(timeLine,obj_OPF,linewidth=0.35,c='orange')
ax.plot(timeLine,obj_OPF,linewidth=1.0,c='green',label='Optimal',linestyle='dotted')
ax.plot(timeLine,obj_control,linewidth=0.8,c='blue',label='Data-driven')
# plt.plot(timeLine,relative_track_control,linewidth=1.0,c='blue')
# plt.plot(timeLine,tracking_error_PD,linewidth=0.35,c='green')
ax.grid(ls='--')
ax2=ax.twinx()
ax2.plot(timeLine,tracking_error_control,linewidth=0.5,c='brown',label='Abs-gap')
ax2.set_ylim(-10,230)
# ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))
ax.set_xlabel('Time',fontsize=14)
ax.set_ylabel("OPF objective",c='blue',fontsize=14)
ax2.set_ylabel(r'$\left|f(x^t)-f(x^{*,t})\right|$',c='brown',fontsize=14)
ax2.tick_params(axis='y',colors='brown',labelsize=14)
ax2.spines['right'].set_color('brown')
ax2.spines['left'].set_color('blue')
fig.legend(loc='lower right',bbox_to_anchor=(0.999,0.3), bbox_transform=ax.transAxes,fontsize=15)
axins5=inset_axes(ax,width="40%",height="30%",bbox_to_anchor=(0.1,0.1,1,1),
                  loc='upper right',bbox_transform=ax.transAxes)
axins5.plot(timeLine,obj_control,linewidth=1.5,c='blue')
axins5.plot(timeLine,obj_OPF,linewidth=1.5,c='green',linestyle='dotted')
axins5.set_position([0.6, 0.6, 0.2, 0.2])
xlim0=2190#timeLine[zone_left]-(timeLine[zone_right]-timeLine[zone_left])*x_ratio
xlim1=2250#timeLine[zone_right]+(timeLine[zone_right]-timeLine[zone_left])*x_ratio
ylim0=308#np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1=410#np.max(y)+(np.max(y)-np.min(y))*y_ratio
axins5.set_xlim(xlim0,xlim1)
axins5.set_ylim(ylim0,ylim1)
axins5.set_xticks(ticks=[2200,2220,2240],labels=['19:40','19:42',''])
axins5.grid(ls='--')
mark_inset(ax, axins5, loc1=2, loc2=3, fc="none", ec='k', lw=1)
plt.savefig('figures/test_figures/testing_tracking_control.png',bbox_inches='tight')
plt.show()

peak_obj_OPF=np.max(obj_OPF)
count_ralative_index=np.where(obj_OPF>peak_obj_OPF*0.1)
relative_track_control=tracking_error_control/obj_OPF
relative_track_PD=tracking_error_PD/obj_OPF
print('The absolute tracking error by learning dynamics is: {}, and relative error is: {}'.format(np.mean(tracking_error_control),np.mean(relative_track_control[count_ralative_index])))
print('The absolute tracking error by primal-dual dynamics is: {}, and relative error is: {}'.format(np.mean(tracking_error_PD),np.mean(relative_track_PD[count_ralative_index])))

## calculate the voltage violation
# v_control_violate_index=v_control<0.95
# v_control_violate=v_control[v_control_violate_index]
# # v_control_violate[v_control_violate>0]=0
# # v_control_violate_mean=np.sum(np.sum(v_control_violate))/4799
#
#
# v_PD_violate_index=v_PD<0.95
# v_PD_violate=v_PD[v_PD_violate_index]
v_control_violate=v_control-0.95
v_control_violate[v_control_violate>0]=0
v_control_violate_mean=np.sum(np.sum(v_control_violate))/4799

v_PD_violate=v_PD-0.95
v_PD_violate[v_PD_violate>0]=0
v_PD_violate_mean=np.sum(np.sum(v_PD_violate))/4799

print('The violation voltage by proposed control is:',np.abs(v_control_violate_mean))
print('The violation voltage by primal-dual control is:',np.abs(v_PD_violate_mean))

# ## Plot the demand trend
# fig=plt.figure(6,figsize=(8,4))
# demand=scio.loadmat('loadshape/testing_load_profiles.mat')
# base_load=scio.loadmat('loadshape/testing_load_statistics.mat')
# scale_p=demand['load_p']/base_load['load_p']
# scale_q=demand['load_q']/base_load['load_q']
# scale_index=(scale_p+scale_q)/2
# scale_index=scale_index[:4799,:]
# demand_all=base_load['demand'][0,:4799]
#
# scale_index_mean=np.mean(scale_index,axis=1)
#
# # plt.plot(timeLine,demand['load_p'][:4799,:],color='lightblue')
# plt.plot(timeLine,scale_index,linewidth=0.5,color='lightblue')
# plt.plot(timeLine, scale_index_mean, color='blue')
# plt.plot(timeLine,demand_all,color='brown')
# plt.show()

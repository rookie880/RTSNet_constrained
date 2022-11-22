#%%
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from EKF_test import EKFTest
from Extended_RTS_Smoother_test import S_Test
from Extended_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T,wandb_switch

if wandb_switch:
    import wandb

from Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipeline_EKF import Pipeline_EKF
import matplotlib.pyplot as plt
# from PF_test import PFTest

from datetime import datetime

from KalmanNet_nn import KalmanNetNN
from RTSNet_nn import RTSNetNN

from Plot import Plot_extended as Plot

from filing_paths import path_model, path_session
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m, n, delta_t, m1x_0, m2x_0
from model import f, h, c, cf


if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")


print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

######################################
###  Compare EKF, RTS and RTSNet   ###
######################################
offset = 0
sequential_training = False
path_results = 'ERTSNet/'
PlotfileName = path_results + 'Car_' + strTime + '.png'
DatafolderName = 'Simulations/Car/data/T200' + '/'

R = torch.tensor([[0.3**2, 0.0],
                  [0.,     0.3**2]])
Q = torch.tensor([[0.0001**2, 0   ],
                  [0      , (2*delta_t)**2]])


# traj_resultName = ['traj_lor_KNetFull_rq1030_T2000_NT100.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
dataFileName = ['data_car_T200.pt']#,'data_lor_v20_r1e-2_T100.pt','data_lor_v20_r1e-3_T100.pt','data_lor_v20_r1e-4_T100.pt']
# KFRTSResultName = 'KFRTS_partialh_rq3050_T2000' 

#Generate and load data 
sys_model = SystemModel(f, Q, h, R, T, T_test, m, n, c=c, cf=cf)
sys_model.InitSequence(m1x_0, m2x_0)

#print("Start Data Gen")
#DataGen(sys_model, DatafolderName + dataFileName[0], T, T_test, randomInit=False)

print("Data Load")
print(dataFileName[0])
[train_input_long, train_target_long, cv_input, cv_target, test_input, test_target] =  torch.load(DatafolderName + dataFileName[0],map_location=dev)  

train_target = train_target_long[:,:,0:T]
train_input = train_input_long[:,:,0:T] 

print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())


xtemp = torch.linspace(train_target[0, 0, :].min(), train_target[0, 0, :].max(), 100)
fig, ax = plt.subplots(nrows=2)
ax[0].plot(train_target[0, 0, :], train_target[0, 1, :], label='training data', c='tab:blue')
ax[0].fill_between(xtemp, torch.cos(xtemp)-0.1, torch.cos(xtemp)+0.1, label='Constraints', alpha=0.2, color='tab:orange')
ax[0].legend(loc='upper right')
ax[0].grid()

ax[1].scatter(test_input[0, 0, :], test_input[0, 1, :], label='train_input data', color='tab:blue', s=5)
ax[1].fill_between(xtemp, torch.cos(xtemp)-0.1, torch.cos(xtemp)+0.1, label='Constraints', alpha=0.2, color='tab:orange')
ax[1].legend(loc='upper right')
ax[1].grid()
plt.show()


for rindex in range(1):
   # Model with full info
   sys_model = SystemModel(f, Q, h, R, T, T_test, m, n, c=c, cf=cf)
   sys_model.InitSequence(m1x_0, m2x_0)
   
   #Evaluate EKF true
   print("Evaluate EKF true")
   [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target)
   print("Evaluate RTS true")
   [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out] = S_Test(sys_model, test_input, test_target)

   xtemp = torch.linspace(train_target[0, 0, :].min(), train_target[0, 0, :].max(), 100)
   fig, ax = plt.subplots(nrows=2)
   ax[0].plot(EKF_out[0, 0, :], EKF_out[0, 1, :], label='EKF_out', c='tab:blue')
   ax[0].fill_between(xtemp, torch.cos(xtemp)-0.1, torch.cos(xtemp)+0.1, label='Constraints', alpha=0.2, color='tab:orange')
   ax[0].legend(loc='upper right')
   ax[0].grid()

   ax[1].plot(ERTS_out[0, 0, :], ERTS_out[0, 1, :], label='ERTS_out', c='tab:blue')
   ax[1].fill_between(xtemp, torch.cos(xtemp)-0.1, torch.cos(xtemp)+0.1, label='Constraints', alpha=0.2, color='tab:orange')
   ax[1].legend(loc='upper right')
   ax[1].grid()
   plt.show()

   #%%
   # RTSNet with full info
   ## Build Neural Network
   print("RTSNet with full model info")
   RTSNet_model = RTSNetNN()
   RTSNet_model.NNBuild(sys_model)
   
   ## Train Neural Network
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
   RTSNet_Pipeline.setssModel(sys_model)
   RTSNet_Pipeline.setModel(RTSNet_model)
   RTSNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=30, learningRate=1e-3, weightDecay=1e-6) 

   if wandb_switch:
    wandb.init(project="Constrained_RTSNet")
    wandb.log({
      "learning_rate": RTSNet_Pipeline.learningRate,
      "batch_size": RTSNet_Pipeline.N_B,
      "weight_decay": RTSNet_Pipeline.weightDecay})

   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
   
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime, x_out_test] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
   
# Close wandb run 
if wandb_switch:
   wandb.finish() 

# %%
legend = ["RTSNet", "Ground Truth", "MB RTS", "LB and UB"]
font_size = 14
plt.plot(x_out_test[0, :].detach().numpy(), x_out_test[1, :].detach().numpy(), label=legend[0])
plt.plot(test_target[-1, 0, :].detach().numpy(), test_target[-1, 1, :].detach().numpy(), label=legend[1])
plt.plot(ERTS_out[-1, 0, :], ERTS_out[-1, 1, :], label=legend[2])
plt.fill_between(test_target[-1, 0, :], torch.cos(test_target[-1, 0, :]) - 0.1, torch.cos(test_target[-1, 0, :]) + 0.1,
                 alpha=0.2, label=legend[3])

plt.legend(fontsize=font_size)
plt.xlabel('x', fontsize=font_size)
plt.ylabel('y', fontsize=font_size)
plt.grid()
plt.savefig(PlotfileName)
# %%

#%%
#path = r'C:\Users\aves\Documents\LiRA-aves\Map Localization\RTSNet_ICASSP22'
#sys.path.insert(0, path)
import math
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd
from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from parameters import m, n, delta_t, H_design

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")


def f(x):
    x_ = torch.zeros(x.shape)
    x_[0] = x[0] + 0.5*delta_t
    x_[1] = x[1] + 0.5*delta_t*torch.sin(-torch.sin(x[0])*torch.pi/4)
    return x_  

def h(x):
    return torch.matmul(H_design,x).to(cuda0)
    #return toSpherical(x)

def getJacobian(x, a):
    
    # if(x.size()[1] == 1):
    #     y = torch.reshape((x.T),[x.size()[0]])
    try:
        if(x.size()[1] == 1):
            y = torch.reshape((x.T),[x.size()[0]])
    except:
        y = torch.reshape((x.T),[x.size()[0]])
        
    if(a == 'ObsAcc'):
        g = h
    elif(a == 'ModAcc'):
        g = f
    #elif(a == 'ObsInacc'):
    #    g = hInacc
    #elif(a == 'ModInacc'):
    #    g = fInacc

    Jac = autograd.functional.jacobian(g, y)
    Jac = Jac.view(-1,m)
    return Jac


#x = torch.tensor([[1],[1]]).float() 
#H = getJacobian(x, 'ObsAcc')
#print(H)
#print(h(x))

#F = getJacobian(x, 'ModAcc')
#print(F)
#print(f(x))
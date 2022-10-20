import torch
import math

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

#########################
### Design Parameters ###
#########################
m = 2
n = 2

m1x_0 = torch.ones(m, 1) 
m1x_0 = torch.zeros(m)
m1x_0[0] = 0
m1x_0[1] = 1

m2x_0 = torch.tensor([[0.01**2, 0      ],
                      [0      , 0.01**2],
])

# Length of Time Series Sequence
# T = math.ceil(3000 / ratio)
# T_test = math.ceil(6e6 * ratio)
T = 200
delta_t = 1/20
T_test = 200

H_design = torch.eye(m)

# Noise Matrices
Q_non_diag = False
R_non_diag = False

R = torch.tensor([[0.3**2, 0.0],
                  [0.,     0.3**2]])
Q = torch.tensor([[0.0001**2, 0   ],
                  [0      , (2*delta_t)**2]])

#########################
### Model Parameters ####
#########################

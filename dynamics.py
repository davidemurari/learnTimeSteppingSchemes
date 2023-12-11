import torch
import numpy as np
import torch.nn as nn

def ode(t,y,system = "SIR"):
    
    if system=="Robert":
        a,b,c = 0.04,1e4,3e7
        return np.array([
            -a*y[0]+b*y[1]*y[2],
            a*y[0]-b*y[1]*y[2]-c*y[1]**2,
            c*y[1]**2
            ])
    elif system=="SIR":
        gamma,beta,N = 0.1,0.1,1.
        return np.array([
            -beta*y[1]*y[0]/N,
            beta*y[0]*y[1]/N - gamma*y[1],
            gamma*y[1]
        ])

class ode_torch(nn.Module):
    def __init__(self,system):
        super().__init__()
        self.system = system
    def forward(self,t,y):
        if self.system=="Robert":
            a,b,c = 0.04,1e4,3e7
            res = torch.cat((
                -a*y[:,0:1]+b*y[:,1:2]*y[:,2:3],
                a*y[:,0:1]-b*y[:,1:2]*y[:,2:3]-c*y[:,1:2]**2,
                c*y[:,1:2]**2
            ),dim=1)
            return res
        elif self.system=="SIR":
            gamma,beta,N = 0.1,0.1,1.
            res = torch.cat((
                -beta*y[:,1:2]*y[:,0:1]/N,
                beta*y[:,0:1]*y[:,1:2]/N - gamma*y[:,1:2],
                gamma*y[:,1:2]
            ),dim=1)
            return res


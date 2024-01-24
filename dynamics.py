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
        
        if self.system=="Robert":
            self.k1 = 0.04
            self.k2 = 3e7
            self.k3 = 1e4
        elif self.system=="SIR":
            self.beta = 0.1
            self.gamma = 0.1
            self.N = 1.
        
    def forward(self,t,y):
        if self.system=="Robert":
            res = torch.cat((
                -self.k1*y[:,0:1]+self.k3*y[:,1:2]*y[:,2:3],
                self.k1*y[:,0:1]-self.k3*y[:,1:2]*y[:,2:3]-self.k2*y[:,1:2]**2,
                self.k2*y[:,1:2]**2
            ),dim=1)
            return res
        elif self.system=="SIR":
            res = torch.cat((
                -self.beta*y[:,1:2]*y[:,0:1]/self.N,
                self.beta*y[:,0:1]*y[:,1:2]/self.N - self.gamma*y[:,1:2],
                self.gamma*y[:,1:2]
            ),dim=1)
            return res


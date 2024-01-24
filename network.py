import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamics import ode_torch


class activation(nn.Module):
        def __init__(self,):
            super().__init__()
        def forward(self,x):
            return torch.tanh(x)

def mat_batch_mats(M,mats):
    return torch.einsum('ij,bjk->bik',M,mats)

class network(nn.Module):
        def __init__(self,d=3,nlayers=3,system="Robert",num_hidden=20,dtype=torch.float32):
            super().__init__()
    
            self.d = d
            self.output_dim = 3
            
            self.nlayers = nlayers
            self.system = system
            
            self.act = lambda x : torch.tanh(x)
            
            self.H = num_hidden

            self.lift = nn.Linear(self.d+1,self.H,dtype=dtype)
            
            self.linears = nn.ModuleList([nn.Linear(self.H,self.H,dtype=dtype) for _ in range(self.nlayers)])
            self.proj = nn.Linear(self.H,self.output_dim,bias=False,dtype=dtype)
                        
            self.f = lambda t : 1-torch.exp(-t)

        def parametric(self,t,x):
            
            vec = torch.tensor([[1.,1e-4,1.]]).repeat(len(t),1)
            
            input = torch.cat((t,x),dim=1)
            #lift = torch.sin(self.lift(input)) #See how it performs without the sin, so with self.act
            lift = self.act(self.lift(input))
            for i in range(self.nlayers):
                lift = self.act(self.linears[i](lift))
            if self.system=="Robert":
                
                return self.proj(lift) * vec
            else:
                return self.proj(lift)
       
        def forward(self,t,x):
            
            t = t.reshape(-1,1)
            x = x.reshape(-1,self.d)
                        
            res = x + self.parametric(t,x) - self.parametric(torch.zeros_like(t),x)
            return res
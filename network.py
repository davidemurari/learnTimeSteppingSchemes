import torch
import torch.nn as nn
import torch.nn.functional as F


class activation(nn.Module):
        def __init__(self,):
            super().__init__()
        def forward(self,x):
            return torch.tanh(x)

def mat_batch_mats(M,mats):
    return torch.einsum('ij,bjk->bik',M,mats)

class network(nn.Module):
        def __init__(self,d=3,system="Robert",num_hidden=20,dtype=torch.float32):
            super().__init__()
    
            self.input_dim = d
            self.output_dim = 3
            
            self.nlayers = 3
            
            self.system = system
            
            self.act = lambda x : torch.tanh(x)
            
            self.H = num_hidden

            self.lift = nn.Linear(self.input_dim+1,self.H,dtype=dtype)
            self.linears = nn.ModuleList([nn.Linear(self.H,self.H,dtype=dtype) for _ in range(self.nlayers)])
            self.proj = nn.Linear(self.H,self.output_dim,dtype=dtype)
                        
            self.f = lambda t : 1-torch.exp(-t)

        def parametric(self,t,x):
            
            t = t / 10
            
            input = torch.cat((t,x),dim=1)
            lift = torch.sin(self.lift(input))
            for i in range(self.nlayers):
                lift = self.act(self.linears[i](lift))
            return self.proj(lift)
       
        def forward(self,t,x):
            
            t = t.reshape(-1,1)
            x = x.reshape(-1,self.input_dim)
                        
            res = x + self.parametric(t,x) - self.parametric(torch.zeros_like(t),x)
            return res
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
        def __init__(self,d,system="Robert",num_hidden=20):
            super().__init__()
            
            self.num_hidden = num_hidden
            
            self.d = d
            
            self.nlayers = 0
            
            self.input_dim = 3
            self.output_dim = 3
            
            self.system = system
            
            self.lift = nn.Linear(self.d+1,self.num_hidden)
            self.linears = nn.ModuleList([nn.Linear(self.num_hidden,self.num_hidden) for _ in range(self.nlayers)]) 
            self.proj = nn.Linear(self.num_hidden,d,bias=False)         
            self.nl = activation()

        def parametric(self,t,x):
            
            input = torch.cat((t,x),dim=1)
            
            input = self.nl(self.lift(input))
            for i in range(self.nlayers):
                input = self.nl(self.linears[i](input))
            update = self.proj(input)
            return update
       
        def forward(self,t,x):
            
            t = t.reshape(-1,1)
            x = x.reshape(-1,self.d)
            
            update = self.parametric(t,x)
            t0 = torch.zeros_like(t)
            update_0 = self.parametric(t0,x)
                
            corrected_update = update - update_0
            if self.system=="Robert":
                corrected_update[:,1] *= 1e-4
            return x + corrected_update
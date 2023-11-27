import torch
import torch.nn as nn
import torch.nn.functional as F

class activation(nn.Module):
        def __init__(self,):
            super().__init__()
        def derivative(self,x):
            return 1-self.forward(x)**2
        def forward(self,x):
            return torch.sigmoid(x)*x

def mat_batch_mats(M,mats):
    return torch.einsum('ij,bjk->bik',M,mats)

class network(nn.Module):
        def __init__(self,t_0,t_1,d,num_hidden=10):
            super().__init__()
            
            self.num_hidden = num_hidden
            self.t_0 = t_0
            self.t_1 = t_1
            
            self.d = d
            
            self.nlayers = 1
            
            self.input_dim = 3
            self.output_dim = 3
            
            self.lift = nn.Linear(self.d+1,self.num_hidden)
            self.linears = nn.ModuleList([nn.Linear(self.num_hidden,self.num_hidden) for _ in range(self.nlayers)]) 
            self.proj = nn.Linear(self.num_hidden,d,bias=False)         
            self.nl = activation()

        def parametric(self,t,x):
            
            input = torch.cat((t,x),dim=1)
            
            input = torch.sin(torch.pi * 2 * self.lift(input))
            for i in range(self.nlayers):
                input = self.nl(self.linears[i](input))
            update = self.proj(input)
            return update
        
        
        
        def forward(self,t,x):
            
            update = self.parametric(t,x)
            t0 = torch.zeros_like(t)
            update_0 = self.parametric(t0,x)
                
            corrected_update = update - update_0
            corrected_update[:,1] *= 1e-4
            return x + corrected_update
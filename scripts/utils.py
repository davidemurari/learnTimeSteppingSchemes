from scipy.integrate import solve_ivp
from scripts.dynamics import *
import time as time_lib

def act(t,w,b,act_name="Tanh"):
    if act_name=="Tanh":
        return np.tanh(w*t+b), (1-np.tanh(w*t+b)**2)*w #function at the node, derivative at the node    
    elif act_name=="Sigmoid":
        func = lambda x : 1/(1+np.exp(-x))
        func_p = lambda x : func(x)*(1-func(x))
        return func(w*t+b), func_p(w*t+b)*w
    else: #sin
        func = lambda x : np.sin(x)
        func_p = lambda x: np.cos(x)
        return func(w*t+b), func_p(w*t+b)*w
    
class flowMap:
    def __init__(self,y0,initial_proj,weight,bias,dt=1,n_t=5,n_x=10,L=10,LB=-1.,UB=1.,system="Rober",act_name="Tanh",verbose=False):
        
        self.system = system
        self.vec = vecField(system)
        self.act = lambda t,w,b : act(t,w,b,act_name=act_name)
        self.dt = dt
        self.d = len(y0) #dimension phase space
        
        self.verbose = verbose
        
        self.n_x = n_x
        self.h = np.zeros((n_x,L))
        self.hd = np.zeros((n_x,L))
        
        self.iter = 0
        
        self.y0 = y0
        self.y0_supp = y0
        
        self.L = L #number of neurons
        self.LB = LB #Lower boundary for weight and bias sampling 
        self.UB = UB #Upper boundary for weight and bias sampling
        
        if len(weight)==0:
            self.weight = np.random.uniform(low=self.LB,high=self.UB,size=(self.L))
        else:
            self.weight = weight
        if len(bias)==0:
            self.bias = np.random.uniform(low=self.LB,high=self.UB,size=(self.L))
        else:
            self.bias = bias
        
        self.n_t = n_t
        if system=="Rober":
            self.t_tot = np.logspace(-5,np.log10(dt),self.n_t)
        else:
            self.t_tot = np.linspace(0,dt,self.n_t)
        self.x = np.linspace(0,dt,self.n_x)

        for i in range(n_x):
            self.h[i], self.hd[i] = self.act(self.x[i],self.weight,self.bias)

        self.h0 = self.h[0] #at the initial time, i.e. at x=0.
        self.hf = self.h[-1]
        self.hd0 = self.hd[0]
        self.hdf = self.hd[-1]
        
        self.computational_time = None
        
        #vv = np.concatenate((np.ones(self.L),np.zeros(2*self.L)),axis=0).reshape(1,-1)
        #vv = np.kron(self.y0,np.ones(self.L)).reshape(1,-1) #[y0_x,y0_x,...,y0_x,y0_y,...y0_y,y0_z,y0_z,...,y0_z] in case d=3
        self.computed_projection_matrices = np.tile(initial_proj,(self.n_t-1,1)) #one per time subintreval
        self.computed_initial_conditions = np.zeros((self.n_t-1,self.d)) #one per time subinterval
        self.training_err_vec = np.zeros((self.n_t,1))
        self.sol = np.zeros((self.n_t,self.d))
        
    
    def to_mat(self,w):    
        return w.reshape((self.L,self.d),order='F') #order F means it reshapes by columns
        #return np.concatenate((w[:self.L].reshape(-1,1),w[self.L:2*self.L].reshape(-1,1),w[2*self.L:].reshape(-1,1)),axis=1)
    
    def residual(self,c_i,xi_i):
        y = (self.h-self.h0)@self.to_mat(xi_i) + self.y0_supp.reshape(1,-1)
        vecValue = self.vec.eval(y)
        y_dot = c_i * self.hd @ self.to_mat(xi_i)
        Loss = (y_dot-vecValue)
        return Loss.reshape(-1,order='F')
    
    def re(self,a,H):
        return np.einsum('i,ij->ij',a,H)
    
    def jac_residual(self,c_i,xi_i):
        
        y = (np.einsum('jk,ik->ij',self.h-self.h0,self.to_mat(xi_i).T) + self.y0_supp.reshape(-1,1)).T
        H = self.h - self.h0
                    
        if self.system=="Rober":
            #Verified
            y1,y2,y3 = y[:,0],y[:,1],y[:,2]
            k1,k2,k3 = self.vec.k1, self.vec.k2, self.vec.k3
            zz = np.zeros((self.n_x,self.L))
            row1 = np.concatenate((c_i*self.hd+k1*H,-k3*self.re(y3,H),-k3*self.re(y2,H)),axis=1)
            row2 = np.concatenate((-k1*H,c_i*self.hd+self.re(2*k2*y2+k3*y3,H),k3*self.re(y2,H)),axis=1)
            row3 = np.concatenate((zz,-2*k2*self.re(y2,H),c_i*self.hd),axis=1)
            return np.concatenate((row1,row2,row3),axis=0)
        
        elif self.system=="SIR":
            #Verified
            y1,y2,y3 = y[:,0],y[:,1],y[:,2]
            zz = np.zeros((self.n_x,self.L))
            beta,gamma,N = self.vec.beta, self.vec.gamma, self.vec.N
            row1 = np.concatenate((c_i*self.hd+beta*self.re(y2,H)/N,beta*self.re(y1,H)/N,zz),axis=1)
            row2 = np.concatenate((-beta*self.re(y2,H)/N,c_i*self.hd-beta*self.re(y1,H)/N+gamma*H,zz),axis=1)
            row3 = np.concatenate((zz,-gamma*H,c_i*self.hd),axis=1)
            return np.concatenate((row1,row2,row3),axis=0)
    
        elif self.system=="Brusselator":
            #Verified
            xx,yy = y[:,0],y[:,1]
            zz = np.zeros((self.n_x,self.L))
            A,B = self.vec.A, self.vec.B
            
            row1 = np.concatenate((c_i*self.hd-2*self.re(xx*yy,H)+(B+1)*H,-self.re(xx**2,H)),axis=1)
            row2 = np.concatenate((-B*H+2*self.re(xx*yy,H),c_i*self.hd+self.re(xx**2,H)),axis=1)
            return np.concatenate((row1,row2),axis=0)
    
        elif self.system=="Arenstorf":
            #To implement
            xx,xxp,yy,yyp = y[:,0],y[:,1],y[:,2],y[:,3]
            zz = np.zeros((self.n_x,self.L))
            a,b = self.vec.a,self.vec.b
            
            D1 = ((xx+a)**2+yy**2)**(3/2)
            D2 = ((xx-b)**2+yy**2)**(3/2)
            D1_dx = 3*D1**(1/3)*(xx+a)
            D2_dx = 3*D2**(1/3)*(xx-b)
            D1_dy = 3*D1**(1/3)*yy
            D2_dy = 3*D2**(1/3)*yy
            
           
            dxpp_dx = 1.-b/D1+b*(xx+a)*D1_dx/D1**2 - a/D2 + a*(xx-b)*D2_dx/D2**2
            dxpp_dy = b*(xx+a)*D1_dy/D1**2 + a*(xx-b)*D2_dy/D2**2
            
            dypp_dx = yy*(b*D1_dx/D1**2+a*D2_dx/D2**2)
            dypp_dy = 1.-b/D1+b*yy*D1_dy/D1**2-a/D2+a*yy*D2_dy/D2**2
            
            row1 = np.concatenate((c_i*self.hd,-H,zz,zz),axis=1)
            row2 = np.concatenate((-self.re(dxpp_dx,H),c_i*self.hd,-self.re(dxpp_dy,H),-2*H),axis=1)
            row3 = np.concatenate((zz,zz,c_i*self.hd,-H),axis=1)
            row4 = np.concatenate((-self.re(dypp_dx,H),-2*H,-self.re(dypp_dy,H),c_i*self.hd),axis=1)
            return np.concatenate((row1,row2,row3,row4),axis=0)
        
        elif self.system=="Lorenz":
            #Verified
            y1,y2,y3 = y[:,0],y[:,1],y[:,2]
            zz = np.zeros((self.n_x,self.L))
            sigma,r,b = self.vec.sigma, self.vec.r, self.vec.b
            row1 = np.concatenate((c_i*self.hd+sigma*H,-sigma*H,zz),axis=1)
            row2 = np.concatenate((self.re(y3,H)-r*H,c_i*self.hd+H,self.re(y1,H)),axis=1)
            row3 = np.concatenate((-self.re(y2,H),-self.re(y1,H),c_i*self.hd+b*H),axis=1)
            return np.concatenate((row1,row2,row3),axis=0)

        else:
            pass
    
    def approximate_flow_map(self,IterMax=100,IterTol=1e-4):
        
        self.training_err_vec[0] = 0.        
        self.sol[0] = self.y0_supp
        
        initial_time = time_lib.time()

        for i in range(self.n_t-1):
            
            self.iter = 1 #Just added
            
            c_i = self.dt / (self.t_tot[i+1]-self.t_tot[i])
            xi_i = self.computed_projection_matrices[i] 
            Loss = self.residual(c_i,xi_i)
            l2 = [2.,1]
            
            self.computed_initial_conditions[i] = self.y0_supp
                
            while np.abs(l2[1])>IterTol and self.iter<IterMax and np.abs(l2[0]-l2[1])>IterTol:
                
                l2[0] = l2[1] #this says that we suppose to converge after this block of code runs
                JJ = self.jac_residual(c_i,xi_i)
                dxi = np.linalg.lstsq(JJ,Loss,rcond=None)[0]
                xi_i = xi_i - dxi
                Loss = self.residual(c_i,xi_i)
                l2[1] = np.linalg.norm(Loss,ord=2) #This is used as a stopping criterion
                
                self.iter+=1
                            
            #Store the computed result
            self.computed_projection_matrices[i] = xi_i
            
            y = (np.einsum('jk,ik->ij',self.h-self.h0,xi_i.reshape(len(self.y0),self.L)) + self.y0_supp.reshape(-1,1)).T
            self.y0_supp = y[-1]
            self.sol[i+1] = self.y0_supp
            
            self.training_err_vec[i+1] = np.sqrt(np.mean(Loss**2))

        final_time = time_lib.time()
        

        self.computational_time = final_time-initial_time
        if self.verbose:
            print(f"Training complete. Required time {self.computational_time}")
    
    def analyticalApproximateSolution(self,t):
        j = np.searchsorted(self.t_tot,t,side='left') #determines the index of the largest number in t_tot that is smaller than t
        #In other words, it finds where to place t in t_tot in order to preserve its increasing ordering
          
        y_0 = self.sol[max(0,j-1)]        
        jp = 1 if j==0 else j
        x = np.array([(t - self.t_tot[max(0,j-1)]) / (self.t_tot[jp]-self.t_tot[max(0,j-1)])])
        
        h,_ = act(x,self.weight,self.bias)
        h0,_ = act(0*x,self.weight,self.bias)
        y = (h-h0)@self.to_mat(self.computed_projection_matrices[max(0,j-1)]) + y_0
        return y
    
    def plotOverTimeRange(self,time):
        sol_approximation = np.zeros((self.d,len(time)))
        for i,t in enumerate(time):
            sol_approximation[:,i] = self.analyticalApproximateSolution(t)
        return sol_approximation
    
def getCoarse(time,data,previous=[],networks=[]):
    
    LB = data["LB"]
    UB = data["UB"]
    L = data["L"]
    y0 = data["y0"]
    weight = data["weight"]
    bias = data["bias"]
    n_x = data["n_x"]
    n_t = data["n_t"]
    system = data["system"]
    y0 = data["y0"]
    
    dts = np.diff(time)
    
    coarse_approx = np.zeros((len(time),len(y0)))
    coarse_approx[0] = y0
    
    initial_proj = np.ones(len(y0)*L).reshape(1,-1)#np.kron(y0,np.ones(L)).reshape(1,-1)
    
    if len(previous)==0:
        for i in range(len(time)-1):
            flow = flowMap(y0=coarse_approx[i],initial_proj=initial_proj,weight=weight,bias=bias,dt=dts[i],n_t=n_t,n_x=n_x,L=L,LB=LB,UB=UB,system=system,act_name="Tanh")
            flow.approximate_flow_map()
            coarse_approx[i+1] = flow.analyticalApproximateSolution(dts[i])
            networks.append(flow)

    else:
        for i in range(len(time)-1):
            flow = flowMap(y0=previous[i],initial_proj=initial_proj,weight=weight,bias=bias,dt=dts[i],n_t=n_t,n_x=n_x,L=L,LB=LB,UB=UB,system=system,act_name="Tanh")
            flow.approximate_flow_map()
            coarse_approx[i+1] = flow.analyticalApproximateSolution(dts[i])
            networks[i] = flow
            
    return coarse_approx
    
def getNextCoarse(time,y,i,data,networks=[]):
    
    dts = np.diff(time)
    
    LB = data["LB"]
    UB = data["UB"]
    L = data["L"]
    weight = data["weight"]
    bias = data["bias"]
    n_x = data["n_x"]
    n_t = data["n_t"]
    system = data["system"]
    y0 = data["y0"]
    
    initial_proj = np.ones(len(y0)*L).reshape(1,-1)#np.kron(y0,np.ones(L)).reshape(1,-1)
    
    flow = flowMap(y0=y,initial_proj=initial_proj,weight=weight,bias=bias,dt=dts[i],n_t=n_t,n_x=n_x,L=L,LB=LB,UB=UB,system=system,act_name="Tanh")
    flow.approximate_flow_map()
    networks[i] = flow
    return flow.analyticalApproximateSolution(dts[i])
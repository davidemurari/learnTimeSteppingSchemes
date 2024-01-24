from scipy.integrate import solve_ivp
from dynamics import *
import time as time_lib

f = lambda t,y : np.array([
    -0.04*y[0] + 1e4 * y[1] * y[2],
    0.04*y[0] - 1e4 * y[1] * y[2] - 3e7*y[1]**2,
    3e7*y[1]**2
])

def fineIntegratorParallel(coarse_values_parareal,i,dts):
        return solve_ivp(f,[0,dts[i]], y0=coarse_values_parareal[i], method = 'Radau', atol=1e-4, rtol=1e-4).y.T[-1]

class vecField:
    def __init__(self,system="Rober"):
        self.system = system 
        if self.system=="Rober":
            self.k1 = 0.04
            self.k2 = 3e7
            self.k3 = 1e4
    
    def eval(self,y):
        if self.system=="Rober":
            # Chemical parameters definition
            
            if len(y.shape)==2:
                y1,y2,y3 = y[:,0:1],y[:,1:2],y[:,2:3]
                return np.concatenate([
                    -self.k1*y1 + self.k3*y2*y3,
                    self.k1*y1 - self.k2*(y2**2) - self.k3*y2*y3,
                    self.k2*(y2**2)
                ],axis=1)
            else:
                y1,y2,y3 = y[0],y[1],y[2]
                return np.array([
                -self.k1*y1 + self.k3*y2*y3,
                self.k1*y1 - self.k2*(y2**2) - self.k3*y2*y3,
                self.k2*(y2**2)
            ])
        else:
            pass

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
    def __init__(self,y0,weight,bias,dt=1,n_t=5,n_x=10,L=10,LB=-1.,UB=1.,system="Rober",act_name="Tanh",verbose=False):
        
        self.system = system
        self.vec = vecField(system)
        self.act = lambda t,w,b : act(t,w,b,act_name="Tanh")
        self.dt = dt
        
        self.verbose = verbose
        
        self.n_x = n_x
        self.h = np.zeros((n_x,L))
        self.hd = np.zeros((n_x,L))
        
        self.iter = 0
        
        if len(y0)==0:
            self.y0 = np.random.randn(3)
            self.y0_supp = self.y0.copy()
            if self.verbose:
                print("No initial condition provided. Set it to random.")
        else:
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
        self.t_tot = np.logspace(-5,np.log10(dt),self.n_t)
        self.x = np.linspace(0,dt,self.n_x)

        for i in range(n_x):
            self.h[i], self.hd[i] = self.act(self.x[i],self.weight,self.bias)

        self.h0 = self.h[0] #at the initial time, i.e. at x=0.
        self.hf = self.h[-1]
        self.hd0 = self.hd[0]
        self.hdf = self.hd[-1]
        
        self.computational_time = None
        
        vv = np.concatenate((np.ones(self.L),np.zeros(2*self.L)),axis=0).reshape(1,-1)
        self.computed_projection_matrices = np.tile(vv,(self.n_t-1,1)) #one per time subintreval
        self.computed_initial_conditions = np.zeros((self.n_t-1,3)) #one per time subinterval
        self.training_err_vec = np.zeros((self.n_t,1))
        self.sol = np.zeros((self.n_t,3))
        
    
    def to_mat(self,w):
        return np.concatenate((w[:self.L].reshape(-1,1),w[self.L:2*self.L].reshape(-1,1),w[2*self.L:].reshape(-1,1)),axis=1)
    
    def residual(self,c_i,xi_i):
        y = (self.h-self.h0)@self.to_mat(xi_i) + self.y0_supp.reshape(1,-1)
        vecValue = self.vec.eval(y)
        y_dot = c_i * self.hd @ self.to_mat(xi_i)
        Loss = (y_dot-vecValue)
        return np.concatenate((Loss[:,0],Loss[:,1],Loss[:,2]),axis=0)
    
    def re(self,a,H):
        return np.einsum('i,ij->ij',a,H)
    
    def jac_residual(self,c_i,xi_i):
        
        if self.system=="Rober":
            
            k1,k2,k3 = self.vec.k1, self.vec.k2, self.vec.k3
            
            y = (np.einsum('jk,ik->ij',self.h-self.h0,self.to_mat(xi_i).T) + self.y0_supp.reshape(-1,1)).T
            y1,y2,y3 = y[:,0],y[:,1],y[:,2]
            zz = np.zeros((self.n_x,self.L))
            row1 = np.concatenate((c_i*self.hd+k1*(self.h-self.h0),-k3*self.re(y3,self.h-self.h0),-k3*self.re(y2,self.h-self.h0)),axis=1)
            row2 = np.concatenate((-k1*(self.h-self.h0),c_i*self.hd+self.re(2*k2*y2+k3*y3,self.h-self.h0),k3*self.re(y2,self.h-self.h0)),axis=1)
            row3 = np.concatenate((zz,-2*k2*self.re(y2,self.h-self.h0),c_i*self.hd),axis=1)

            JJ = np.concatenate((row1,row2,row3),axis=0)
            return JJ
        else:
            pass
    
    def approximate_flow_map(self,IterMax=100,IterTol=1e-6):
        
        self.training_err_vec[0] = 0.        
        self.sol[0] = self.y0_supp
        
        initial_time = time_lib.time()

        for i in range(self.n_t-1):
            
            c_i = self.dt / (self.t_tot[i+1]-self.t_tot[i]) #1/dt_i
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
            
            y = (np.einsum('jk,ik->ij',self.h-self.h0,xi_i.reshape(3,self.L)) + self.y0_supp.reshape(-1,1)).T
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
          
        xi_1_i = self.computed_projection_matrices[max(0,j-1),:self.L]
        xi_2_i = self.computed_projection_matrices[max(0,j-1),self.L:2*self.L]
        xi_3_i = self.computed_projection_matrices[max(0,j-1),2*self.L:]
            
        y1_0 = self.sol[max(0,j-1),0]
        y2_0 = self.sol[max(0,j-1),1]
        y3_0 = self.sol[max(0,j-1),2]
        
        jp = 1 if j==0 else j
        x = np.array([(t - self.t_tot[max(0,j-1)]) / (self.t_tot[jp]-self.t_tot[max(0,j-1)])])
        
        h,_ = act(x,self.weight,self.bias)
        h0,_ = act(0*x,self.weight,self.bias)
                
        y1_sol = np.dot(h-h0,xi_1_i) + y1_0
        y2_sol = np.dot(h-h0,xi_2_i) + y2_0
        y3_sol = np.dot(h-h0,xi_3_i) + y3_0
            
        return np.array([y1_sol,y2_sol,y3_sol])
    
    def plotOverTimeRange(self,time):
        sol_approximation = np.zeros((3,len(time)))
        for i,t in enumerate(time):
            sol_approximation[:,i] = self.analyticalApproximateSolution(t)
        return sol_approximation
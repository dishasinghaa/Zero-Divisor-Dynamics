import numpy as np, time
store = np.load('/home/claude/probe_store_fixed.npy', allow_pickle=True).item()

H = 32
class Net:
    def __init__(self,H): self.H=H
    def params(self): return [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
    def forward(self,X):
        self.X=X; self.z1=X@self.W1+self.b1; self.h1=np.tanh(self.z1)
        self.z2=self.h1@self.W2+self.b2; self.h2=np.tanh(self.z2)
        return self.h2@self.W3+self.b3
    def loss_and_grads(self,X,Y):
        P=self.forward(X); n=X.shape[0]; diff=P-Y; loss=np.mean(diff**2)
        dP=2*diff/(n*2)
        gW3=self.h2.T@dP; gb3=dP.sum(0)
        dh2=dP@self.W3.T; dz2=dh2*(1-self.h2**2)
        gW2=self.h1.T@dz2; gb2=dz2.sum(0)
        dh1=dz2@self.W2.T; dz1=dh1*(1-self.h1**2)
        gW1=X.T@dz1; gb1=dz1.sum(0)
        return loss,[gW1,gb1,gW2,gb2,gW3,gb3]

shapes = [(2,H),(H,),(H,H),(H,),(H,2),(2,)]
sizes = [int(np.prod(s)) for s in shapes]

def make_fn(X,Y):
    net = Net(H)
    def loss_and_grad(theta):
        arrs=[]; i=0
        for s,n in zip(shapes,sizes): arrs.append(theta[i:i+n].reshape(s)); i+=n
        net.W1,net.b1,net.W2,net.b2,net.W3,net.b3 = arrs
        loss, grads = net.loss_and_grads(X,Y)
        return loss, np.concatenate([g.ravel() for g in grads])
    return loss_and_grad

def sgld_llc(fn, w0, n, gamma, num_steps, burn_in, lr, num_chains, seed_base=0):
    beta = 1.0/np.log(n)
    loss0,_ = fn(w0)
    means=[]
    for c in range(num_chains):
        rng=np.random.default_rng(seed_base+c); theta=w0.copy(); coll=[]
        for t in range(num_steps):
            loss,grad = fn(theta)
            tot = n*beta*grad + gamma*(theta-w0)
            theta = theta - 0.5*lr*tot + rng.normal(0,1,size=theta.shape)*np.sqrt(lr)
            if t>=burn_in: coll.append(loss)
        means.append(np.mean(coll))
    llc = n*beta*(np.array(means)-loss0)
    return llc.mean(), llc.std(), beta

d = store['x_star      (NON-HYPERBOLIC)']['w0'].size
xs = store['x_star      (NON-HYPERBOLIC)']
fn = make_fn(xs['X'], xs['Y'])
n = xs['X'].shape[0]

print(f"d={d}  n={n}  reference lambda=d/2={d/2:.1f}")
print("Calibrating gamma at x* probe (short chains for speed)...")
t0=time.time()
for gamma in [0.5, 2.0, 10.0, 50.0]:
    llc, sd, beta = sgld_llc(fn, xs['w0'], n, gamma=gamma, num_steps=150, burn_in=50,
                              lr=1e-7, num_chains=3, seed_base=0)
    print(f"  gamma={gamma:6.1f}  lambda_hat={llc:9.2f} +/- {sd:6.2f}   ({time.time()-t0:.0f}s elapsed)")

print()
print("lr sweep at gamma=1.0 (find where the chain actually MOVES, without blowing up)...")
loss0,_ = fn(xs['w0'])
print(f"  loss at w0 = {loss0:.4e}")
for lr in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
    rng=np.random.default_rng(0); theta=xs['w0'].copy()
    trace=[]
    for t in range(150):
        loss,grad = fn(theta)
        tot = n*(1/np.log(n))*grad + 1.0*(theta-xs['w0'])
        theta = theta - 0.5*lr*tot + rng.normal(0,1,size=theta.shape)*np.sqrt(lr)
        trace.append(loss)
    print(f"  lr={lr:.0e}   loss: start={trace[0]:.3e}  end={trace[-1]:.3e}  max={max(trace):.3e}  "
          f"{'DIVERGED' if trace[-1]>1e3 or not np.isfinite(trace[-1]) else ''}")

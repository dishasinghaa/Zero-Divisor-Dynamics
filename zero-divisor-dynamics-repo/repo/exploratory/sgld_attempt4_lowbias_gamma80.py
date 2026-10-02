"""
Phase 4, least-biased regime: gamma/(n*beta) ~ 0.39, matching the toy
validation's PROVEN-unbiased range (gamma=1 there, at n*beta=66.8, ratio~0.015-1).
Costs ~10x more steps than the gamma=8000 run because lr is 100x smaller
(smaller noise steps -> slower exploration -> needs more of them).
This WILL take a while -- run it as a script, not interactively, and let it finish.
"""
import numpy as np, time

store = np.load('probe_store_fixed.npy', allow_pickle=True).item()
H = 32
shapes = [(2,H),(H,),(H,H),(H,),(H,2),(2,)]
sizes = [int(np.prod(s)) for s in shapes]

class Net:
    def loss_and_grads(self,X,Y):
        z1=X@self.W1+self.b1; h1=np.tanh(z1); z2=h1@self.W2+self.b2; h2=np.tanh(z2)
        P=h2@self.W3+self.b3; n=X.shape[0]; diff=P-Y; loss=np.mean(diff**2)
        dP=2*diff/(n*2)
        gW3=h2.T@dP; gb3=dP.sum(0)
        dh2=dP@self.W3.T; dz2=dh2*(1-h2**2); gW2=h1.T@dz2; gb2=dz2.sum(0)
        dh1=dz2@self.W2.T; dz1=dh1*(1-h1**2); gW1=X.T@dz1; gb1=dz1.sum(0)
        return loss,[gW1,gb1,gW2,gb2,gW3,gb3]

def make_fn(X,Y):
    net=Net()
    def fn(theta):
        arrs=[]; i=0
        for s,n in zip(shapes,sizes): arrs.append(theta[i:i+n].reshape(s)); i+=n
        net.W1,net.b1,net.W2,net.b2,net.W3,net.b3=arrs
        loss,grads=net.loss_and_grads(X,Y)
        return loss, np.concatenate([g.ravel() for g in grads])
    return fn

def gelman_rubin(chains):
    m,n = chains.shape
    chain_means=chains.mean(axis=1); chain_vars=chains.var(axis=1,ddof=1)
    W=chain_vars.mean(); B=n*chain_means.var(ddof=1)
    var_hat=((n-1)/n)*W+B/n
    return np.sqrt(var_hat/W) if W>0 else np.inf

def sgld_llc(fn,w0,n,gamma,num_steps,burn_in,lr,num_chains,seed_base=0):
    beta=1.0/np.log(n); loss0,_=fn(w0); post_burn=[]
    for c in range(num_chains):
        rng=np.random.default_rng(seed_base+c); theta=w0.copy(); coll=[]
        for t in range(num_steps):
            loss,grad=fn(theta)
            tot=n*beta*grad+gamma*(theta-w0)
            theta=theta-0.5*lr*tot+rng.normal(0,1,size=theta.shape)*np.sqrt(lr)
            if t>=burn_in: coll.append(loss)
        post_burn.append(coll)
    post_burn=np.array(post_burn)
    r_hat=gelman_rubin(post_burn)
    llc=n*beta*(post_burn.mean(axis=1)-loss0)
    return llc.mean(), llc.std(), llc, r_hat

# LESS-BIASED regime: gamma/n*beta ~ 0.39 (vs 39 before) -- but needs ~10x steps to mix
LR, GAMMA, STEPS, BURN, CHAINS = 1e-8, 80.0, 30000, 8000, 6
print(f"lr={LR} gamma={GAMMA} (gamma/n*beta~0.39, unbiased regime) steps={STEPS} burn_in={BURN} chains={CHAINS}")
print("This will take a while -- printing progress per probe as it finishes.")
print("="*92)
results = {}
for name, d in store.items():
    fn = make_fn(d['X'], d['Y']); n = d['X'].shape[0]
    t0=time.time()
    m, sd, per_chain, rh = sgld_llc(fn, d['w0'], n, GAMMA, STEPS, BURN, LR, CHAINS, seed_base=0)
    results[name] = dict(mean=m, sd=sd, per_chain=per_chain, r_hat=rh)
    flag = " OK" if rh < 1.05 else " <-- still poor, needs even more steps"
    tag = " <<< x*" if "NON-HYPERBOLIC" in name else ""
    print(f"{name:<32} lambda_hat={m:8.4f} +/-{sd:6.4f}  R-hat={rh:6.4f}{flag}{tag}  [{time.time()-t0:.0f}s]")

np.save('phase4_unbiased_results.npy', results, allow_pickle=True)
print("\nsaved -> phase4_unbiased_results.npy")

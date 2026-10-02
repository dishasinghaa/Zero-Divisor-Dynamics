"""
Phase 4 — SCALED UP with convergence diagnostics (R-hat).
Run this in a fresh notebook. Needs: numpy (nothing else).
IMPORTANT: needs 'probe_store_fixed.npy' in the SAME FOLDER as this script
           (that file was produced by phases23_fixed.py in an earlier step —
           re-run that first if you don't have it, or ask for it to be regenerated).
"""
import numpy as np, time

store = np.load('probe_store_fixed.npy', allow_pickle=True).item()
H = 32
shapes = [(2,H),(H,),(H,H),(H,),(H,2),(2,)]
sizes  = [int(np.prod(s)) for s in shapes]

class Net:
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

def make_fn(X,Y):
    net=Net()
    def fn(theta):
        arrs=[]; i=0
        for s,n in zip(shapes,sizes): arrs.append(theta[i:i+n].reshape(s)); i+=n
        net.W1,net.b1,net.W2,net.b2,net.W3,net.b3 = arrs
        loss,grads = net.loss_and_grads(X,Y)
        return loss, np.concatenate([g.ravel() for g in grads])
    return fn

def gelman_rubin(chains):
    """
    chains: array of shape (num_chains, num_samples) -- the post-burn-in loss
    trace from each chain. Returns R-hat. Values near 1.0 = good mixing.
    """
    m, n = chains.shape
    chain_means = chains.mean(axis=1)
    chain_vars  = chains.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n * chain_means.var(ddof=1)
    var_hat = ((n-1)/n)*W + B/n
    R_hat = np.sqrt(var_hat / W) if W > 0 else np.inf
    return R_hat

def sgld_llc_with_diagnostics(fn, w0, n, gamma, num_steps, burn_in, lr, num_chains, seed_base=0):
    beta = 1.0/np.log(n)
    loss0,_ = fn(w0)
    all_traces = []
    post_burn  = []
    for c in range(num_chains):
        rng = np.random.default_rng(seed_base+c)
        theta = w0.copy()
        trace = []
        for t in range(num_steps):
            loss, grad = fn(theta)
            tot = n*beta*grad + gamma*(theta-w0)
            theta = theta - 0.5*lr*tot + rng.normal(0,1,size=theta.shape)*np.sqrt(lr)
            trace.append(loss)
        all_traces.append(trace)
        post_burn.append(trace[burn_in:])

    post_burn = np.array(post_burn)
    r_hat = gelman_rubin(post_burn)
    chain_means = post_burn.mean(axis=1)
    llc_per_chain = n*beta*(chain_means - loss0)
    return llc_per_chain.mean(), llc_per_chain.std(), llc_per_chain, r_hat, np.array(all_traces)

# ---------------- SCALED-UP RUN ----------------
LR, GAMMA, STEPS, BURN, CHAINS = 1e-6, 1.0, 4000, 1500, 8
print(f"SCALED RUN: steps={STEPS} burn_in={BURN} chains={CHAINS}")
print("="*95)

results = {}
for name, d in store.items():
    fn = make_fn(d['X'], d['Y'])
    n  = d['X'].shape[0]
    t0 = time.time()
    mean_llc, sd_llc, per_chain, r_hat, traces = sgld_llc_with_diagnostics(
        fn, d['w0'], n, GAMMA, STEPS, BURN, LR, CHAINS, seed_base=0)
    results[name] = dict(mean=mean_llc, sd=sd_llc, per_chain=per_chain,
                          r_hat=r_hat, traces=traces)
    flag = "  <-- POOR MIXING, do not trust" if r_hat > 1.05 else "  OK"
    tag = " <<< x*" if "NON-HYPERBOLIC" in name else ""
    print(f"{name:<32} lambda_hat={mean_llc:8.4f} +/-{sd_llc:6.4f}  "
          f"R-hat={r_hat:6.3f}{flag}{tag}  [{time.time()-t0:.0f}s]")

np.save('phase4_scaled_results.npy', results, allow_pickle=True)
print("\nsaved -> phase4_scaled_results.npy")

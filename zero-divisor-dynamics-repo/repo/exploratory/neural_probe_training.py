"""
Phase 2 + 3, FIXED: (1) common loss threshold instead of fixed steps,
(2) per-probe target normalization so every probe is an equally-scaled task.
"""
import numpy as np

a1, a2, c12, c21 = 2.450, 1.420, 0.285, 0.340
OMEGA_STAR = 0.6339752836

def F(x):
    x1, x2 = x[..., 0], x[..., 1]
    d1 = x1*(a1 - x1**2) - c12*x1*x2 + OMEGA_STAR*x2
    d2 = x2*(a2 - x2**2) - c21*x1*x2 - OMEGA_STAR*x1
    return np.stack([d1, d2], axis=-1)

COMPARISON_SET = {
    'x_star      (NON-HYPERBOLIC)': np.array([-1.7248077365, -0.8044615332]),
    'sink        (control)'       : np.array([-1.06507,  1.49462]),
    'saddle_A    (control)'       : np.array([-0.45600,  1.33830]),
    'source      (control)'       : np.array([ 0.00000,  0.00000]),
    'saddle_B    (control)'       : np.array([ 0.28322, -1.21314]),
    'spiral_sink (control)'       : np.array([ 1.50873, -1.28505]),
}

def sample_probe(x0, eps, N, rng):
    r = eps*np.sqrt(rng.random(N)); theta = 2*np.pi*rng.random(N)
    pts = x0 + np.stack([r*np.cos(theta), r*np.sin(theta)], axis=-1)
    return pts, F(pts)

class ProbeNet:
    def __init__(self, H=32, rng=None):
        r = rng
        self.W1=r.normal(0,np.sqrt(2/2),(2,H)); self.b1=np.zeros(H)
        self.W2=r.normal(0,np.sqrt(2/H),(H,H)); self.b2=np.zeros(H)
        self.W3=r.normal(0,np.sqrt(2/H),(H,2)); self.b3=np.zeros(2)
    def params(self): return [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
    def get_flat(self): return np.concatenate([p.ravel() for p in self.params()])
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

def train_probe_to_threshold(X, Y, H=32, target_loss=1e-6, max_steps=12000, lr=3e-3, seed=0):
    rng=np.random.default_rng(seed); net=ProbeNet(H=H,rng=rng)
    m=[np.zeros_like(p) for p in net.params()]; v=[np.zeros_like(p) for p in net.params()]
    b1c,b2c,eps_=0.9,0.999,1e-8
    loss=np.inf
    for t in range(1,max_steps+1):
        loss,g=net.loss_and_grads(X,Y)
        for i,(p,gi) in enumerate(zip(net.params(),g)):
            m[i]=b1c*m[i]+(1-b1c)*gi; v[i]=b2c*v[i]+(1-b2c)*gi**2
            mh=m[i]/(1-b1c**t); vh=v[i]/(1-b2c**t)
            p-=lr*mh/(np.sqrt(vh)+eps_)
        if loss < target_loss: break
    return net, loss, t

if __name__ == "__main__":
    EPS, N, H, SEED, TARGET = 0.25, 1500, 32, 0, 3e-6
    print("="*94)
    print(f"PHASE 2+3 FIXED   eps={EPS}  N={N}  H={H}  target_loss={TARGET}  Omega*={OMEGA_STAR}")
    print("="*94)
    print(f"{'probe point':<32}{'Y scale(sigma)':>15}{'steps used':>12}{'raw final MSE':>15}{'norm MSE':>12}")
    print("-"*94)

    store={}
    for name, x0 in COMPARISON_SET.items():
        rng = np.random.default_rng(SEED)
        X, Y_raw = sample_probe(x0, EPS, N, rng)
        y_scale = Y_raw.std()                      # per-probe normalization
        Y = Y_raw / y_scale
        net, norm_loss, steps = train_probe_to_threshold(X, Y, H=H, target_loss=TARGET, seed=SEED)
        raw_loss = norm_loss * y_scale**2
        w0 = net.get_flat()
        store[name] = dict(x0=x0, X=X, Y=Y, y_scale=y_scale, w0=w0, loss=norm_loss, steps=steps)
        flag = "  <= hit cap, did not converge" if steps>=39999 else ""
        print(f"{name:<32}{y_scale:15.5f}{steps:12d}{raw_loss:15.3e}{norm_loss:12.3e}{flag}")

    print("-"*94)
    d = store['x_star      (NON-HYPERBOLIC)']['w0'].size
    print(f"d = {d}   regular-point reference lambda = d/2 = {d/2:.1f}")
    np.save('/home/claude/probe_store_fixed.npy', store, allow_pickle=True)
    print("saved -> probe_store_fixed.npy")

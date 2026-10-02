import numpy as np
from scipy.optimize import fsolve

a1,a2,c12,c21 = 2.450,1.420,0.285,0.340
Oms = 0.6339752835555

def F(x1,x2):
    d1 = x1*(a1-x1**2)-c12*x1*x2+Oms*x2
    d2 = x2*(a2-x2**2)-c21*x1*x2-Oms*x1
    return d1,d2

def Fvec(v): return np.array(F(v[0],v[1]))

# ============ STEP 1: blind equilibrium discovery (no x* hint anywhere) ============
print("STEP 1: blind sweep for ALL equilibria (no labels, no hints)")
print("="*70)
found = []
for gx in np.linspace(-3,3,40):
    for gy in np.linspace(-3,3,40):
        s,info,ier,msg = fsolve(Fvec,[gx,gy],full_output=True,xtol=1e-13)
        if ier==1 and np.max(np.abs(Fvec(s)))<1e-10:
            if not any(np.allclose(s,p,atol=1e-6) for p in found):
                found.append(s)
found = sorted(found, key=lambda p:(round(p[0],4),round(p[1],4)))
print(f"Found {len(found)} equilibria, unlabeled:")
for i,p in enumerate(found):
    print(f"  candidate_{i}: ({p[0]:+.5f}, {p[1]:+.5f})")

# ============ STEP 2: compute zeta exponent lambda at EVERY candidate, blind ============
def poly_features(u1,u2):
    return np.stack([np.ones_like(u1),u1,u2,u1**2,u1*u2,u2**2,
                      u1**3,u1**2*u2,u1*u2**2,u2**3], axis=-1)

def estimate_lambda(x0,y0, eps=0.15, N=2500, seed=0, ts=np.logspace(-6,-3,10)):
    rng = np.random.default_rng(seed)
    r = eps*np.sqrt(rng.random(N)); th = 2*np.pi*rng.random(N)
    u1 = r*np.cos(th); u2 = r*np.sin(th)
    d1,d2 = F(x0+u1, y0+u2)
    Phi = poly_features(u1,u2)
    c1,*_ = np.linalg.lstsq(Phi, d1, rcond=None)
    c2,*_ = np.linalg.lstsq(Phi, d2, rcond=None)
    gu = np.linspace(-eps,eps,260)
    GU1,GU2 = np.meshgrid(gu,gu)
    mask = GU1**2+GU2**2<=eps**2
    PhiG = poly_features(GU1.ravel(),GU2.ravel())
    Kg = ((PhiG@c1)**2+(PhiG@c2)**2).reshape(GU1.shape)
    Kg_in = Kg[mask]; cell=(gu[1]-gu[0])**2
    Vs = np.array([np.sum(Kg_in<=t)*cell for t in ts])
    valid = Vs>0
    if valid.sum()<4: return np.nan
    return np.polyfit(np.log(ts[valid]), np.log(Vs[valid]),1)[0]

print()
print("STEP 2: estimate lambda at every candidate, no labels used")
print("="*70)
lambdas = []
for i,p in enumerate(found):
    # average over a few seeds for stability
    vals = [estimate_lambda(p[0],p[1],seed=s) for s in range(5)]
    vals = [v for v in vals if not np.isnan(v)]
    m = np.mean(vals); sd=np.std(vals)
    lambdas.append(m)
    print(f"  candidate_{i} ({p[0]:+.4f},{p[1]:+.4f})   lambda = {m:.4f} +/- {sd:.4f}")

# ============ STEP 3: UNSUPERVISED outlier flagging -- no hint which is x* ============
lambdas = np.array(lambdas)
mu, sigma = lambdas.mean(), lambdas.std()
z = (lambdas-mu)/sigma
print()
print("STEP 3: automatic outlier flag (z-score < -1.5, i.e. unusually LOW lambda)")
print("="*70)
flagged = np.where(z < -1.5)[0]
for i,p in enumerate(found):
    tag = "  <<< FLAGGED AS ANOMALOUS" if i in flagged else ""
    print(f"  candidate_{i}  lambda={lambdas[i]:.4f}  z={z[i]:+.2f}{tag}")

print()
print(f"Flagged candidates: {[tuple(np.round(found[i],4)) for i in flagged]}")
print(f"Actual x* (never told to the algorithm): (-1.72481, -0.80446)")

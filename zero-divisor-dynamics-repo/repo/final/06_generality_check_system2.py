"""
Generality check: a STRUCTURALLY DIFFERENT system (trig nonlinearity instead
of cubic competition terms), with its own engineered saddle-node, run through
the EXACT SAME unmodified pipeline (same eps, N, ts, poly degree, z-threshold)
as the original system. No retuning. If this still works, it's real evidence
of a general method, not an artifact of one friendly equation.
"""
import numpy as np
from scipy.optimize import fsolve

# A DIFFERENT system: trig + quadratic coupling, structurally unlike the
# original cubic competition model
def F2(x1,x2,p):
    d1 = np.sin(x1) - 0.3*x1*x2 + p
    d2 = -x2 + 0.25*x1**2
    return d1,d2

def Fvec2(v,p): return np.array(F2(v[0],v[1],p))

def Jac2(x1,x2,p):
    return np.array([[np.cos(x1)-0.3*x2, -0.3*x1],[0.5*x1, -1.0]])

def aug2(v):
    x1,x2,p = v
    f1,f2 = F2(x1,x2,p)
    return [f1,f2,np.linalg.det(Jac2(x1,x2,p))]

print("STEP 0: engineer a saddle-node in this DIFFERENT system")
s = fsolve(aug2, [1.0, 0.3, -0.5], xtol=1e-14)
x1s,x2s,ps = s
print(f"  x* = ({x1s:.6f}, {x2s:.6f})   p* = {ps:.6f}")
print(f"  |F(x*)| = {np.max(np.abs(F2(x1s,x2s,ps))):.2e}")
ev = np.linalg.eigvals(Jac2(x1s,x2s,ps))
print(f"  eigenvalues: {ev}  (one should be ~0)")

# ============ Same blind pipeline, UNMODIFIED ============
def poly_features(u1,u2):
    return np.stack([np.ones_like(u1),u1,u2,u1**2,u1*u2,u2**2,
                      u1**3,u1**2*u2,u1*u2**2,u2**3], axis=-1)

def estimate_lambda(x0,y0, p, eps=0.15, N=2500, seed=0, ts=np.logspace(-6,-3,10)):
    rng = np.random.default_rng(seed)
    r = eps*np.sqrt(rng.random(N)); th = 2*np.pi*rng.random(N)
    u1 = r*np.cos(th); u2 = r*np.sin(th)
    d1,d2 = F2(x0+u1, y0+u2, p)
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

print("\nSTEP 1: blind equilibrium discovery at p* (no hints)")
found=[]
for gx in np.linspace(-3,3,40):
    for gy in np.linspace(-3,3,40):
        sv,info,ier,msg = fsolve(lambda v: Fvec2(v,ps),[gx,gy],full_output=True,xtol=1e-13)
        if ier==1 and np.max(np.abs(Fvec2(sv,ps)))<1e-10:
            if not any(np.allclose(sv,q,atol=1e-6) for q in found): found.append(sv)
found = sorted(found, key=lambda q:(round(q[0],4),round(q[1],4)))
print(f"  found {len(found)} equilibria")

print("\nSTEP 2+3: lambda at each, blind outlier flag (SAME z<-1.5 rule, untouched)")
lambdas=[]
for q in found:
    vals = [estimate_lambda(q[0],q[1],ps,seed=s) for s in range(5)]
    vals = [v for v in vals if not np.isnan(v)]
    lambdas.append(np.mean(vals))
lambdas=np.array(lambdas)
mu,sigma = lambdas.mean(), lambdas.std()
z=(lambdas-mu)/sigma
for i,q in enumerate(found):
    tag = "  <<< FLAGGED" if z[i]<-1.5 else ""
    print(f"  ({q[0]:+.4f},{q[1]:+.4f})  lambda={lambdas[i]:.4f}  z={z[i]:+.2f}{tag}")
print(f"\nActual engineered x* (never told to algorithm): ({x1s:.4f},{x2s:.4f})")

import numpy as np
from scipy.optimize import fsolve

a1,a2,c12,c21 = 2.450,1.420,0.285,0.340
OMS = 0.6339800000
def F(x1,x2,Om): return (x1*(a1-x1**2)-c12*x1*x2+Om*x2, x2*(a2-x2**2)-c21*x1*x2-Om*x1)
def J(x1,x2,Om):
    return np.array([[a1-3*x1**2-c12*x2,-c12*x1+Om],[-c21*x2-Om,a2-3*x2**2-c21*x1]])
def aug(v):
    x1,x2,Om=v; f1,f2=F(x1,x2,Om); return [f1,f2,np.linalg.det(J(x1,x2,Om))]

s = fsolve(aug,[-1.7248,-0.8045,0.634],xtol=1e-14)
x1s,x2s,Oms = s
print("REFINED x*:")
print(f"  x* = ({x1s:.10f}, {x2s:.10f})   Omega* = {Oms:.10f}")
print(f"  |F(x*)| = {np.max(np.abs(F(x1s,x2s,Oms))):.3e}")
Jm=J(x1s,x2s,Oms); ev,evec=np.linalg.eig(Jm)
print(f"  det J = {np.linalg.det(Jm):.3e}   tr J = {np.trace(Jm):.6f}")
print(f"  eigenvalues = {ev}")
i0=np.argmin(np.abs(ev))
print(f"  DEGENERATE (null) direction : {np.round(evec[:,i0],6)}   (lambda={ev[i0]:.2e})")
print(f"  contracting direction       : {np.round(evec[:,1-i0],6)}   (lambda={ev[1-i0]:.6f})")
print(f"  Omega shift from 0.620: {100*(Oms-0.620)/0.620:+.2f}%")

print("\n"+"="*74); print(f"FULL EQUILIBRIA LANDSCAPE at Omega* = {Oms:.6f}"); print("="*74)
sols=[]
for gx in np.linspace(-3,3,30):
    for gy in np.linspace(-3,3,30):
        p,info,ier,msg = fsolve(lambda v:F(v[0],v[1],Oms),[gx,gy],full_output=True)
        if ier==1 and np.max(np.abs(F(p[0],p[1],Oms)))<1e-10:
            if not any(np.allclose(p,q,atol=1e-5) for q in sols): sols.append(p)
sols=sorted(sols,key=lambda p:(round(p[0],4),round(p[1],4)))
print(f"{'point':>24} {'det':>10} {'trace':>9}  {'type':<15} {'|min Re(ev)|':>12}")
for p in sols:
    Jp=J(p[0],p[1],Oms); ev=np.linalg.eigvals(Jp); d=np.linalg.det(Jp); t=np.trace(Jp)
    if abs(ev[0].imag)>1e-9: kind="spiral sink" if ev[0].real<0 else "spiral source"
    elif ev[0].real*ev[1].real<0: kind="SADDLE"
    elif ev[0].real<0: kind="sink"
    else: kind="source"
    star = "  <<< x* (NON-HYPERBOLIC)" if np.allclose(p,[x1s,x2s],atol=1e-4) else ""
    print(f"({p[0]:+9.5f},{p[1]:+9.5f}) {d:+10.4f} {t:+9.4f}  {kind:<15} {np.min(np.abs(ev.real)):12.2e}{star}")
print(f"\n  total: {len(sols)} equilibria")

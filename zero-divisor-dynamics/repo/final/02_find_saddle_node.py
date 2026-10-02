import numpy as np
from scipy.optimize import fsolve

a1, a2, c12, c21 = 2.450, 1.420, 0.285, 0.340

def F(x1,x2,Om): return (x1*(a1-x1**2)-c12*x1*x2+Om*x2, x2*(a2-x2**2)-c21*x1*x2-Om*x1)
def J(x1,x2,Om):
    return np.array([[a1-3*x1**2-c12*x2, -c12*x1+Om],[-c21*x2-Om, a2-3*x2**2-c21*x1]])

def aug_det(v):
    x1,x2,Om = v; f1,f2 = F(x1,x2,Om)
    return [f1, f2, np.linalg.det(J(x1,x2,Om))]

def aug_tr(v):
    x1,x2,Om = v; f1,f2 = F(x1,x2,Om)
    return [f1, f2, np.trace(J(x1,x2,Om))]

base = [(-1.7408,-0.9055),(-1.7042,-0.7000),(-1.0795,1.4949),
        (-0.4401,1.3321),(0.2762,-1.2114),(1.5134,-1.2800)]

print("="*74); print("SADDLE-NODE CANDIDATES  (det J = 0)"); print("="*74)
found=[]
for (gx,gy) in base:
    for gOm in np.linspace(-4,4,33):
        s,info,ier,msg = fsolve(aug_det,[gx,gy,gOm],full_output=True)
        if ier==1 and np.max(np.abs(aug_det(s)))<1e-9:
            if not any(np.allclose(s,p,atol=1e-5) for p in found): found.append(s)
for s in found:
    x1,x2,Om = s; Jm=J(x1,x2,Om); ev=np.linalg.eigvals(Jm)
    print(f"  x*=({x1:+8.5f},{x2:+8.5f})  Omega*={Om:+8.5f}  det={np.linalg.det(Jm):+.2e} tr={np.trace(Jm):+7.4f}  ev={np.round(ev,6)}")

print()
print("="*74); print("HOPF CANDIDATES  (tr J = 0, need det > 0)"); print("="*74)
found2=[]
for (gx,gy) in base:
    for gOm in np.linspace(-4,4,33):
        s,info,ier,msg = fsolve(aug_tr,[gx,gy,gOm],full_output=True)
        if ier==1 and np.max(np.abs(aug_tr(s)))<1e-9:
            if not any(np.allclose(s,p,atol=1e-5) for p in found2): found2.append(s)
for s in found2:
    x1,x2,Om = s; Jm=J(x1,x2,Om); d=np.linalg.det(Jm); ev=np.linalg.eigvals(Jm)
    tag = "TRUE HOPF (center)" if d>0 else "det<0 -> not Hopf"
    print(f"  x*=({x1:+8.5f},{x2:+8.5f})  Omega*={Om:+8.5f}  det={d:+8.4f} tr={np.trace(Jm):+.2e}  {tag}  ev={np.round(ev,5)}")

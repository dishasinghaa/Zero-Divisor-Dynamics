import numpy as np
from scipy.optimize import fsolve, root

a1, a2, c12, c21 = 2.450, 1.420, 0.285, 0.340
OM0 = 0.620

def F(x1, x2, Om):
    return (x1*(a1 - x1**2) - c12*x1*x2 + Om*x2,
            x2*(a2 - x2**2) - c21*x1*x2 - Om*x1)

def J(x1, x2, Om):
    return np.array([[a1 - 3*x1**2 - c12*x2, -c12*x1 + Om],
                     [-c21*x2 - Om,          a2 - 3*x2**2 - c21*x1]])

# --- Step 1: baseline equilibria at Omega = 0.620 ---
print("="*70)
print("BASELINE EQUILIBRIA at Omega = 0.620")
print("="*70)
sols = []
for gx in np.linspace(-3, 3, 25):
    for gy in np.linspace(-3, 3, 25):
        s, info, ier, msg = fsolve(lambda v: F(v[0], v[1], OM0), [gx, gy], full_output=True)
        if ier == 1 and np.max(np.abs(F(s[0], s[1], OM0))) < 1e-10:
            if not any(np.allclose(s, p, atol=1e-6) for p in sols):
                sols.append(s)
sols = sorted(sols, key=lambda p: (round(p[0],4), round(p[1],4)))
for s in sols:
    ev = np.linalg.eigvals(J(s[0], s[1], OM0))
    det = np.linalg.det(J(s[0], s[1], OM0)); tr = np.trace(J(s[0], s[1], OM0))
    if np.iscomplexobj(ev) and abs(ev[0].imag) > 1e-9:
        kind = "spiral sink" if ev[0].real < 0 else "spiral source"
    elif ev[0].real*ev[1].real < 0: kind = "SADDLE"
    elif ev[0].real < 0: kind = "sink"
    else: kind = "source"
    print(f"  ({s[0]:+8.4f},{s[1]:+8.4f})  det={det:+9.4f} tr={tr:+8.4f}  {kind:14s} ev={np.round(ev,4)}")
print(f"\n  total equilibria found: {len(sols)}")

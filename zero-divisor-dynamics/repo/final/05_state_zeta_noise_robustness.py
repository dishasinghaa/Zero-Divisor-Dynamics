import numpy as np

a1,a2,c12,c21 = 2.450,1.420,0.285,0.340
Oms = 0.6339752835555
xs,ys = -1.7248077364591, -0.8044615332334   # x*
osx,osy = -1.06507, 1.49462                    # ordinary sink, for comparison

def F(x1,x2):
    d1 = x1*(a1-x1**2)-c12*x1*x2+Oms*x2
    d2 = x2*(a2-x2**2)-c21*x1*x2-Oms*x1
    return d1,d2

def poly_features(u1,u2):
    # degree-3 polynomial basis: 1,u1,u2,u1^2,u1u2,u2^2,u1^3,u1^2u2,u1u2^2,u2^3
    return np.stack([np.ones_like(u1),u1,u2,u1**2,u1*u2,u2**2,
                      u1**3,u1**2*u2,u1*u2**2,u2**3], axis=-1)

def fit_and_estimate_lambda(x0,y0, eps, N, rel_noise, rng, ts):
    r = eps*np.sqrt(rng.random(N)); th = 2*np.pi*rng.random(N)
    u1 = r*np.cos(th); u2 = r*np.sin(th)
    d1,d2 = F(x0+u1, y0+u2)
    # multiplicative relative noise on the OBSERVED field components
    scale = np.sqrt(d1**2+d2**2).mean() + 1e-12
    d1_obs = d1 + rel_noise*scale*rng.normal(size=N)
    d2_obs = d2 + rel_noise*scale*rng.normal(size=N)

    Phi = poly_features(u1,u2)
    c1,*_ = np.linalg.lstsq(Phi, d1_obs, rcond=None)
    c2,*_ = np.linalg.lstsq(Phi, d2_obs, rcond=None)

    # evaluate K_fit on a dense grid to estimate sublevel-set area
    gu = np.linspace(-eps,eps,300)
    GU1,GU2 = np.meshgrid(gu,gu)
    mask = GU1**2+GU2**2 <= eps**2
    PhiG = poly_features(GU1.ravel(),GU2.ravel())
    d1g = (PhiG@c1).reshape(GU1.shape); d2g=(PhiG@c2).reshape(GU1.shape)
    Kg = d1g**2+d2g**2
    Kg_in = Kg[mask]
    cell = (gu[1]-gu[0])**2

    Vs=[]
    for t in ts:
        Vs.append(np.sum(Kg_in<=t)*cell)
    Vs = np.array(Vs)
    valid = Vs>0
    if valid.sum()<4: return np.nan
    slope = np.polyfit(np.log(ts[valid]), np.log(Vs[valid]), 1)[0]
    return slope

ts = np.logspace(-6,-3,10)
EPS, N = 0.15, 2500
noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
BOOT = 15

print(f"{'noise':>8}  {'x* lambda (mean+-sd)':>24}  {'sink lambda (mean+-sd)':>24}  {'separation':>10}")
print("-"*78)
for nl in noise_levels:
    xs_lams=[]; sk_lams=[]
    for b in range(BOOT):
        rng = np.random.default_rng(1000+b)
        l1 = fit_and_estimate_lambda(xs,ys, EPS, N, nl, rng, ts)
        rng2 = np.random.default_rng(2000+b)
        l2 = fit_and_estimate_lambda(osx,osy, EPS, N, nl, rng2, ts)
        if not np.isnan(l1): xs_lams.append(l1)
        if not np.isnan(l2): sk_lams.append(l2)
    xs_lams=np.array(xs_lams); sk_lams=np.array(sk_lams)
    sep = (sk_lams.mean()-xs_lams.mean())
    print(f"{nl:8.3f}  {xs_lams.mean():10.4f} +/- {xs_lams.std():6.4f}   "
          f"{sk_lams.mean():10.4f} +/- {sk_lams.std():6.4f}   {sep:10.4f}")

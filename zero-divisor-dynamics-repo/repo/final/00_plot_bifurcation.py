import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

a1,a2,c12,c21 = 2.450,1.420,0.285,0.340
OM0, OMS = 0.620, 0.6339752836

def F(x1,x2,Om): return (x1*(a1-x1**2)-c12*x1*x2+Om*x2, x2*(a2-x2**2)-c21*x1*x2-Om*x1)
def J(x1,x2,Om): return np.array([[a1-3*x1**2-c12*x2,-c12*x1+Om],[-c21*x2-Om,a2-3*x2**2-c21*x1]])

def equilibria(Om):
    sols=[]
    for gx in np.linspace(-3,3,30):
        for gy in np.linspace(-3,3,30):
            p,info,ier,_=fsolve(lambda v:F(v[0],v[1],Om),[gx,gy],full_output=True)
            if ier==1 and np.max(np.abs(F(p[0],p[1],Om)))<1e-11:
                if not any(np.allclose(p,q,atol=1e-5) for q in sols): sols.append(p)
    return sols

def classify(p,Om):
    ev=np.linalg.eigvals(J(p[0],p[1],Om))
    if np.min(np.abs(ev.real))<1e-6 and np.max(np.abs(ev.imag))<1e-9: return 'degenerate'
    if abs(ev[0].imag)>1e-9: return 'spiral sink' if ev[0].real<0 else 'spiral source'
    if ev[0].real*ev[1].real<0: return 'saddle'
    return 'sink' if ev[0].real<0 else 'source'

COL={'sink':'#1f6fb4','saddle':'#d1495b','source':'#e09f3e','spiral sink':'#2a9d8f',
     'spiral source':'#9b5de5','degenerate':'#000000'}

fig,axes=plt.subplots(1,3,figsize=(17,5.6))
panels=[(OM0,'BEFORE:  $\\Omega = 0.620$\n7 equilibria — all hyperbolic',axes[0]),
        (OMS,'AT BIFURCATION:  $\\Omega^* = 0.6339753$\n6 equilibria — $x^*$ non-hyperbolic',axes[1])]

for Om,title,ax in panels:
    X,Y=np.meshgrid(np.linspace(-2.6,2.4,30),np.linspace(-2.2,2.2,30))
    U,V=F(X,Y,Om); M=np.hypot(U,V)
    ax.streamplot(X,Y,U,V,color=np.log1p(M),cmap='Greys',density=1.15,linewidth=0.7,arrowsize=0.75)
    for p in equilibria(Om):
        k=classify(p,Om)
        ax.scatter(*p,s=210 if k=='degenerate' else 95,c=COL[k],
                   edgecolors='k',zorder=6,marker='*' if k=='degenerate' else 'o',
                   linewidths=1.4 if k=='degenerate' else 0.8)
    ax.set_title(title,fontsize=11.5,fontweight='bold')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xlim(-2.6,2.4); ax.set_ylim(-2.2,2.2); ax.grid(alpha=0.13)

# annotate the merging pair
axes[0].annotate('sink\n(-1.741,-0.906)',xy=(-1.7408,-0.9055),xytext=(-2.45,-1.75),
   fontsize=8.2,arrowprops=dict(arrowstyle='->',lw=1.1,color='#1f6fb4'),color='#1f6fb4',fontweight='bold')
axes[0].annotate('saddle\n(-1.704,-0.700)',xy=(-1.7042,-0.7000),xytext=(-2.55,0.35),
   fontsize=8.2,arrowprops=dict(arrowstyle='->',lw=1.1,color='#d1495b'),color='#d1495b',fontweight='bold')
axes[1].annotate('$x^*$  COLLIDED\n(-1.7248,-0.8045)\n$\\det J=0$',xy=(-1.72481,-0.80446),xytext=(-2.55,-1.85),
   fontsize=8.6,arrowprops=dict(arrowstyle='->',lw=1.5,color='k'),fontweight='bold')

# zoom panel: the collision
ax=axes[2]
for Om,c,lab,mk in [(0.620,'#888','$\\Omega=0.620$','o'),(0.628,'#c77','$\\Omega=0.628$','o'),
                    (0.632,'#b44','$\\Omega=0.632$','o'),(OMS,'#000','$\\Omega^*$ (merged)','*')]:
    pts=[p for p in equilibria(Om) if -2.0<p[0]<-1.5 and -1.1<p[1]<-0.5]
    for i,p in enumerate(pts):
        ax.scatter(*p,s=230 if mk=='*' else 80,c=c,marker=mk,edgecolors='k',
                   zorder=5,linewidths=1.2,label=lab if i==0 else None)
ax.plot([-1.7408,-1.72481],[-0.9055,-0.80446],'--',c='#1f6fb4',lw=1.2,alpha=0.65)
ax.plot([-1.7042,-1.72481],[-0.7000,-0.80446],'--',c='#d1495b',lw=1.2,alpha=0.65)
ax.set_title('THE COLLISION (zoom)\nsaddle + sink $\\to$ single degenerate point',fontsize=11.5,fontweight='bold')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.grid(alpha=0.18); ax.legend(fontsize=8.3,loc='upper left')

from matplotlib.lines import Line2D
handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=v,markeredgecolor='k',markersize=8,label=k)
         for k,v in COL.items() if k!='degenerate']
handles.append(Line2D([0],[0],marker='*',color='w',markerfacecolor='k',markeredgecolor='k',markersize=15,label='degenerate $x^*$'))
fig.legend(handles=handles,loc='lower center',ncol=6,fontsize=9,frameon=False,bbox_to_anchor=(0.5,-0.035))
plt.suptitle('Saddle–Node Bifurcation: Construction of a Genuine Non-Hyperbolic Singularity',
             fontsize=13.5,fontweight='bold',y=1.0)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/saddle_node_bifurcation.png',dpi=185,bbox_inches='tight',facecolor='white')
print("saved")

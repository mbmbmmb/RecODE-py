"""Figure S1 of the Supplementary Material: fitted functional parameters against the truth
at several sieve sizes (median over replications, n = 2000, quantile knots).

Reads bic_model_selection/results/knot_single_curves.npz (Cox-type, settings 9-10),
knot_single_noanchor_curves.npz (AFT-type, settings 11-12) and knot_s4_fixed_curves.npz
(general model, setting 4), produced by knot_single.py and knot_setting4.py.

Shows that over p = 0.10-0.20 the fitted curves are nearly indistinguishable, and that
the visible failure at large p is OVERFITTING (oscillation), not bias.
"""
import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'bic_model_selection')
S=np.load(os.path.join(R,'results','knot_single_curves.npz'),allow_pickle=True)
SA=np.load(os.path.join(R,'results','knot_single_noanchor_curves.npz'),allow_pickle=True)
F=np.load(os.path.join(R,'results','knot_s4_fixed_curves.npz'),allow_pickle=True)
SHOW=[0.05,0.1,0.14,0.2,0.35]
def key(p): return ('%g'%p)
cols=plt.cm.viridis(np.linspace(0.05,0.85,len(SHOW)))

PAN=[('s10',r'(a) ODE-Cox, simple:  $\alpha(t)=1+t$',r'$t$',r'$\log\hat\alpha(t)$'),
     ('s9', r'(b) ODE-Cox, complex:  $\alpha(t)=1+\sqrt{t}+t^2$',r'$t$',r'$\log\hat\alpha(t)$'),
     ('s12',r'(c) ODE-AM, simple:  $q(u)=1/(1+u)$',r'$u$',r'$\log\hat q(u)$'),
     ('s11',r'(d) ODE-AM, complex:  $q(u)=1/(1+\sqrt{u})$',r'$u$',r'$\log\hat q(u)$')]
fig,ax=plt.subplots(3,2,figsize=(13.4,13.2))
A=ax.ravel()
for i,(tag,ttl,xl,yl) in enumerate(PAN):
    D=SA if tag in ('s11','s12') else S          # AFT-type fits are not anchored
    a=A[i]; g=D[f'{tag}_grid']
    for c,p in zip(cols,SHOW):
        k=f'{tag}_p{key(p)}'
        if k in D.files: a.plot(g,D[k],color=c,lw=1.7,label=f'$p$={p:g}')
    a.plot(g,D[f'{tag}_true'],'k--',lw=2.6,label='truth')
    a.set_title(ttl,fontsize=11); a.set_xlabel(xl); a.set_ylabel(yl)
    a.grid(alpha=.25); a.legend(fontsize=8.2)
# setting 4: both functional parameters free
ps=list(np.round(F['ps'],4))
for j,(arr,gr,tr,ttl,xl,yl) in enumerate([
    (F['med_alpha'],F['gt'],F['true_la'],r'(e) ODE-Flex:  $\alpha(t)=t+1$',r'$t$',r'$\log\hat\alpha(t)$'),
    (F['med_q'],   F['gu'],F['true_lq'],r'(f) ODE-Flex:  $q(u)=2/(1+u)$',r'$u$',r'$\log\hat q(u)$')]):
    a=A[4+j]
    for c,p in zip(cols,SHOW):
        if p in ps and p >= 0.10:                 # p = 0.05 is excluded, as in Table S1
            a.plot(gr,arr[ps.index(p)],color=c,lw=1.7,label=f'$p$={p:g}')
    a.plot(gr,tr,'k--',lw=2.6,label='truth')
    a.set_title(ttl,fontsize=11); a.set_xlabel(xl); a.set_ylabel(yl)
    a.grid(alpha=.25); a.legend(fontsize=8.2)
fig.suptitle('Fitted functional parameters against the truth, by sieve size\n'
             r'(median of 20-30 replications, $n=2000$, quantile knots; $\#$knots $=\lceil m^p\rceil$)',
             fontsize=13,y=0.995)
fig.tight_layout(rect=[0,0,1,0.975])
out=os.path.join(R,'knot_curves_all_settings.png')
fig.savefig(out,dpi=150,bbox_inches='tight'); print('wrote',out)
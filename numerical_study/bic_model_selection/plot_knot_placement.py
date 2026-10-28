"""Figure S2 of the Supplementary Material: knot placement against the number of knots
(q(u) = 1 + 0.7 sin(1.5u), n = 1000, 40 replications), with the BIC selection marked.

Reads results/place_p*.csv (produced by ../knot_placement.py) and writes
knot_placement.png next to this script. With the argument "annotated", writes the
annotated version used in the response letter to knot_placement_vs_count.png.
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
N = 1000
SUPP = not (len(sys.argv) > 1 and sys.argv[1] == "annotated")

d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "results", "place_p*.csv"))],
              ignore_index=True)
d = d[d.kind.isin(["equal", "quantile"]) & d.ok].reset_index(drop=True)
d["npar"] = d.lq + 2
d["BIC"] = 2 * N * d.negll + np.log(N) * d.npar

# BIC selection over all ten cells, separately on each dataset
sel = d.loc[d.groupby("seed").BIC.idxmin()]
counts = sel.groupby(["kind", "lq"]).size()
nds = d.seed.nunique()

fig, ax = plt.subplots(figsize=(7.3, 5.4), dpi=150)
if SUPP:
    style = {"equal": ("#c0392b", r"equally spaced on $[0,\,2\max\hat\mu]$"),
             "quantile": ("#2e86c1", r"sample quantiles of $\hat\mu$")}
else:
    style = {"equal": ("#c0392b", r"equal on $[0,\,2\max\hat\mu]$  (K1/K3)"),
             "quantile": ("#2e86c1", r"quantile of $\hat\mu$  (K2/K4)")}
for kind, (col, lab) in style.items():
    s = d[d.kind == kind].groupby("lq").rmse_q
    med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
    ax.plot(med.index, med.values, "-o", color=col, lw=2.2, ms=6, label=lab)
    ax.fill_between(med.index, q1.values, q3.values, color=col, alpha=0.13, lw=0)

qmed = d[d.kind == "quantile"].groupby("lq").rmse_q.median()
parts = []
for (kind, lq), c in counts.sort_values(ascending=False).items():
    y = d[(d.kind == kind) & (d.lq == lq)].rmse_q.median()
    ax.scatter([lq], [y], s=220 * np.sqrt(c / nds), facecolors="none",
               edgecolors="#d62728", linewidths=2.2, zorder=5)
    parts.append(f"{c}/{nds} at {lq} knots")
ax.scatter([], [], s=160, facecolors="none", edgecolors="#d62728", linewidths=2.2,
           label="selected by BIC (" + ", ".join(parts) + ")")

top = counts.idxmax()
if not SUPP:
  ax.annotate("BIC picks this cell", xy=(top[1], qmed[top[1]]), xytext=(9.3, 0.70),
            color="#d62728", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4,
                            connectionstyle="arc3,rad=-0.25", shrinkB=9))

ax.set_xscale("log")
ax.set_xticks([3, 4, 6, 9, 14])
ax.set_xticklabels(["3", "4", "6", "9", "14"])
ax.set_xlim(2.6, 17)
ax.set_xlabel("number of knots" if SUPP else "number of knots (identical on both arms)")
ax.set_ylabel(r"median RMSE of $\log\hat q$")
if not SUPP:
    ax.set_title("Knot placement vs knot count, same sieve size\n"
                 r"Setting 7  $q=1+0.7\sin(1.5u)$,  $n=1000$, " + f"{nds} reps")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
out = os.path.join(HERE, "knot_placement.png" if SUPP else "knot_placement_vs_count.png")
fig.savefig(out)
print("wrote", out, dict(counts))

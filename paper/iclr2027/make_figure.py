import json, os, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Figure is placed at width=0.63\linewidth (~3.47in on the ICLR 5.5in text width), so the
# native figure width is 3.45in and all fonts are printed at ~1:1 (>= 6.3pt).
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 6.8, 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'legend.fontsize': 6.3, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2})
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "curc_validation_results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "order_preservation_depth.pdf")
seeds = [42, 1337, 2024, 8888, 12345]
op = {s: json.load(open(f"{R}/t5-small/seed_{s}/order_preservation_results.json"))["per_layer_order_preservation"] for s in seeds}
po = json.load(open(f"{R}/pythia-1.4b/seed_42/order_preservation_results.json"))["per_layer_order_preservation"]
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.62), sharey=True)
col = {"baseline": "#1f77b4", "monotonic": "#d62728"}
ax = axes[0]
for key, ls in [("baseline_t5", "-"), ("monotonic_t5", "-"), ("baseline_t5_last_token", "--"), ("monotonic_t5_last_token", "--")]:
    layers = sorted(op[42][key].keys(), key=int); x = [int(L) for L in layers]
    m = [st.mean([op[s][key][L]["mean"] for s in seeds]) for L in layers]
    sd = [st.stdev([op[s][key][L]["mean"] for s in seeds]) for L in layers]
    c = col["baseline" if key.startswith("baseline") else "monotonic"]
    ax.plot(x, m, ls, color=c, marker="o", ms=2.2, lw=1.2)
    ax.fill_between(x, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)], color=c, alpha=0.15, lw=0)
ax.set_title("T5-small encoder (5 seeds)", pad=3); ax.set_xlabel("Layer (0 = embeddings)")
ax.set_ylabel("Frac. coords. $Ah_{\\mathrm{weak}} \\leq Ah_{\\mathrm{strong}}$")
ax.set_ylim(0.3, 1.02); ax.set_xticks(range(0, 7))
ax = axes[1]
for key, ls in [("baseline_pythia", "--"), ("monotonic_pythia", "--"), ("baseline_pythia_mean_pool", "-"), ("monotonic_pythia_mean_pool", "-")]:
    layers = sorted(po[key].keys(), key=int); x = [int(L) for L in layers]
    m = [po[key][L]["mean"] for L in layers]; lo = [po[key][L]["ci_low"] for L in layers]; hi = [po[key][L]["ci_high"] for L in layers]
    c = col["baseline" if key.startswith("baseline") else "monotonic"]
    ax.plot(x, m, ls, color=c, marker="o", ms=1.6, lw=1.2)
    ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
ax.set_title("Pythia-1.4B (seed 42, 95% CI)", pad=3); ax.set_xlabel("Layer (0 = embeddings)")
ax.set_ylim(0.3, 1.02)
for a in axes:
    a.grid(True, lw=0.3, alpha=0.4); a.tick_params(pad=1.5)
handles = [Line2D([0], [0], color=col["baseline"], ls="-", lw=1.2, label="Baseline, mean pool"),
           Line2D([0], [0], color=col["monotonic"], ls="-", lw=1.2, label="Monotone, mean pool"),
           Line2D([0], [0], color=col["baseline"], ls="--", lw=1.2, label="Baseline, last token"),
           Line2D([0], [0], color=col["monotonic"], ls="--", lw=1.2, label="Monotone, last token")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, columnspacing=1.2, handlelength=1.8, handletextpad=0.4, bbox_to_anchor=(0.54, 1.01), labelspacing=0.15)
plt.tight_layout(rect=(0, 0, 1, 0.85), w_pad=0.5, h_pad=0.2)
plt.savefig(OUT); print("saved", OUT)

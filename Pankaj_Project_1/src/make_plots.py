"""makes the plots for the report, reads results/*.json and saves results/figures/*.png"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_DIR, BLIMP_PARADIGMS, load_json  # noqa: E402

FIG = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG, exist_ok=True)
# colours for the 3 series (blue, orange, green) and for text/grid
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", os.path.relpath(os.path.join(FIG, name)))   # short path, relative to where it runs


def task1():
    r = load_json("task1_results.json")
    # test ppl of each 4-gram method, log scale because add-k is ~100x worse
    names = list(r["main_table_4gram"])
    vals = [r["main_table_4gram"][n]["test"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.barh(names, vals, color=S1, height=0.55)
    ax.set_xscale("log")
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v * 1.08, i, f"{v:,.1f}", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Test perplexity (log scale, lower is better)")
    ax.set_title("Task 1: 4-gram smoothing variants on WikiText-2", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=max(vals) * 4)
    save(fig, "task1_smoothing.png")

    # kneser-ney: validation ppl vs lambda for each order
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for n, c in zip(("2", "3", "4"), (S1, S2, S3)):
        d = r["kn_lambda_tuning"][n]
        lams = sorted(float(k) for k in d if float(k) < 1.0)
        ax.plot(lams, [d[str(l) if str(l) in d else f"{l:g}"] for l in lams], color=c, marker="o",
                markersize=5, label=f"{n}-gram")
        if n == "4":
            best = float(r["kn_best_lambda"][n])
            best_txt = f"best 4-gram: λ={best:g}, val PPL {d[str(best)]:.1f}"
    ax.set_xlabel("λ (weight on the discounted higher-order estimate)")
    ax.set_ylabel("Validation perplexity")
    ax.set_title(f"Kneser–Ney λ grid search ({best_txt})", loc="left", color=INK, fontsize=9.5)
    ax.legend()
    save(fig, "task1_kn_lambda.png")


def task2():
    # training and validation perplexity per epoch
    r = load_json("task2_training_log.json")
    ep = [e["epoch"] for e in r["log"]]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(ep, [e["train_ppl"] for e in r["log"]], color=S1, label="train (with dropout)")
    ax.plot(ep, [e["val_ppl"] for e in r["log"]], color=S2, label="validation")
    ax.set_yscale("log")
    if r.get("asgd_epoch"):
        ax.axvline(r["asgd_epoch"], color=INK2, linewidth=1, linestyle="--")
        ax.text(r["asgd_epoch"] + 0.5, ax.get_ylim()[1] * 0.8, "NT-ASGD\ntriggered", fontsize=8, color=INK2)
    last = r["log"][-1]
    ax.annotate(f"{last['val_ppl']:.1f}", (last["epoch"], last["val_ppl"]), textcoords="offset points",
                xytext=(4, -10), fontsize=8, color=INK2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_title("Task 2: AWD-LSTM training on WikiText-2", loc="left", color=INK)
    ax.legend()
    save(fig, "task2_training.png")


def task3():
    # hybrid: validation ppl and blimp accuracy vs lambda (two separate plots)
    r = load_json("task3_results.json")
    lams = sorted(float(k) for k in r["lambda_grid_validation_ppl"])
    get = lambda d, l: d[str(l)] if str(l) in d else d[f"{l:g}"]  # noqa: E731
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.2))
    a1.plot(lams, [get(r["lambda_grid_validation_ppl"], l) for l in lams], color=S1, marker="o", markersize=5)
    b = r["best_lambda"]
    a1.annotate(f"best λ={b:g}\n{r['lambda_grid_validation_ppl'][str(b)]:.1f}",
                (b, r["lambda_grid_validation_ppl"][str(b)]), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8, color=INK2)
    a1.set_xlabel("λ (weight on KN;  0 = AWD-LSTM only, 1 = KN only)")
    a1.set_ylabel("Validation perplexity")
    a1.set_title("Hybrid: validation perplexity", loc="left", color=INK)
    a2.plot(lams, [get(r["blimp_overall_by_lambda"], l) * 100 for l in lams], color=S1, marker="o", markersize=5)
    a2.set_xlabel("λ (weight on KN)")
    a2.set_ylabel("BLiMP accuracy, %")
    a2.set_title("Hybrid: BLiMP accuracy", loc="left", color=INK)
    save(fig, "task3_lambda.png")

    # blimp accuracy per paradigm for the 3 models
    t1, t2 = load_json("task1_results.json"), load_json("task2_results.json")
    kn = t1["blimp"][[k for k in t1["blimp"] if "Kneser" in k][0]]["per_paradigm"]
    series = [("T1: KN 4-gram", kn, S1), ("T2: AWD-LSTM", t2["blimp"]["per_paradigm"], S2),
              (f"T3: Hybrid λ={b:g}", r["blimp"]["per_paradigm"], S3)]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    w = 0.26
    for j, (name, d, c) in enumerate(series):
        xs = [i + (j - 1) * w for i in range(len(BLIMP_PARADIGMS))]
        ax.bar(xs, [d[p]["acc"] * 100 for p in BLIMP_PARADIGMS], width=w - 0.03, color=c, label=name)
    ax.axhline(50, color=INK2, linewidth=1, linestyle=":")
    ax.text(-0.45, 51.5, "chance", fontsize=8, color=INK2, ha="left")
    ax.set_xticks(range(len(BLIMP_PARADIGMS)))
    ax.set_xticklabels([p.replace("_", " ") for p in BLIMP_PARADIGMS], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy, %")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.set_title("BLiMP accuracy per paradigm", loc="left", color=INK)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
    save(fig, "blimp_per_paradigm.png")


if __name__ == "__main__":
    for f, need in ((task1, "task1_results.json"), (task2, "task2_training_log.json"),
                    (task3, "task3_results.json")):
        if os.path.exists(os.path.join(RESULTS_DIR, need)):
            f()

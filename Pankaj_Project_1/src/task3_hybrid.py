"""task 3 - hybrid of kneser-ney and awd-lstm

    P_hybrid(w | context) = lam * P_KN(w | context) + (1 - lam) * P_AWD(w | context)

no new training here. we load the per-word probabilities saved by task 1 (4-gram
kneser-ney) and task 2 (best awd-lstm checkpoint). both models scored exactly the
same words with the same vocab, so we can mix them word by word. since both are
proper probability distributions, the mix is one too.

lam is picked by grid search over 0.0, 0.1, ..., 1.0 on validation only.

run:  python src/task3_hybrid.py     (after task1_ngram.py and task2_awd_lstm.py eval)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE_DIR, RESULTS_DIR, BLIMP_PARADIGMS, blimp_accuracy,  # noqa: E402
                    format_blimp_table, get_blimp, get_data, load_json, ppl_from_logprobs, save_json)

LAMBDAS = [round(0.1 * i, 1) for i in range(11)]


def mix(lp_kn, lp_awd, lam):
    """log of the mixed probability: log(lam * exp(a) + (1-lam) * exp(b))"""
    if lam == 0.0:
        return lp_awd
    if lam == 1.0:
        return lp_kn
    return np.logaddexp(np.log(lam) + lp_kn, np.log(1 - lam) + lp_awd)


def load(name):
    return np.load(os.path.join(CACHE_DIR, name), allow_pickle=True)


def bucket_analysis(ids, lp, vocab):
    """where does each model do better? test perplexity split by
    (a) whether the 4-gram (3 context words + target) was seen in train
    (b) how often the target word appears in train"""
    from task1_ngram import NGramCounts, stream_keys
    C = NGramCounts(ids["train"], len(vocab))
    test = ids["test"]
    _, seen4 = C.raw[4][stream_keys(test, 4, C.B, C.PAD)]
    _, seen2 = C.raw[2][stream_keys(test, 2, C.B, C.PAD)]
    freq = np.bincount(ids["train"], minlength=len(vocab))[test[1:]]
    tgt = test[1:]
    buckets = {
        "4-gram seen in train": seen4,
        "4-gram unseen, bigram seen": (~seen4) & seen2,
        "bigram unseen": ~seen2,
        "target freq > 10k": freq > 10000,
        "target freq 1k-10k": (freq > 1000) & (freq <= 10000),
        "target freq 100-1k": (freq > 100) & (freq <= 1000),
        "target freq <= 100": (freq <= 100) & (tgt != vocab.unk),
        "target = <unk>": tgt == vocab.unk,
    }
    out = {}
    for name, mask in buckets.items():
        out[name] = {"share": float(mask.mean())}
        for model, arr in lp.items():
            out[name][model] = ppl_from_logprobs(arr[mask])
    return out


def main():
    # load the saved results and per-word log probs from task 1 and task 2
    vocab, ids = get_data()
    t1, t2 = load_json("task1_results.json"), load_json("task2_results.json")
    kn = {s: load(f"task1_kn_{s}_logprobs.npy") for s in ("validation", "test")}
    awd = {s: load(f"task2_{s}_logprobs.npy") for s in ("validation", "test")}
    for s in kn:
        assert kn[s].shape == awd[s].shape, (s, kn[s].shape, awd[s].shape)

    # tune lambda on validation
    tuning = {lam: ppl_from_logprobs(mix(kn["validation"], awd["validation"], lam)) for lam in LAMBDAS}
    best = min(LAMBDAS, key=lambda l: tuning[l])
    print("validation PPL by lambda:", {l: round(p, 2) for l, p in tuning.items()}, "-> best", best)
    test_ppl = {lam: ppl_from_logprobs(mix(kn["test"], awd["test"], lam)) for lam in LAMBDAS}  # only for the plot, not used to pick lambda

    hybrid = {s: mix(kn[s], awd[s], best) for s in ("validation", "test")}

    # blimp, mixing the per-word scores the same way
    blimp = get_blimp(vocab)
    kg, kb = load("task1_kn_blimp_good.npy"), load("task1_kn_blimp_bad.npy")
    ag, ab = load("task2_blimp_good.npy"), load("task2_blimp_bad.npy")
    hg = [mix(a, b, best).mean() for a, b in zip(kg, ag)]
    hb = [mix(a, b, best).mean() for a, b in zip(kb, ab)]
    blimp_h = blimp_accuracy(blimp, hg, hb)
    # blimp accuracy for every lambda (extra, for the plot)
    blimp_curve = {}
    for lam in LAMBDAS:
        g = [mix(a, b, lam).mean() for a, b in zip(kg, ag)]
        b_ = [mix(a, b, lam).mean() for a, b in zip(kb, ab)]
        blimp_curve[lam] = blimp_accuracy(blimp, g, b_)["overall"]

    kn_name = t1["best_variant"] if "Kneser" in t1["best_variant"] else \
        [k for k in t1["main_table_4gram"] if "Kneser" in k][0]
    kn_blimp = t1["blimp"][kn_name]

    buckets = bucket_analysis(ids, {"KN 4-gram": kn["test"], "AWD-LSTM": awd["test"],
                                    f"Hybrid (λ={best})": hybrid["test"]}, vocab)

    # how often does the hybrid give a higher prob than awd-lstm alone
    gain = hybrid["test"] - awd["test"]
    res = {
        "lambda_grid_validation_ppl": tuning, "lambda_grid_test_ppl": test_ppl, "best_lambda": best,
        "validation_ppl": ppl_from_logprobs(hybrid["validation"]),
        "test_ppl": ppl_from_logprobs(hybrid["test"]),
        "blimp": blimp_h, "blimp_overall_by_lambda": blimp_curve,
        "bucket_analysis_test": buckets,
        "frac_test_tokens_hybrid_better_than_awd": float((gain > 0).mean()),
        "comparison": {
            "Task 1: KN 4-gram": {"validation_ppl": t1["main_table_4gram"][kn_name]["validation"],
                                  "test_ppl": t1["main_table_4gram"][kn_name]["test"],
                                  "blimp": kn_blimp["overall"]},
            "Task 2: AWD-LSTM": {"validation_ppl": t2["validation_ppl"], "test_ppl": t2["test_ppl"],
                                 "blimp": t2["blimp"]["overall"]},
            f"Task 3: Hybrid (λ={best})": {"validation_ppl": ppl_from_logprobs(hybrid["validation"]),
                                           "test_ppl": ppl_from_logprobs(hybrid["test"]),
                                           "blimp": blimp_h["overall"]},
        },
    }
    save_json(res, "task3_results.json")

    # write the results tables, including the final task 1 vs 2 vs 3 comparison

    md = ["# Task 3 - Interpolated KN + AWD-LSTM hybrid", "",
          "## λ grid search (λ = weight of KN; tuned on validation only)", "",
          "| λ | " + " | ".join(f"{l:g}" for l in LAMBDAS) + " |",
          "|---|" + "---|" * len(LAMBDAS),
          "| Validation PPL | " + " | ".join(f"{tuning[l]:.2f}" for l in LAMBDAS) + " |",
          "| Test PPL (not used for selection) | " + " | ".join(f"{test_ppl[l]:.2f}" for l in LAMBDAS) + " |",
          "| BLiMP overall % | " + " | ".join(f"{blimp_curve[l]*100:.1f}" for l in LAMBDAS) + " |",
          "", f"Selected λ = **{best}**.", "",
          "## Final comparison (Task 1 vs Task 2 vs Task 3)", "",
          "| Model | Validation PPL | Test PPL | BLiMP overall % |", "|---|---|---|---|"]
    for name, r in res["comparison"].items():
        md.append(f"| {name} | {r['validation_ppl']:.2f} | {r['test_ppl']:.2f} | {r['blimp']*100:.1f} |")
    md += ["", "## BLiMP per paradigm (%)", "",
           format_blimp_table({"T1: KN 4-gram": kn_blimp, "T2: AWD-LSTM": t2["blimp"],
                               f"T3: Hybrid λ={best}": blimp_h}),
           "", "## Where does the n-gram help? Test PPL by token bucket", "",
           "| Bucket | Share of test tokens | " + " | ".join(k for k in list(buckets.values())[0] if k != "share") + " |",
           "|---|---|" + "---|" * 3]
    for name, r in buckets.items():
        md.append(f"| {name} | {r['share']*100:.1f}% | " +
                  " | ".join(f"{v:.1f}" for k, v in r.items() if k != "share") + " |")
    md += ["", f"The hybrid assigns higher probability than the AWD-LSTM alone to "
           f"{res['frac_test_tokens_hybrid_better_than_awd']*100:.1f}% of test tokens."]
    with open(os.path.join(RESULTS_DIR, "task3_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""task 1 - n-gram language model with smoothing

builds 1 to 4-gram counts from the train split and tries 5 smoothing methods:
  - add-k with k = 1, 0.5, 0.1
  - good-turing (katz backoff, counts r <= 5 are discounted)
  - interpolated kneser-ney with d = 0.75, lambda tuned on validation

everything is done with numpy arrays so it runs fast. each n-gram (w1..wn)
is turned into one int64 number  w1*B^(n-1) + ... + wn  where B = vocab size + 1
(the extra id is a padding token we only use as left context at the start).
looking up a count is then just a binary search in a sorted array.

run:  python src/task1_ngram.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BLIMP_CONTEXT, CACHE_DIR, RESULTS_DIR, blimp_accuracy,  # noqa: E402
                    format_blimp_table, get_blimp, get_data, ppl_from_logprobs, save_json)

MAX_N = 4
KN_DISCOUNT = 0.75
KN_LAMBDAS = [round(0.1 * i, 1) for i in range(1, 10)]
ADD_KS = [1.0, 0.5, 0.1]
GT_MAX_R = 5
GT_MIN_BACKOFF = 1e-3   # min backoff mass for katz, explained in KatzGoodTuring


# ---------------- count tables ----------------
class Table:
    """sorted keys -> values, lookup works on whole arrays at once"""

    def __init__(self, keys, vals):
        self.keys, self.vals = keys, vals

    def __getitem__(self, q):
        # binary search, keys that are not in the table get value 0
        idx = np.searchsorted(self.keys, q)
        idx[idx == len(self.keys)] = 0
        found = self.keys[idx] == q
        return np.where(found, self.vals[idx], 0), found


def group_by_history(keys, vals, B):
    """for each history h: total count c(h) and number of different next words"""
    prefix = keys // B                          # history part of the key, already sorted
    starts = np.flatnonzero(np.r_[True, prefix[1:] != prefix[:-1]])
    hk = prefix[starts]
    hsum = np.add.reduceat(vals, starts)
    htypes = np.diff(np.r_[starts, len(keys)])
    return Table(hk, hsum.astype(np.float64)), Table(hk, htypes.astype(np.float64)), starts


class NGramCounts:
    def __init__(self, ids, V, max_n=MAX_N):
        self.V, self.B, self.PAD, self.N, self.max_n = V, V + 1, V, len(ids), max_n
        # raw counts for every order, by sliding a window over the token stream
        self.raw = {}
        for n in range(1, max_n + 1):
            k = np.zeros(len(ids) - n + 1, dtype=np.int64)
            for j in range(n):
                k = k * self.B + ids[j: len(ids) - n + 1 + j]
            u, c = np.unique(k, return_counts=True)
            self.raw[n] = Table(u, c.astype(np.float64))
        # continuation counts for kneser-ney: in how many different contexts does g appear
        self.cont = {}
        for m in range(1, max_n):
            suffix = self.raw[m + 1].keys % (self.B ** m)
            u, c = np.unique(suffix, return_counts=True)
            self.cont[m] = Table(u, c.astype(np.float64))

    def describe(self):
        return {n: len(t.keys) for n, t in self.raw.items()}


# ---------------- smoothing methods ----------------
class AddK:
    def __init__(self, C, n, k):
        self.C, self.n, self.k = C, n, k
        self.name = f"Add-k (k={k:g})"
        if n > 1:
            self.hsum, _, _ = group_by_history(C.raw[n].keys, C.raw[n].vals, C.B)

    def prob(self, g):
        # P(w|h) = (c(h,w) + k) / (c(h) + k*V)
        C, k, V = self.C, self.k, self.C.V
        c, _ = C.raw[self.n][g]
        if self.n == 1:
            return (c + k) / (C.N + k * V)
        ch, _ = self.hsum[g // C.B]
        return (c + k) / (ch + k * V)


class KatzGoodTuring:
    """katz backoff using good-turing discounted counts for r <= 5

    r* = (r+1) N_{r+1} / N_r, and katz's discount ratio is
    d_r = (r*/r - (k+1)N_{k+1}/N_1) / (1 - (k+1)N_{k+1}/N_1) with k = 5
    seen n-gram:   P(w|h) = d_c * c(h,w) / c(h)
    unseen n-gram: P(w|h) = alpha(h) * P_katz(w|h'), alpha makes it sum to 1
    unigrams are just c(w)/N since every vocab word appears in train.

    problem we found: counts above 5 are not discounted, so if every word seen
    after some history appeared more than 5 times, there is no mass left for
    unseen words and they get probability 0 (perplexity becomes infinite).
    to fix this, such histories keep at least min_backoff mass and the seen
    probabilities are scaled down so it still sums to 1.
    """

    def __init__(self, C, n, min_backoff=GT_MIN_BACKOFF):
        self.C, self.n, self.min_backoff = C, n, min_backoff
        self.name = "Good-Turing (Katz)" if min_backoff > 0 else "Good-Turing (Katz, no floor)"
        self.disc, self.hsum, self.alpha, self.nr, self.scale = {}, {}, {}, {}, {}
        # go from low order to high order, since alpha for order m needs P for order m-1
        for m in range(2, n + 1):
            t = C.raw[m]
            # N_r = how many n-grams occur exactly r times
            r_counts = np.bincount(t.vals.astype(np.int64), minlength=GT_MAX_R + 3)
            Nr = r_counts.astype(np.float64)
            kk = GT_MAX_R
            common = (kk + 1) * Nr[kk + 1] / Nr[1]
            # discount ratio d_r for r = 1..5
            d = np.ones(GT_MAX_R + 1)
            for r in range(1, kk + 1):
                rstar = (r + 1) * Nr[r + 1] / Nr[r]
                d[r] = (rstar / r - common) / (1 - common)
            d = np.where((d > 0) & (d <= 1), d, 1.0)
            self.disc[m] = d
            self.nr[m] = {"N_r": Nr[1:kk + 2].tolist(), "d_r": d[1:].tolist()}
            hsum, _, starts = group_by_history(t.keys, t.vals, C.B)
            self.hsum[m] = hsum
            # leftover mass beta(h) = 1 - sum of discounted probs of seen words
            pdisc = self._dstar(m, t.vals) / hsum.vals.repeat(np.diff(np.r_[starts, len(t.keys)]))
            lower = self.prob(t.keys % (C.B ** (m - 1)), m - 1)
            beta = np.clip(1.0 - np.add.reduceat(pdisc, starts), 0, None)
            new_beta = np.maximum(beta, self.min_backoff)
            self.scale[m] = Table(hsum.keys, (1 - new_beta) / (1 - beta))   # scale down seen probs
            # alpha(h) = beta(h) / (1 - lower-order prob of the seen words)
            den = 1.0 - np.add.reduceat(lower, starts)
            alpha = new_beta / np.clip(den, 1e-12, None)
            self.alpha[m] = Table(hsum.keys, alpha)
            self.nr[m]["histories_floored"] = int((beta < self.min_backoff).sum())

    def _dstar(self, m, c):
        # discounted count: d_r * r for r <= 5, raw count otherwise
        ci = c.astype(np.int64)
        d = np.where(ci <= GT_MAX_R, self.disc[m][np.minimum(ci, GT_MAX_R)], 1.0)
        return d * c

    def prob(self, g, m=None):
        m = self.n if m is None else m
        C = self.C
        if m == 1:
            c, _ = C.raw[1][g]
            return c / C.N
        c, seen = C.raw[m][g]
        h = g // C.B
        lower = self.prob(g % (C.B ** (m - 1)), m - 1)
        ch, _ = self.hsum[m][h]
        a, hseen = self.alpha[m][h]
        a = np.where(hseen, a, 1.0)                 # history never seen -> just use lower order
        with np.errstate(divide="ignore", invalid="ignore"):
            p_seen = self._dstar(m, c) / ch * self.scale[m][h][0]
        return np.where(seen, p_seen, a * lower)


class InterpolatedKN:
    """interpolated kneser-ney with discount d and an extra weight lambda

    for order m >= 2 (raw counts at the top order, continuation counts below):
      P_KN(w|h) = max(c(h,w) - d, 0)/c(h) + d * N1+(h .)/c(h) * P_lower(w|h')
      P(w|h)    = lambda * P_KN(w|h) + (1 - lambda) * P_lower(w|h')
    lambda = 1 is normal interpolated kneser-ney. if h was never seen we just
    use P_lower. the unigram level uses continuation counts mixed with uniform.
    """

    def __init__(self, C, n, lam, d=KN_DISCOUNT):
        self.C, self.n, self.lam, self.d = C, n, lam, d
        self.name = f"Kneser-Ney (d={d}, λ={lam:g})"
        self.tab, self.hsum, self.htypes = {}, {}, {}
        # top order uses raw counts, lower orders use continuation counts
        for m in range(1, n + 1):
            t = C.raw[m] if m == n else C.cont[m]
            self.tab[m] = t
            if m > 1:
                self.hsum[m], self.htypes[m], _ = group_by_history(t.keys, t.vals, C.B)
        t1 = self.tab[1]
        self.s1, self.ty1 = t1.vals.sum(), len(t1.keys)

    def prob(self, g, m=None):
        m = self.n if m is None else m
        C, d = self.C, self.d
        if m == 1:
            # discounted continuation unigram + leftover mass spread uniformly
            c, _ = self.tab[1][g]
            return np.maximum(c - d, 0) / self.s1 + d * self.ty1 / self.s1 / C.V
        c, _ = self.tab[m][g]
        h = g // C.B
        S, hseen = self.hsum[m][h]
        T, _ = self.htypes[m][h]
        lower = self.prob(g % (C.B ** (m - 1)), m - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pkn = np.maximum(c - d, 0) / S + d * T / S * lower
        return np.where(hseen, self.lam * pkn + (1 - self.lam) * lower, lower)


# ---------------- helpers for scoring text ----------------
def stream_keys(ids, n, B, PAD):
    """n-gram keys for predicting ids[1:], with up to n-1 words of left context"""
    padded = np.concatenate([np.full(n - 1, PAD, dtype=np.int64), ids])
    M = len(ids) - 1
    k = np.zeros(M, dtype=np.int64)
    for j in range(n):          # the word we predict, ids[t], is at padded[t + n - 1]
        k = k * B + padded[1 + j: 1 + j + M]
    return k


def sentence_keys(sentences, ctx, n, B, PAD):
    """keys for every word of every sentence, each sentence starts after ctx"""
    keys, lengths = [], []
    for s in sentences:
        seq = np.concatenate([ctx, s])
        k = stream_keys(seq, n, B, PAD)[len(ctx) - 1:]     # keep only the sentence words
        keys.append(k)
        lengths.append(len(s))
    return np.concatenate(keys), np.array(lengths)


def logp(model, keys):
    return np.log(np.clip(model.prob(keys), 1e-300, None))


def split_by_lengths(arr, lengths):
    return np.split(arr, np.cumsum(lengths)[:-1])


# ---------------- main ----------------
def main():
    t0 = time.time()
    vocab, ids = get_data()
    V = len(vocab)
    C = NGramCounts(ids["train"], V)
    print(f"vocab {V} | train tokens {C.N} | unique n-grams {C.describe()} | {time.time()-t0:.1f}s", flush=True)

    def eval_keys(split, n):
        return stream_keys(ids[split], n, C.B, C.PAD)

    keys = {(s, n): eval_keys(s, n) for s in ("validation", "test") for n in range(1, MAX_N + 1)}

    # sanity check: probabilities over the whole vocab should sum to 1
    rng = np.random.default_rng(0)
    probe_ctx = ids["validation"][rng.integers(3, 10000, size=3)]

    def check_norm(model, n):
        sums = []
        for c in probe_ctx:
            ctx = np.array([c] * (n - 1), dtype=np.int64)
            base = 0
            for x in ctx:
                base = base * C.B + x
            sums.append(model.prob(base * C.B + np.arange(V, dtype=np.int64)).sum())
        return sums

    # kneser-ney: grid search lambda on validation, separately for each order
    kn_tuning = {}
    best_lam = {}
    for n in range(2, MAX_N + 1):
        kn_tuning[n] = {}
        for lam in KN_LAMBDAS + [1.0]:
            kn_tuning[n][lam] = ppl_from_logprobs(logp(InterpolatedKN(C, n, lam), keys[("validation", n)]))
        best_lam[n] = min(KN_LAMBDAS, key=lambda l: kn_tuning[n][l])     # 1.0 is only for reference, not picked
        print(f"KN n={n}: val PPL by lambda " +
              ", ".join(f"{l:g}:{p:.1f}" for l, p in kn_tuning[n].items()) + f" -> best {best_lam[n]}", flush=True)

    # evaluate every smoothing method for n = 1..4
    def make(variant, n):
        if variant.startswith("addk"):
            return AddK(C, n, float(variant[4:]))
        if variant == "gt":
            return KatzGoodTuring(C, n)
        if variant == "kn":
            return InterpolatedKN(C, n, best_lam.get(n, 1.0)) if n > 1 else InterpolatedKN(C, 1, 1.0)
        raise ValueError(variant)

    variants = [f"addk{k:g}" for k in ADD_KS] + ["gt", "kn"]
    table = {}
    models = {}
    for v in variants:
        for n in range(1, MAX_N + 1):
            m = make(v, n)
            models[(v, n)] = m
            table[(v, n)] = {s: ppl_from_logprobs(logp(m, keys[(s, n)])) for s in ("validation", "test")}
            print(f"{m.name:28s} n={n}: val {table[(v, n)]['validation']:10.2f} | "
                  f"test {table[(v, n)]['test']:10.2f}", flush=True)

    # count how many tokens get probability 0 with plain katz (no min backoff)
    gt_raw = {}
    for n in range(2, MAX_N + 1):
        m = KatzGoodTuring(C, n, min_backoff=0.0)
        p = m.prob(keys[("validation", n)]), m.prob(keys[("test", n)])
        gt_raw[n] = {"val_zero_prob_tokens": int((p[0] <= 0).sum()), "test_zero_prob_tokens": int((p[1] <= 0).sum())}
        print(f"Katz without floor n={n}: zero-probability tokens val/test = "
              f"{gt_raw[n]['val_zero_prob_tokens']}/{gt_raw[n]['test_zero_prob_tokens']} -> PPL = inf", flush=True)

    norm = {models[(v, MAX_N)].name: check_norm(models[(v, MAX_N)], MAX_N) for v in variants}
    print("normalisation check (sum_w P(w|h), should be 1):",
          {k: [round(x, 6) for x in s] for k, s in norm.items()}, flush=True)

    # pick the best 4-gram model by validation perplexity, use it for blimp
    best_v = min(variants, key=lambda v: table[(v, MAX_N)]["validation"])
    best = models[(best_v, MAX_N)]
    print(f"best variant by validation PPL: {best.name}", flush=True)

    blimp = get_blimp(vocab)
    ctx = vocab.encode(BLIMP_CONTEXT)
    gk, gl = sentence_keys([e["good"] for e in blimp], ctx, MAX_N, C.B, C.PAD)
    bk, bl = sentence_keys([e["bad"] for e in blimp], ctx, MAX_N, C.B, C.PAD)
    blimp_res = {}
    for v in variants:                     # only the best one is required, rest are extra
        m = models[(v, MAX_N)]
        g = split_by_lengths(logp(m, gk), gl)
        b = split_by_lengths(logp(m, bk), bl)
        blimp_res[m.name] = blimp_accuracy(blimp, [x.mean() for x in g], [x.mean() for x in b])
        if v == "kn":                      # save per-word scores, task 3 needs them
            np.save(os.path.join(CACHE_DIR, "task1_kn_blimp_good.npy"), np.array(g, dtype=object), allow_pickle=True)
            np.save(os.path.join(CACHE_DIR, "task1_kn_blimp_bad.npy"), np.array(b, dtype=object), allow_pickle=True)
    kn = models[("kn", MAX_N)]
    for s in ("validation", "test"):
        np.save(os.path.join(CACHE_DIR, f"task1_kn_{s}_logprobs.npy"), logp(kn, keys[(s, MAX_N)]))

    # save everything to json
    res = {
        "vocab_size": V, "train_tokens": int(C.N), "unique_ngrams": C.describe(),
        "eval_tokens": {s: int(len(ids[s]) - 1) for s in ("validation", "test")},
        "kn_lambda_tuning": {str(n): {str(l): p for l, p in d.items()} for n, d in kn_tuning.items()},
        "kn_best_lambda": best_lam,
        "good_turing_stats": {n: models[("gt", n)].nr[n] for n in range(2, MAX_N + 1)},
        "good_turing_min_backoff": GT_MIN_BACKOFF,
        "good_turing_without_floor": gt_raw,
        "perplexity": {f"{models[(v, n)].name} | n={n}": table[(v, n)] for v in variants for n in range(1, MAX_N + 1)},
        "main_table_4gram": {models[(v, MAX_N)].name: table[(v, MAX_N)] for v in variants},
        "best_variant": best.name,
        "blimp": blimp_res,
        "normalisation_check": norm,
        "runtime_s": time.time() - t0,
    }
    save_json(res, "task1_results.json")

    # and write the results tables as markdown
    md = ["# Task 1 - Smoothed n-gram LM results", "",
          f"Vocabulary {V} (30,000 + `<unk>`), {C.N:,} training tokens; "
          f"evaluated on {len(ids['test'])-1:,} test tokens.", "",
          "## Perplexity of the five smoothing variants (4-gram)", "",
          "| Variant | Validation PPL | Test PPL |", "|---|---|---|"]
    for v in variants:
        r = table[(v, MAX_N)]
        md.append(f"| {models[(v, MAX_N)].name} | {r['validation']:.2f} | {r['test']:.2f} |")
    md += ["", "## Test perplexity by n-gram order", "",
           "| Variant | " + " | ".join(f"n={n}" for n in range(1, MAX_N + 1)) + " |",
           "|---|" + "---|" * MAX_N]
    for v in variants:
        md.append(f"| {models[(v, MAX_N)].name.split(' (')[0] + (' ' + v[4:] if v.startswith('addk') else '')} | " +
                  " | ".join(f"{table[(v, n)]['test']:.1f}" for n in range(1, MAX_N + 1)) + " |")
    md += ["", f"Good-Turing note: pure Katz back-off (no floor) gives zero probability to "
           f"{gt_raw[MAX_N]['test_zero_prob_tokens']} test tokens at n={MAX_N} (perplexity = ∞), because histories "
           f"whose continuations all occur > {GT_MAX_R} times keep no back-off mass; the reported numbers use a "
           f"minimum back-off mass of {GT_MIN_BACKOFF:g} for such histories."]
    md += ["", "## Kneser-Ney λ grid search (validation PPL; λ=1.0 = textbook IKN, reference only)", "",
           "| n | " + " | ".join(f"λ={l:g}" for l in KN_LAMBDAS + [1.0]) + " | best λ |",
           "|---|" + "---|" * (len(KN_LAMBDAS) + 2)]
    for n in range(2, MAX_N + 1):
        md.append(f"| {n} | " + " | ".join(f"{kn_tuning[n][l]:.1f}" for l in KN_LAMBDAS + [1.0]) + f" | {best_lam[n]} |")
    md += ["", f"## BLiMP accuracy (%) - best variant by validation PPL: **{best.name}**", "",
           format_blimp_table({best.name: blimp_res[best.name]}), "",
           "### BLiMP accuracy (%) for every 4-gram variant (supplementary)", "",
           format_blimp_table(blimp_res)]
    with open(os.path.join(RESULTS_DIR, "task1_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

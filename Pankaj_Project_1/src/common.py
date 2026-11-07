"""shared code for all 3 tasks: data loading, tokenizer, vocab, blimp and metrics

everything that has to be exactly the same in task 1, 2 and 3 is kept here,
so the models are compared on the same tokens and the same blimp pairs.
"""
import json
import os
import re
from collections import Counter

import numpy as np

# folders, can be changed with environment variables
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(ROOT, "cache"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

VOCAB_SIZE = 30000      # keep the 30k most frequent train tokens (+ <unk>)
MIN_COUNT = 2           # words seen less than 2 times in train become <unk>
UNK = "<unk>"

# the 10 blimp paradigms we use: the 6 suggested in the datasets document + 4 more
# so that morphology, syntax and semantics are all covered. the field shown in
# the results tables comes from blimp itself.
BLIMP_PARADIGMS = [
    # morphology
    "anaphor_gender_agreement",
    "determiner_noun_agreement_1",
    "regular_plural_subject_verb_agreement_1",
    "irregular_past_participle_adjectives",
    # syntax
    "wh_island",
    "adjunct_island",
    "drop_argument",
    # semantics / syntax-semantics
    "npi_present_1",
    "principle_A_domain_1",
    "existential_there_quantifiers_1",
]

# every blimp sentence is scored after this context. wikitext is one long stream
# with no <s> token, so a sentence normally starts right after a full stop.
BLIMP_CONTEXT = ["."]

# ---------------- tokenizer ----------------
# splits words, clitics ('s, n't, 're ...) and every punctuation mark into separate tokens
_TOKEN_RE = re.compile(r"\w+(?=n't)|n't|'\w+|\w+|[^\w\s]")


def normalize_wikitext(text):
    """undo wikitext's special formatting so it looks like normal english (same as blimp)"""
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r"n 't\b", "n't", text)      # "didn 't" -> "didn't" 
    return text


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


# ---------------- vocabulary ----------------
class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.unk = self.stoi[UNK]

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        s, u = self.stoi, self.unk
        return np.array([s.get(t, u) for t in tokens], dtype=np.int64)


def _load_wikitext_tokens():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    out = {}
    for split in ("train", "validation", "test"):
        # join all lines into one long stream, lines are not real sentences
        text = " ".join(ds[split]["text"])
        out[split] = tokenize(normalize_wikitext(text))
    return out


def get_data():
    """returns the vocab and the token ids of each split. cached after the first run"""
    vpath = os.path.join(CACHE_DIR, "vocab.json")
    paths = {s: os.path.join(CACHE_DIR, f"{s}.npy") for s in ("train", "validation", "test")}
    if os.path.exists(vpath) and all(os.path.exists(p) for p in paths.values()):
        with open(vpath) as f:
            vocab = Vocab(json.load(f))
        return vocab, {s: np.load(p) for s, p in paths.items()}

    # vocab is built from train only
    toks = _load_wikitext_tokens()
    counts = Counter(toks["train"])
    kept = [w for w, c in counts.most_common() if c >= MIN_COUNT and w != UNK][:VOCAB_SIZE]
    vocab = Vocab([UNK] + kept)
    ids = {s: vocab.encode(t) for s, t in toks.items()}
    with open(vpath, "w") as f:
        json.dump(vocab.itos, f)
    for s, p in paths.items():
        np.save(p, ids[s])
    return vocab, ids


# ---------------- blimp ----------------
def get_blimp(vocab):
    """loads the blimp pairs, tokenized with the same tokenizer and vocab as wikitext-2

    returns a list of dicts with paradigm, field, good ids and bad ids
    """
    from datasets import load_dataset
    data = []
    for p in BLIMP_PARADIGMS:
        ds = load_dataset("nyu-mll/blimp", p)["train"]
        for ex in ds:
            data.append({
                "paradigm": p,
                "field": ex["field"],
                "good": vocab.encode(tokenize(ex["sentence_good"])),
                "bad": vocab.encode(tokenize(ex["sentence_bad"])),
            })
    return data


def blimp_accuracy(data, good_scores, bad_scores):
    """accuracy = how often the good sentence gets the higher score, per paradigm and overall

    good_scores / bad_scores are length normalised log probs, one per pair
    """
    per = {}
    for ex, g, b in zip(data, good_scores, bad_scores):
        d = per.setdefault(ex["paradigm"], {"field": ex["field"], "correct": 0, "n": 0})
        d["correct"] += int(g > b)
        d["n"] += 1
    for d in per.values():
        d["acc"] = d["correct"] / d["n"]
    overall = sum(d["correct"] for d in per.values()) / sum(d["n"] for d in per.values())
    return {"overall": overall, "per_paradigm": per}


def ppl_from_logprobs(logp):
    """perplexity = exp(-average log prob)"""
    return float(np.exp(-np.mean(logp)))


def save_json(obj, name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def format_blimp_table(results_by_model):
    """makes a markdown table of blimp accuracy, one column per model"""
    names = list(results_by_model)
    first = results_by_model[names[0]]["per_paradigm"]
    lines = ["| Paradigm | Field | " + " | ".join(names) + " |",
             "|---|---|" + "---|" * len(names)]
    for p in BLIMP_PARADIGMS:
        row = [p, first[p]["field"]] + [f"{results_by_model[n]['per_paradigm'][p]['acc']*100:.1f}" for n in names]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| **Overall** | | " + " | ".join(f"**{results_by_model[n]['overall']*100:.1f}**" for n in names) + " |")
    return "\n".join(lines)

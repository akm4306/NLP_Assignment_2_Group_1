# NLP CS60075 Assignment II — Project 1 on WikiText-2

N-gram language modeling with smoothing (Task 1), a reproduction of the AWD-LSTM
(Task 2) and an interpolated KN + AWD-LSTM hybrid (Task 3), all trained and
evaluated on **WikiText-2** (`Salesforce/wikitext`, `wikitext-2-raw-v1`), plus the
shared BLiMP grammaticality benchmark. Perplexity is measured on the WikiText-2 test
split, and every hyper-parameter is tuned on validation only.

## How to run

| Option | Command | Notes |
|---|---|---|
| **Google Colab** | Open `26AI60R06_NLP_Assignment_2.ipynb` → *Runtime → Change runtime type → T4 GPU* → *Run all* | Self-contained: the notebook writes the `src/` files itself. Set `QUICK = True` for a quick test, and `USE_DRIVE = True` to keep checkpoints in Drive. |
| **Local** | `pip install -r requirements.txt && bash run_all.sh` | `QUICK=1 bash run_all.sh` for a quick test (a few minutes). |

`run_all.sh` runs Task 1 → Task 2 (train + eval) → Task 3 → figures. AWD-LSTM
training saves a checkpoint after every epoch and **resumes automatically** when
re-run. Each step can also be run on its own:

```bash
python src/task1_ngram.py                       # task 1
python src/task2_awd_lstm.py train --epochs 50  # task 2 training (needs a gpu)
python src/task2_awd_lstm.py eval               # task 2 test ppl + blimp, saves per-word scores for task 3
python src/task3_hybrid.py                      # task 3 (uses the outputs of task 1 and task 2)
python src/make_plots.py                        # figures
```

**Run times** (measured on a GTX 1650 GPU with 4 GB and a Ryzen 5 5600H CPU): Task 1
takes under 1 min. AWD-LSTM training takes about 5 min per epoch, 4.15 h for 50 epochs.
Task 2 evaluation takes about 2 min and Task 3 about 1 min.

## Layout

```
src/common.py          tokenizer, vocabulary, WikiText-2 / BLiMP loading, metrics (shared by all tasks)
src/task1_ngram.py     count tables, Add-k, Good-Turing (Katz), interpolated Kneser-Ney, λ grid search
src/task2_awd_lstm.py  AWD-LSTM model, NT-ASGD training loop, evaluation
src/task3_hybrid.py    λ-interpolation of KN and AWD-LSTM, final comparison, error analysis
src/make_plots.py      figures
26AI60R06_NLP_Assignment_2.ipynb   Colab notebook (contains the same src/ code)
run_all.sh             runs all three tasks with one command
results/               task{1,2,3}_results.md / .json tables, figures/
logs/                  console output of the runs that produced results/
tools/build_notebook.py  regenerates the notebook from src/
```

## Shared preprocessing (identical for all tasks)

* Each split is concatenated into **one token stream**, with no sentence or line markers.
* WikiText storage artifacts are undone: `@-@` → `-`, `@,@` → `,`, `@.@` → `.`, and
  `didn 't` → `didn't`, so the text is tokenized the same way as BLiMP's normal English.
* The tokenizer is punctuation-aware and lowercases everything. Words, clitics (`'s`,
  `n't`) and every punctuation mark become separate tokens.
* The vocabulary is the 30,000 most frequent training tokens with count ≥ 2, plus `<unk>`
  (|V| = 30,001). Everything else maps to `<unk>`.
* Every model scores exactly the same positions: tokens 2…N of each split, each
  conditioned on everything before it. This is what lets Task 3 mix the two models
  token by token.
* **BLiMP:** 10 paradigms, 1,000 pairs each, covering morphology, syntax and semantics
  (listed in `src/common.py`). Each sentence is tokenized with the same vocabulary and
  scored after the context token `.`, standing in for the end of a previous sentence
  because the stream has no `<s>` symbol. Scores are length-normalized log-probabilities,
  and a pair counts as correct when the model prefers the good sentence.

## Deviations from the paper and the assignment (stated explicitly)

* **AWD-LSTM epochs:** about 50 instead of the paper's 500–750, as allowed by the
  assignment. The model is the paper's full size (hidden size 1150).
* **Memory-saving loss on the 4 GB GPU:** the softmax cross-entropy is computed in
  chunks with a fused gradient (`ChunkedSoftmaxCE`). It is mathematically identical
  to the standard loss (verified against `F.cross_entropy` to 1e-16).
* **AWD-LSTM WT2 settings not fixed by the assignment** follow the official code:
  between-layer dropout 0.2, input dropout 0.65, AR α = 2, TAR β = 1, weight decay
  1.2e-6, variable-length BPTT, and NT-ASGD with non-monotone interval 5, checked once
  per epoch. Fine-tuning and pointer/cache models are not used.
* **Good–Turing:** Katz back-off with Good–Turing discounted counts for r ≤ 5. Pure Katz
  gives probability zero, and therefore infinite perplexity, to 225 test tokens (4-gram model).
  This happens when every continuation of a history occurred more than 5 times, so no
  mass is left for back-off. Such histories keep a minimum back-off mass of 1e-3.
  Counts without this floor are reported in `results/task1_results.json`.
* **Kneser–Ney with λ:** `P = λ·P_IKN(w|h) + (1−λ)·P_lower(w|h')` at every order ≥ 2, where
  P_IKN is textbook interpolated KN with d = 0.75. λ = 1 recovers standard IKN. λ is
  grid-searched over {0.1,…,0.9} on validation.

# NLP Project 1: N-gram Smoothing, AWD-LSTM, and Hybrid Language Modeling

**Course:** CS60075 — Natural Language Processing, IIT Kharagpur  
**Dataset:** WikiText-2 (raw)  
**Evaluation:** Perplexity (PPL) and BLiMP grammaticality accuracy

## Project overview

This project compares classical and neural approaches to next-word prediction and investigates whether combining them improves performance. The implementation is organized into three tasks in a single Jupyter notebook, `code.ipynb`:

1. **Task 1 — Classical language modeling:** Build a 4-gram model and compare Add-k, Good–Turing, and interpolated Kneser–Ney smoothing.
2. **Task 2 — Neural language modeling:** Implement and train a three-layer AWD-LSTM-inspired language model.
3. **Task 3 — Hybrid language modeling:** Interpolate Kneser–Ney and AWD-LSTM probabilities, selecting the mixing weight on validation data.

The notebook evaluates all three approaches using **WikiText-2 perplexity** and **10 BLiMP paradigms** (10,000 grammatical/ungrammatical sentence pairs).

## Repository structure

```text
.
├── code.ipynb                         # All three tasks; execute cells in order
├── data/
│   └── wikitext2/
│       └── wikitext-2-raw-v1/
│           ├── wiki.train.raw         # Training text
│           ├── wiki.valid.raw         # Validation text
│           └── wiki.test.raw          # Test text
├── results/
│   ├── task1/
│   │   ├── perplexity.csv             # Comparison of five smoothing variants
│   │   ├── kn_lambda_grid.csv         # Kneser–Ney interpolation tuning
│   │   ├── blimp.csv                  # BLiMP accuracy by paradigm
│   │   └── results.json               # Task 1 summary
│   ├── task2/
│   │   ├── training_log.csv           # Epoch-by-epoch training statistics
│   │   ├── blimp.csv                  # AWD-LSTM BLiMP results
│   │   ├── blimp_pairs.csv            # Individual BLiMP pair scores
│   │   ├── valid_token_logprobs.npy   # Validation token log-probabilities
│   │   ├── test_token_logprobs.npy    # Test token log-probabilities
│   │   └── results.json               # Task 2 summary
│   └── task3/
│       ├── omega_grid.csv             # Hybrid weight search
│       ├── comparison.csv             # Final comparison of all three models
│       ├── blimp.csv                  # Hybrid BLiMP results
│       ├── blimp_pairs.csv            # Individual hybrid BLiMP pair scores
│       ├── ngram_buckets.csv          # Performance by observed N-gram context
│       ├── token_examples.csv         # Examples of gains and regressions
│       ├── valid_token_logprobs.npy   # Hybrid validation log-probabilities
│       ├── test_token_logprobs.npy    # Hybrid test log-probabilities
│       └── results.json               # Task 3 summary
└── checkpoints/                      # Optional saved AWD-LSTM checkpoints
    ├── task2_awd_lstm_h1150_best.pt
    └── task2_awd_lstm_h1150_last.pt
```

**Note:** `data/`, `results/`, and `checkpoints/` show the paths expected or created by the notebook. Include the generated result files in the repository if you want readers to inspect the reported metrics. The notebook can download the WikiText-2 raw files automatically if its expected training file is missing; it also downloads the selected BLiMP paradigms through Hugging Face.

## Methodology

### Task 1: Classical 4-gram language model

- Lowercase and tokenize WikiText-2; retain up to **30,000** frequent vocabulary tokens (minimum training frequency: 2). Map rare or unseen tokens to `<UNK>` and use sentence-boundary tokens.
- Construct unigram, bigram, trigram, and 4-gram count tables.
- Implement **Add-k** smoothing for `k = 1, 0.5, 0.1`, **Good–Turing** smoothing for counts up to 5, and **interpolated Kneser–Ney** with discount `d = 0.75`.
- Tune Kneser–Ney's interpolation parameter over `λ = 0.1, 0.2, ..., 0.9` using **validation perplexity**.
- Evaluate the smoothing methods on the test set; evaluate the best method on BLiMP.

**Best classical model:** Kneser–Ney with `λ = 0.8`.

### Task 2: AWD-LSTM-inspired neural language model

- Use a **3-layer LSTM** with embedding size **400**, hidden sizes **1150 / 1150 / 400**, and tied input embedding/output weights.
- Apply **embedding dropout (0.1)**, **DropConnect (0.5)** on recurrent weights, and **output dropout (0.4)**.
- Train with an initial learning rate of **30**, batch size **80**, BPTT length **70**, gradient clipping at **0.25**, and **NT-ASGD** optimization.
- Train for **50 epochs**, selecting the best checkpoint using validation loss. In the recorded run, parameter averaging was triggered after **epoch 42**.
- Evaluate test perplexity and BLiMP accuracy; save per-token log-probabilities for Task 3.

This is a **course-level reproduction inspired by AWD-LSTM**, not a claim of identical training conditions or a bit-for-bit reproduction of the original paper.

### Task 3: Hybrid model

Interpolate the next-token probabilities from the two models:

\[
P_{\text{hybrid}}(w\mid h) = \omega P_{\text{KN}}(w\mid h) + (1-\omega) P_{\text{AWD}}(w\mid h)
\]

- Reuse the Kneser–Ney model from Task 1 and the trained AWD-LSTM from Task 2.
- Search `ω = 0.0, 0.1, ..., 1.0`, selecting the value with the lowest **validation perplexity**.
- Mix probabilities in log space using a numerically stable log-sum-exp calculation.
- Evaluate the selected hybrid on test perplexity and BLiMP; analyze gains and losses by N-gram context frequency and by individual examples.

**Best mixing weight:** `ω = 0.1` (10% Kneser–Ney + 90% AWD-LSTM).

## Experimental results

### Task 1: Smoothing comparison

| Smoothing method | Validation PPL ↓ | Test PPL ↓ |
|---|---:|---:|
| Add-k (`k = 1`) | 21,546.98 | 21,259.14 |
| Add-k (`k = 0.5`) | 19,902.88 | 19,585.38 |
| Add-k (`k = 0.1`) | 16,250.31 | 15,893.70 |
| Good–Turing | 9,781.20 | 9,566.91 |
| **Interpolated Kneser–Ney** | **243.95** | **233.29** |

Kneser–Ney achieved **62.91% BLiMP accuracy** across the 10 selected paradigms.

### Overall comparison

| Model | Validation PPL ↓ | Test PPL ↓ | BLiMP accuracy ↑ |
|---|---:|---:|---:|
| Task 1: Kneser–Ney | 243.95 | 233.29 | 62.91% |
| Task 2: AWD-LSTM | 90.85 | 87.41 | **77.37%** |
| **Task 3: Hybrid (`ω = 0.1`)** | **86.88** | **84.22** | 75.22% |

The hybrid model reduced test perplexity by approximately **3.66%** compared with AWD-LSTM alone, but its BLiMP accuracy was **2.15 percentage points lower**. Thus, the hybrid improved next-word probability prediction without improving overall grammatical sentence preference. The standalone AWD-LSTM achieved the highest BLiMP accuracy.

## Setup and execution

**Requirements:** Python 3.10+ and a Jupyter-compatible environment. A CUDA GPU is recommended for Task 2; the notebook also selects Apple MPS or CPU when available. Training on CPU may be slow.

Install the packages used by the notebook:

```bash
python -m pip install jupyter numpy pandas pyarrow huggingface_hub torch
```

Start Jupyter:

```bash
jupyter notebook code.ipynb
```

Run the notebook's **code cells in order**: Task 1 → Task 2 → Task 3. They share in-memory variables, vocabulary mappings, and model outputs; Task 3 is **not standalone**. If WikiText-2 raw files are missing, Task 1 downloads them into `data/wikitext2/wikitext-2-raw-v1/`. The BLiMP download also requires internet access on first use.

Task 2 saves its training state in `checkpoints/task2_awd_lstm_h1150_last.pt` and automatically resumes from that file if present. It also saves the best validation checkpoint as `checkpoints/task2_awd_lstm_h1150_best.pt`.

### GitHub upload note

The `.pt` checkpoint files from this run are approximately **129 MB** and **258 MB**, exceeding GitHub's normal **100 MB per-file limit**. Either keep them out of ordinary Git commits and provide a separate download link, or track them with **Git LFS**:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

Generated `.npy` files can also be omitted if repository size is a concern; the notebook regenerates them.

## References

- Merity, S., Keskar, N. S., & Socher, R. (2017). *Regularizing and Optimizing LSTM Language Models.* [arXiv:1708.02182](https://arxiv.org/abs/1708.02182).
- [WikiText-2 on Hugging Face](https://huggingface.co/datasets/Salesforce/wikitext).
- [BLiMP dataset](https://huggingface.co/datasets/nyu-mll/blimp).

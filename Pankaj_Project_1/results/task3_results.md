# Task 3 - Interpolated KN + AWD-LSTM hybrid

## λ grid search (λ = weight of KN; tuned on validation only)

| λ | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Validation PPL | 90.51 | 87.46 | 89.05 | 92.10 | 96.47 | 102.34 | 110.22 | 121.07 | 137.05 | 164.18 | 253.71 |
| Test PPL (not used for selection) | 86.26 | 84.14 | 85.87 | 88.94 | 93.24 | 98.97 | 106.62 | 117.13 | 132.54 | 158.58 | 243.66 |
| BLiMP overall % | 69.9 | 68.7 | 68.0 | 67.4 | 66.8 | 65.9 | 65.2 | 64.4 | 63.4 | 62.1 | 56.3 |

Selected λ = **0.1**.

## Final comparison (Task 1 vs Task 2 vs Task 3)

| Model | Validation PPL | Test PPL | BLiMP overall % |
|---|---|---|---|
| Task 1: KN 4-gram | 253.71 | 243.66 | 56.3 |
| Task 2: AWD-LSTM | 90.51 | 86.26 | 69.9 |
| Task 3: Hybrid (λ=0.1) | 87.46 | 84.14 | 68.7 |

## BLiMP per paradigm (%)

| Paradigm | Field | T1: KN 4-gram | T2: AWD-LSTM | T3: Hybrid λ=0.1 |
|---|---|---|---|---|
| anaphor_gender_agreement | morphology | 23.6 | 53.3 | 47.3 |
| determiner_noun_agreement_1 | morphology | 52.6 | 83.4 | 78.8 |
| regular_plural_subject_verb_agreement_1 | morphology | 56.1 | 82.6 | 81.7 |
| irregular_past_participle_adjectives | morphology | 78.6 | 93.8 | 94.7 |
| wh_island | syntax | 52.1 | 53.3 | 54.9 |
| adjunct_island | syntax | 34.8 | 53.5 | 51.1 |
| drop_argument | syntax | 39.5 | 38.2 | 37.1 |
| npi_present_1 | semantics | 40.5 | 44.1 | 43.2 |
| principle_A_domain_1 | syntax_semantics | 97.8 | 96.8 | 98.7 |
| existential_there_quantifiers_1 | semantics | 87.3 | 99.8 | 99.9 |
| **Overall** | | **56.3** | **69.9** | **68.7** |

## Where does the n-gram help? Test PPL by token bucket

| Bucket | Share of test tokens | KN 4-gram | AWD-LSTM | Hybrid (λ=0.1) |
|---|---|---|---|---|
| 4-gram seen in train | 14.7% | 7.4 | 7.2 | 6.4 |
| 4-gram unseen, bigram seen | 65.1% | 144.3 | 58.2 | 56.8 |
| bigram unseen | 20.2% | 16842.1 | 1871.0 | 1948.8 |
| target freq > 10k | 42.8% | 16.3 | 7.8 | 7.9 |
| target freq 1k-10k | 15.3% | 158.2 | 59.0 | 58.4 |
| target freq 100-1k | 24.1% | 1258.8 | 355.1 | 336.2 |
| target freq <= 100 | 17.7% | 26285.4 | 5758.8 | 5337.7 |
| target = <unk> | 5.6% | 39.4 | 23.9 | 23.8 |

The hybrid assigns higher probability than the AWD-LSTM alone to 23.4% of test tokens.

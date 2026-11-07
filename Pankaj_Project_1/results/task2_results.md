# Task 2 - AWD-LSTM results

Best checkpoint: epoch 49 (hidden size 1150).

| Split | Perplexity |
|---|---|
| Validation | 90.51 |
| **Test** | **86.26** |

## BLiMP accuracy (%)

| Paradigm | Field | AWD-LSTM |
|---|---|---|
| anaphor_gender_agreement | morphology | 53.3 |
| determiner_noun_agreement_1 | morphology | 83.4 |
| regular_plural_subject_verb_agreement_1 | morphology | 82.6 |
| irregular_past_participle_adjectives | morphology | 93.8 |
| wh_island | syntax | 53.3 |
| adjunct_island | syntax | 53.5 |
| drop_argument | syntax | 38.2 |
| npi_present_1 | semantics | 44.1 |
| principle_A_domain_1 | syntax_semantics | 96.8 |
| existential_there_quantifiers_1 | semantics | 99.8 |
| **Overall** | | **69.9** |

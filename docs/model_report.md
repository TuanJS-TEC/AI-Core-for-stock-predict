# Sentiment model report (2026-10-21)

## Data
- Corpus: 17,050 articles (CafeF 13,740, Vietstock 3,310), 2021-10-21 .. 2026-10-20, title-prefiltered on 15 tickers. 33,921 article-company results.
- Annotation: 1,800 samples. `dev` = published before 2026-05-01 (1,528), `eval` = on/after (272, frozen, human-labeled 2026-10-26). Leak check: 0 near-duplicate dev/eval pairs.
- Dev labels are **silver**: by Claude (`annotator_id=claude-opus-5-5`, guideline v1; pilot 100 by the main session, rest by 7 parallel sub-sessions). Human agreement measured on the blind set, see below.
- Dev label mix (1,528): neutral 892, positive 364, negative 68, skip 190, unclear 14. Negative is rare (5% of trainable rows).

## Validation (time split inside dev: last 20% by published_at, from 2025-09-17; 265 rows, only 5 negative)
| Model | version | macro-F1 | F1 pos / neu / neg |
|---|---|---|---|
| Lexicon placeholder | lexicon-v0 | 0.403 | 0.48 / 0.62 / 0.11 |
| TF-IDF + LR | baseline-tfidf-lr-v1 | 0.650 | 0.63 / 0.85 / 0.46 |
| PhoBERT-base-v2, 5 ep, best epoch 3 | phobert-phobert-v1 | 0.702 | n/a |

Caveats: PhoBERT's epoch was chosen on this same validation set (optimistic); baseline was not tuned on it. Negative-class F1 rests on 5 examples. These are agreement-with-Claude scores, not with humans. The real number is on the human-labeled `eval` stream.

## Human `eval` stream (2026-10-26)
272 samples labeled by `tuanhm`. **Labeled twice.** Round 1 (`data/eval_to_label_craft.csv`) applied an intuitive
"is this good news for the company" reading; round 2 (`data/eval_labeled.csv`, `guideline_version=v1-strict`) applied the
guideline's decision rules. 115 of 272 labels changed (positive->neutral 74, neutral->positive 18, negative->neutral 13,
negative->skip 11, unclear->neutral/skip 10). Round-2 mix: 58 positive, 178 neutral, 4 negative, 29 skip, 3 unclear; 240 scored.

| Model | macro-F1 dev val (vs Claude) | macro-F1 eval round 1 | **macro-F1 eval round 2** | F1 pos / neu / neg | AUC pos-vs-neg |
|---|---|---|---|---|---|
| Lexicon placeholder | 0.403 | 0.465 | 0.319 [0.274, 0.360] | 0.38 / 0.58 / 0.00 | 0.603 |
| TF-IDF + LR | 0.650 | 0.374 | **0.608** [0.441, 0.726] | 0.50 / 0.82 / 0.50 | 0.970 |
| PhoBERT-base-v2 | 0.702 | 0.407 | **0.626** [0.462, 0.742] | 0.53 / 0.85 / 0.50 | 0.978 |

CI = 2,000-row bootstrap over the 240 eval rows.

**Round 1 measured the annotator, not the model.** Hand-adjudication of all 98 rows of the blind set and all 26
Claude-neutral/human-directional eval rows found the round-1 labels broke four written rules: price/volume/fund-flow news
(5 rows), promotions, product launches, sponsorship and customer notices (9), executive optimism quotes (3), and target
mentioned only as someone else's holding, supplier or landmark (skip, several). Claude was right on ~20 of those 26,
wrong on ~3 (new large project and new license are `positive` per the rules - now the only guideline gap found).
Under the rules the model was trained on, the ranking is restored and matches dev validation.

Agreement, blind re-label of 100 dev samples (`data/check_blind_labeled.csv`, 98 returned) vs Claude's snapshot
(`data/check_blind_claude_labels.csv`): raw 0.500, Cohen kappa 0.233 over 5 categories; 0.576 / kappa 0.297 over the
85 rows both call pos/neu/neg. But where both commit to a direction (34 rows) the sign matches on 32. Measured against
round-1 labels, so it is a lower bound; not re-measured after round 2.

**Model choice: ship the TF-IDF baseline.** PhoBERT is +0.018 macro-F1, far inside overlapping CIs, and costs torch,
a GPU and the VnCoreNLP jar. Both rank pos-vs-neg at AUC ~0.97, so direction is solved; what is left is the
neutral boundary, where they are indistinguishable.

**Negative is near-absent by construction.** 4 negatives in 240 rows, so one third of macro-F1 rests on 4 samples
(hence CIs ~0.28 wide). Adjudication of the 24 rows that moved off negative found the moves almost all correct: for the
bank tickers, "negative-looking" news is fraud reports, tax enforcement and customer notices where the bank is only
where an account sits - correctly `skip`/`neutral`. One arguable miss: Mexico's 285.5 USD/t anti-dumping duty on HPG is
a sanction (rule says `negative`), labeled neutral because the article frames the impact as small. Macro-F1 on this
eval set cannot be made reliable without targeted negative sampling.
## Shipped artifacts
- `data/models/baseline.joblib` (trained on all 1,324 dev labels), `data/models/phobert-v1` (trained on first 80% by time).
- Backtest snapshots, 15 tickers x 1,826 daily cutoffs: `data/snapshots_baseline_backtest.parquet`, `data/snapshots_phobert_backtest.parquet` (27,390 rows each, 72% with news). Daily sentiment correlation baseline vs PhoBERT: 0.84. Sanity: NVL is the only ticker with negative mean sentiment (2022-24 debt crisis).

## Trial forecast with user price files (`data/stock/`, run 2026-10-21, reproduced 2026-10-26)
- Price data: 15 files x 1,444 sessions (2021-01-04 .. 2026-10-21), identical calendars, no gaps/NaN/duplicates, OHLC consistent, no move beyond the 7% HOSE band (looks dividend/split-adjusted, in thousand VND). Only 10 tickers overlap the NLP universe; missing BID, CTG, HDB, VHM, NVL; the extra VRE, MBB, SSI, VNM, GAS have little collected news (8-135 results). `data/prices.csv` = the 10 overlapping tickers.
- Setup: `trial_forecast` features, rows from 2021-10-21, test from 2025-09-17 (after PhoBERT v1's training labels), h-session purge, backtest availability. Baseline NLP model was trained on labels up to 2026-04, so its row overlaps the test period.
- Reproducible since 2026-10-26: `python -m nlp_module.forecast.trial_forecast --model-version baseline-tfidf-lr-v1` (the 2026-10-21 run was an ad-hoc script). `--test-start` defaults to 2025-09-17 and must stay at/after the last label the NLP model saw; leaving it to `--test-frac 0.3` instead starts the test at 2025-04-18, inside PhoBERT's training labels, and shrinks every effect (T+5 diff -0.014 instead of -0.029).

| NLP model | h | AUC price | AUC price+text | diff [95% CI, date bootstrap] |
|---|---|---|---|---|
| PhoBERT | T+2 | 0.5189 | 0.5105 | -0.0084 [-0.0201, +0.0039] |
| PhoBERT | T+5 | 0.5463 | 0.5143 | -0.0321 [-0.0502, -0.0145] |
| Baseline | T+2 | 0.5189 | 0.5107 | -0.0081 [-0.0197, +0.0042] |
| Baseline | T+5 | 0.5463 | 0.5170 | -0.0293 [-0.0473, -0.0113] |

Re-run on 2026-10-26 with the production baseline: identical to the 2026-10-21 figures (9,710 train / 2,710 test rows at
T+2; 9,680 / 2,680 at T+5). The `eval` re-labelling does not touch this - eval labels were never used for training, so
`baseline.joblib` and its article scores are unchanged.

### Drift fix (2026-10-26): the T+5 harm was drift, not text
Raw text features drift across the train/test boundary: mean `news_count_72h` 1.755 -> 3.029, `news_count_24h`
0.666 -> 1.136, `sentiment_mean` 0.150 -> 0.196. `rank_cs()` replaces a feature by its cross-sectional rank within
each date, which is invariant to any level shift hitting every ticker (verified: 0.55 -> 0.55 on both sides, and
`test_cross_sectional_rank_removes_a_level_shift`). Same-day values only, so no look-ahead.

| `--rank` | ranked features | T+2 diff [95% CI] | T+5 diff [95% CI] |
|---|---|---|---|
| `none` | - (raw levels) | -0.0081 [-0.0197, +0.0042] | **-0.0293 [-0.0473, -0.0113]** |
| `counts` (default) | `news_count_24h/72h` | -0.0017 [-0.0174, +0.0146] | -0.0118 [-0.0288, +0.0065] |
| `all` | + `sentiment_mean(_24h)` | -0.0049 [-0.0209, +0.0118] | +0.0005 [-0.0151, +0.0179] |

**The only statistically significant result in this whole ablation - text hurting at T+5 - was an artifact of
feature drift.** De-drift the features and it vanishes. The earlier "text features hurt at T+5" claim is withdrawn.

What replaces it is weaker, not better: **no detectable effect either way.** Every CI above contains 0 except the
drifting `none` row. Text neither helps nor hurts direction forecasting on these 10 tickers.

Caveat on the `all` row: the rank mode was chosen by looking at test AUC, which is selection on test. The defensible
claim is only the negative one (the harm was drift); quoting +0.0005 as evidence that text is neutral-to-positive
would need the rank mode picked on validation, i.e. wired into `walk_forward` instead.

- Older diagnostic, unchanged: yearly Spearman IC of `sentiment_mean` vs forward returns is within +-0.05 and flips sign between years.
- Not a verdict on text in general: raw direction is dominated by the market move, the five most news-driven tickers (incl. NVL) have no prices yet, and the model is the simplest linear one.

## Walk-forward, 3 folds (`python -m nlp_module.forecast.walk_forward`, run 2026-10-21)
Per fold, PhoBERT is retrained on dev labels published before the validation window (epoch chosen on labels inside it) and re-scores every article (`nlp_model_version = phobert-fold{A,B,C}`). The forecaster (logistic regression, C in {0.01, 0.1, 1, 10}) and the feature set are chosen on validation, refit on train+validation, and scored once on test. h-session purge before validation and before test. 10 tickers.

| Fold | Train / Val / Test | NLP labels (train/val), val F1 | h | AUC test price | AUC test price+text | Picked on val |
|---|---|---|---|---|---|---|
| A | 2022-23 / 2024H1 / 2024H2 | 480 / 143, 0.638 | 2 | 0.513 | 0.490 | price |
| A | | | 5 | 0.539 | 0.498 | price |
| B | 2022-24 / 2025H1 / 2025H2 | 765 / 206, 0.791 | 2 | 0.502 | 0.505 | price+text |
| B | | | 5 | 0.487 | 0.502 | price+text |
| C | 2022-25 / 2026H1 / 2026-07..10 | 1,195 / 129, 0.758 | 2 | 0.524 | 0.509 | price |
| C | | | 5 | 0.585 | 0.525 | price |

Text beats price-only on test in fold B only (+0.003, +0.015) and loses in A and C. Every AUC is within ~0.05 of 0.5 except fold C T+5 price-only (0.585, 730 test rows over 3.5 months, highly correlated across tickers). No evidence yet that these text features help direction forecasting.

### Re-run 2026-10-26 with the rank mode in the validation search
`--rank` is no longer a hand-set flag here: `none` / `counts` / `all` joins the feature set and C in the grid chosen
on validation, so the de-drifted features are picked without ever consulting test (`walk_forward --skip-nlp`, reusing
the `phobert-fold*` scores already in the DB).

| Fold | h | AUC test price | best text on val | AUC test, that text | Picked on val | Picked test |
|---|---|---|---|---|---|---|
| A | 2 | 0.513 | `text/all` | 0.514 | price | 0.513 |
| A | 5 | 0.539 | `text/all` | 0.501 | price | 0.539 |
| B | 2 | 0.502 | `text/all` | 0.521 | `text/all` | 0.521 |
| B | 5 | 0.487 | `text/none` | 0.502 | `text/none` | 0.502 |
| C | 2 | 0.524 | `text/all` | 0.511 | price | 0.524 |
| C | 5 | 0.585 | `text/all` | 0.573 | price | 0.585 |

Two things this settles:

- **Validation prefers the de-drifted features on its own** - `text/all` wins the text slot in 5 of 6 cells, chosen on
  validation data only. The selection-on-test caveat on the `trial_forecast` drift table is therefore lifted: ranking
  the count and sentiment features is justified without looking at test.
- **De-drifting raises text's test AUC in 5 of 6 cells** vs. the 2026-10-21 run (0.490->0.514, 0.498->0.501,
  0.505->0.521, 0.502 unchanged, 0.509->0.511, 0.525->0.573). Mean text test AUC 0.505 -> 0.520.

What it does not change: text still trails price-only on average (0.520 vs 0.525). Validation picks text in only 2 of 6
cells, both in fold B. Following validation yields mean test AUC 0.5287 vs 0.5250 for always-price - +0.004 over six
folds of ~1,200 test rows each that are highly correlated across tickers, i.e. noise. The verdict stands: **no reliable
direction-forecasting benefit from these text features.** The drift fix removed an artifact, not a hidden signal.

Worth noting the selection itself behaved: validation picked text exactly in the two cells where text won on test, and
price in the four where price won or tied. The protocol works; the signal is what is missing.

## Next
1. ~~Human labels for `eval`~~ done 2026-10-26 (twice; round 2 is the valid one). Production model = `data/models/baseline.joblib`.
2. Targeted negative sampling: draw dev passages both models score negative, label, retrain. Until then macro-F1 is a 4-sample number.
3. Add the two guideline gaps (new large project, new license/business line -> `positive`) and the HPG sanction case; bump to v2.
4. Re-measure Claude-vs-human agreement on a fresh blind set under v2. The 0.297 kappa is against discarded round-1 labels.
5. Dev labels need no re-labeling: Claude's dev prior (27/67/5) now matches the human's rule-following eval prior (24/74/2).
6. ~~`trial_forecast` ablation with the baseline model~~ done 2026-10-26, reproduced the 2026-10-21 figures exactly.
7. ~~Confound (a), `news_count_*` drift~~ fixed 2026-10-26 (`rank_cs`, `--rank`), and it removed the only significant effect in the ablation.
8. ~~Pick `--rank` on validation inside `walk_forward`~~ done 2026-10-26: validation chooses `text/all` in 5 of 6 cells, so the drift fix is justified off-test. Text still shows no reliable benefit.
9. ~~Confound (b), the target~~ excess-return target added 2026-10-26 (`--target excess`). It does not rescue text; see below.
10. ~~Magnitude target~~ done 2026-10-26 (`--target absmove`): news volume does predict magnitude, but redundantly given `vol10`. See below.
11. ~~`uncertain_ratio` is a dead feature~~ fixed 2026-10-26: `abstain` command, threshold picked on dev validation via a calibration twin. See below.
12. Remaining options: (a) prices for the five news-heaviest tickers (BID, CTG, HDB, VHM, NVL - NVL is the only ticker with negative mean sentiment and it is absent from every backtest here); (c) stop treating text as alpha and ship the `evidence` layer, which needs none of this to be useful.

## Market-excess target (`--target excess`, 2026-10-26)
Raw direction is ~dominated by the market move, so the label becomes "did this ticker beat the market", with the market
proxied equal-weight by the mean forward return of the 10 tickers trading that day (`add_targets(excess=True)`;
`test_excess_target_strips_the_market_move`). The proxy reads forward returns, which is legitimate - it is part of the
label, not of the features.

Single split (`trial_forecast`, test from 2025-09-17, baseline scores):

| target | `--rank` | T+2 AUC price / price+text, diff [CI] | T+5 AUC price / price+text, diff [CI] |
|---|---|---|---|
| direction | `all` | 0.5189 / 0.5140, -0.0049 [-0.0209, +0.0118] | 0.5463 / 0.5468, +0.0005 [-0.0151, +0.0179] |
| excess | `none` | 0.5323 / 0.5209, -0.0113 [-0.0281, +0.0064] | 0.5330 / 0.5189, -0.0142 [-0.0291, +0.0009] |
| excess | `counts` | 0.5323 / 0.5149, -0.0174 [-0.0344, -0.0002] | 0.5330 / 0.5120, -0.0211 [-0.0352, -0.0069] |
| excess | `all` | 0.5323 / 0.5121, -0.0202 [-0.0368, -0.0037] | 0.5330 / 0.5236, -0.0095 [-0.0262, +0.0073] |

The excess target helps the PRICE features at T+2 (0.5189 -> 0.5323): stripping the market move does expose something a
per-ticker model can read. It does not help text - every text diff is negative, three of six significantly.

Walk-forward, rank and feature set chosen on validation (`walk_forward --skip-nlp --target excess`):

| Fold | h | AUC test price | best text on val | AUC test, that text | Picked on val | Picked test |
|---|---|---|---|---|---|---|
| A | 2 | 0.474 | `text/none` | 0.488 | `text/none` | 0.488 |
| A | 5 | 0.482 | `text/all` | 0.504 | `text/all` | 0.504 |
| B | 2 | 0.518 | `text/none` | 0.545 | `text/none` | **0.545** |
| B | 5 | 0.531 | `text/counts` | 0.533 | price | 0.531 |
| C | 2 | 0.523 | `text/counts` | 0.501 | price | 0.523 |
| C | 5 | 0.535 | `text/none` | 0.512 | price | 0.535 |

Following validation gives mean test AUC 0.5210 vs 0.5105 for always-price, so +0.0105 - about 3x the +0.0037 edge on
the direction target, and validation again picked correctly in all six cells (text in the three where text won).

**But the two setups contradict each other**, and that is the finding. The single split with a date-bootstrap CI says
text hurts on the excess target; the walk-forward says it helps by +0.01. They differ in test period and in which NLP
model scored the articles (baseline trained on all labels vs. per-fold PhoBERT). An effect that flips sign with the
split is not an effect. Fold A's price-only AUCs are 0.474 and 0.482 - actively anti-predictive - so that fold's
comparison is not trustworthy in either direction.

Conclusion after three confounds (drift, model choice, target): **no stable forecasting signal in these text features
on these 10 tickers.** What is left is a magnitude target, the five missing news-heavy tickers, or accepting that the
value of this subsystem is the `evidence` layer rather than alpha.

## Magnitude target (`--target absmove`, 2026-10-26)
y = 1 if the ticker's |forward h-session return| beat the cross-sectional median that day. Cross-sectional, so the
market-wide volatility regime cancels and the base rate is 0.500 by construction
(`test_absmove_target_ignores_sign_and_ranks_magnitude`).

**Magnitude is far more predictable than direction - this part of the hypothesis held.** Price-only AUC, single split:

| target | T+2 price-only | T+5 price-only |
|---|---|---|
| direction | 0.5189 | 0.5463 |
| excess | 0.5323 | 0.5330 |
| **absmove** | **0.5728** | **0.5814** |

Text ablation on absmove (single split, baseline scores): T+2 diff -0.0104 / -0.0095 / -0.0036 and T+5 +0.0031 /
+0.0017 / -0.0010 for `--rank none/counts/all`. Walk-forward with everything chosen on validation picked price-only in
5 of 6 cells; mean test AUC 0.5557 following validation vs 0.5532 always-price (+0.0025).

### Why: the signal is real but redundant
Univariate AUC against the T+5 magnitude label on the test window (feature alone, no model, de-drifted by rank):

| feature | AUC | | feature | AUC |
|---|---|---|---|---|
| `vol10` | **0.582** | | `news_count_72h` | **0.558** |
| `ret1` | 0.515 | | `news_count_24h` | 0.544 |
| `ret5` | 0.511 | | `sentiment_mean_24h` | 0.534 |
| | | | `has_news` | 0.506 |

`news_count_72h` alone reaches 0.558 - genuine standalone signal, and more than anything in this project ever reached
on direction. **The news-predicts-magnitude hypothesis is confirmed.** But fitted on the same train window:

| model | T+2 | T+5 |
|---|---|---|
| `vol10` only | 0.573 | 0.582 |
| text only (7 features) | 0.514 | 0.536 |
| `vol10` + text | 0.569 | 0.581 |
| price + text | 0.569 | 0.581 |

Adding all seven text features to `vol10` changes nothing (0.582 -> 0.581). This is not collinearity -
corr(`news_count_72h`, `vol10`) is only 0.187 - it is that text's magnitude signal is weaker and already covered for
this purpose by trailing realised volatility, which is free and needs no NLP.

### Bottom line after four confounds
Drift, model choice, target sign, target magnitude - all tested. **News volume predicts how much a stock will move,
but realised volatility predicts it better and for nothing.** The text features have no incremental forecasting value
on these 10 tickers. For a change in that verdict the open leads are (a) the five news-heaviest tickers, which have no
price data, and (b) features that `vol10` cannot see - event-type content, or news arriving on a low-volatility ticker
(surprise) rather than news volume as a level. Both are hypotheses, not findings.

Independent of all of it, the `evidence` layer stands: F1 0.626 and AUC 0.978 pos-vs-neg on human labels is useful for
showing a reader why a ticker is in the news, which needs no incremental AUC to justify.

## Live run on real news, 2026-10-26
First end-to-end run on same-day news, with logging added (`-v` = what was fetched, matched and scored, `-vv` = every
url, link, passage and probability; stdlib `logging`, configured in `cli.py`).

```
nlp_module -v  collect cafef 2026-10-26 2026-10-26 --symbols <15 tickers> --limit 40
nlp_module -vv process --model baseline:data/models/baseline.joblib
nlp_module -v  snapshot --model baseline:... --mode live --start 2026-10-26 --end 2026-10-26
```

235 urls listed in the day's sitemap bucket, 223 title-filtered, **12 stored, 0 failed, 15 s wall clock** at 1 req/s.
Processing gave 25 (article, company) results: 12 `direct`, 10 `market`, 3 `ambiguous`; 16 flagged `needs_review`.
Live snapshot built for 9 tickers, 8 with news. The plumbing works unmodified on live data.

### What worked
- **The list-article heuristic.** A deposit-rate table ("gửi 100 triệu tại Sacombank, SHB, HDBank, MB, VPBank,
  Techcombank và ACB") linked 5 universe tickers, all correctly demoted to `relation=market` and all scored neutral
  0.73-0.75 - the guideline's "deposit-rate table -> neutral" rule, reproduced without being told. Because `market`
  rows are excluded from per-company features, HPG's only article of the day (a multi-company earnings roundup) left
  HPG at `has_news=False`, which is the intended behaviour.
- Two near-duplicate personal-interest pieces about a former Vingroup executive: both neutral. Correct.
- BIDV seizing collateral: neutral. Defensible.

### Errors, and they are the ones already on file
Three sentiment errors in 12 `direct` results, all in the same direction the `eval` stream already measured
(positive recall 0.48 - the model under-calls positive):

| Ticker | Headline | Model | Should be | Why |
|---|---|---|---|---|
| VIC | project of 23,500bn VND invested by Vingroup | neutral 0.52 | positive | new large project - **the exact guideline gap found in the blind-set adjudication** |
| VIC | raised hundreds of billions of KRW in Korean bonds at 8% | neutral 0.64 | positive | successful international funding, same family as "new licence" |
| MWG | unconditional guarantee for Bách Hóa Xanh to borrow 2,000bn | **positive 0.64** | neutral/negative | the parent is assuming contingent liability; the model appears to read "financing + large number" as good |

The MWG row is the one to worry about: confidently wrong in the wrong direction, and not flagged for review.

Two linking errors, both the family adjudicated earlier as `skip`:

- **VHM** on "BIDV thu giữ biệt thự tại Vinhomes Riverside" - the villa is merely *located* in Vinhomes Riverside, a
  landmark, which the guideline calls `skip`. The linker gave it `direct`, score 1.000, because "Vinhomes" is in the
  title. Scored neutral, so the harm is a spurious row inflating `news_count` and diluting `sentiment_mean`.
- **VCB** on a bank-transfer fraud report - the bank is only where the account sits. `direct`, score 0.800, neutral.

### One finding that only a live run exposes
The same article gave **VIC neutral 0.33/0.52 and VHM positive 0.43/0.43**. The body discusses "Vinhomes Vũng Áng", so
the passage selector found no body sentence for VIC and fell back to a 97-character title-only passage, while VHM got
450 characters of substance. Per-company passage selection, not the sentiment model, decided the disagreement.

(An earlier draft of this section claimed title-only passages were not flagged `difficult`. That was wrong: all 1,751
title-only results in the corpus carry `difficult=1` and `needs_review=1` - `passages.py` already covers the case via
`len(link.mentions) == 0 and not ids`. The real gap was downstream, fixed below.)

Also: 79 of 17,062 versions link no company, so they write no result row and are re-linked on every `process` run
(~0.5%, marked with a `ponytail:` note in `pipeline.py`).

## `difficult_ratio`, FEATURE_VERSION v2 (2026-10-26)
The `difficult` flag was computed and stored per result, then dropped at the contract boundary:
`ArticleCompanyResult` had no such field, so `build_snapshot` could not see it and a 97-character title-only passage
fed `sentiment_mean` with the same weight as 961 characters of substance. v2 adds the field and one feature:

```
difficult_ratio = share of the 72h window that is weak evidence (another company in the passage, or title-only)
```

`difficult` rows stay inside `sentiment_mean` - unlike `uncertain`, which is excluded. The ratio only exposes how much
of the window is weak so the consumer can discount it, which keeps every historical `sentiment_mean` comparable.
Snapshots rebuilt for both models: 27,390 rows x 17 columns, 19,710 with news. v1 rows are untouched in
`feature_snapshot` (FEATURE_VERSION is part of its key).

Distribution over the 19,710 news rows: mean 0.281, median 0, **zero in 52.0% of rows and 1.0 in 13.6%**. So one
snapshot row in seven is built *entirely* from weak-evidence passages - a useful quality signal for the evidence layer,
which can now say so instead of showing a confident-looking number.

Forecasting contribution: none. Univariate AUC against the magnitude label is 0.514 (T+2) and 0.524 (T+5), against
0.553 for `news_count_72h` and 0.582 for `vol10`. Adding it to the text set moves every ablation diff within noise
(direction T+5 `rank=all`: +0.0005 -> -0.0039; absmove T+5 `rank=counts`: +0.0017 -> +0.0009), and walk-forward's
validation-selected edge goes from +0.0037 to +0.0028 on direction and +0.0025 to +0.0040 on absmove. Noise both ways.

### Dead feature found while checking this (fixed, see next section)
**`uncertain_ratio` was constant 0.0 across all 19,710 news rows** (`nunique() == 1`, univariate AUC exactly 0.500).
Nothing is ever flagged uncertain because `--abstain-entropy` was never passed in any run in this project, and
`decide()` only abstains when a threshold is given. Every ablation reported above therefore ran with 6 live text
features, not 7. `StandardScaler` zeroes a zero-variance column, so no conclusion changes - but the abstention
machinery, including `evaluate.choose_abstain`, has never been exercised. Either pick a threshold on validation and
pass it to `process`, or drop the feature.

## Abstention turned on, threshold chosen on validation (2026-10-26)
New command: `nlp_module abstain --model <spec> [--min-acc 0.85] [--apply] [--apply-to <version>]`. It scores the dev
validation split (last 20% by `published_at`, 265 rows from 2025-09-17 - never the `eval` stream), prints the coverage
/ accuracy curve from `evaluate.abstention_curve`, and picks the largest coverage reaching `--min-acc`.

**The production baseline cannot be calibrated on dev validation**: it was trained on all 1,324 dev labels, so those
265 rows are in its training set and it reports 0.962 full-coverage accuracy with 100% accuracy out to 62% coverage -
memorisation, not calibration. `train-baseline --val-frac 0.2` now trains a calibration twin on the first 80% by time
(`data/models/baseline-cal.joblib`, `baseline-tfidf-lr-v1-cal`, 1,059 samples); its threshold is applied to the
production version with `--apply-to`. PhoBERT v1 was already trained on the first 80%, so dev validation is a genuine
holdout for it - though its epoch was chosen on that same split, so its threshold is mildly optimistic.

Calibration twin curve (true holdout, full-coverage accuracy **0.784**):

| abstain if H > | coverage | accuracy on kept |
|---|---|---|
| 0.727 | 0.279 | 0.946 |
| 0.830 | 0.555 | 0.905 |
| 0.888 | 0.740 | **0.852** |
| 0.908 | 0.811 | 0.828 |
| 0.953 | 0.891 | 0.801 |
| 0.999 | 0.996 | 0.784 |

Chosen at `--min-acc 0.85`: **baseline H > 0.888** (coverage 0.740), **PhoBERT H > 0.571** (coverage 0.732). The
thresholds differ by a lot because a softmax head is far sharper than TF-IDF + LR; both land at ~73% coverage, which is
the number that matters. Applied to the stored results of `baseline-tfidf-lr-v1`, `phobert-phobert-v1` and
`phobert-fold{A,B,C}`. `uncertain` is a pure function of the stored `entropy` column, so changing the threshold is one
UPDATE and needs no re-scoring.

Snapshots rebuilt. `uncertain_ratio` is now live: mean 0.197 (baseline) / 0.241 (PhoBERT), 64 / 80 distinct values.
**10.7% of news rows abstained on the whole window**, so `sentiment_mean` is missing there - which is the designed
behaviour ("uncertain is never counted as neutral"), and those rows are imputed to sentiment 0 *with* `has_news=1`,
flagged by `uncertain_ratio = 1`. The imputation comment in `add_text` now says so.

### The trade-off, stated plainly
- **Evidence layer gains**: accuracy on labels actually shown goes 0.784 -> 0.852, at the cost of showing nothing for
  26% of passages. For a reader, that is the right direction.
- **Forecaster loses slightly**: removing 26% of the sentiment signal and imputing zeros makes every text diff a bit
  more negative (direction T+5 `rank=counts`: -0.0160 -> -0.0208; absmove T+5 `rank=counts`: +0.0009 -> -0.0024), and
  walk-forward validation now picks text in 1 of 6 direction cells instead of 2. The validation-selected edge is
  unchanged at +0.0027 (direction) and +0.0032 (absmove).

No conclusion moves: there was no signal to lose. Abstention is kept on because the feature now carries information
instead of being a constant, and because the evidence layer is this subsystem's actual deliverable. Reverting is one
command (`abstain --min-acc 0 --apply`). `eval` scores are untouched - `evaluate.metrics` takes `model.predict` directly
and never reads the `uncertain` flag, so macro-F1 0.626 still stands at full coverage.

## Live monitoring: sentiment vs the SSI price board (2026-10-26)
`nlp_module -v quotes --symbols <universe>` reads the SSI iBoard board in one request, stores it in `market_quote`,
and logs each symbol's live sentiment snapshot next to what the price actually did. The same view is in the UI tab
"Sentiment vs Giá SSI". It is a **monitoring** view - `same_sign` is co-movement, never an accuracy score.

First run, 2026-10-26 15:06 UTC, session C (closed), 15 tickers, market proxy **-1.02%**:

```
CTG  sentiment +0.162 (n72= 1, abstain  0%, weak 100%) | price  30600  -1.13% | vs market -0.11% opposite
MWG  sentiment +0.434 (n72= 1, abstain  0%, weak   0%) | price  74800  -0.93% | vs market +0.09% same sign
STB  sentiment +0.109 (n72= 2, abstain  0%, weak  50%) | price  67600  -1.31% | vs market -0.29% opposite
TCB  sentiment +0.088 (n72= 2, abstain  0%, weak  50%) | price  33800  -1.17% | vs market -0.15% opposite
VCB  sentiment +0.076 (n72= 1, abstain  0%, weak 100%) | price  56100  -1.23% | vs market -0.21% opposite
VHM  sentiment +0.133 (n72= 2, abstain 50%, weak  50%) | price  63900  -0.62% | vs market +0.40% same sign
VIC  sentiment +0.115 (n72= 4, abstain 25%, weak  25%) | price 237000  +0.21% | vs market +1.23% same sign
raw move        comparable 7/15, same sign 1 (14%)
move vs market  comparable 7/15, same sign 3 (43%)
```

Three things this first reading shows, all of them already on file:

- **Only 7 of 15 tickers are comparable at all.** The rest have no news in 72h, or the whole window abstained. That
  is missing data, not neutral sentiment, and the UI says so rather than drawing a zero.
- **Every available sentiment is positive while the whole market is down 1%.** Raw-sign agreement is 1/7 (14%);
  against the equal-weight market proxy it is 3/7 (43%), i.e. a coin flip. This is the market-factor confound from
  the forecast ablation reproduced in a single live session, and the reason the comparison is shown
  market-relative: the raw sign mostly reports the index.
- **All-positive sentiment on a down day is the measured bias, not a surprise**: positive recall is 0.48 on the
  human `eval` stream and negative rests on 4 samples. The model rarely says negative.

`same_sign` is therefore a monitoring tripwire - a sustained, large departure from ~50% would be news - not a metric
to optimise. Repeated readings accumulate in `market_quote` (`quotes` appends one row per symbol per run), so the
UI's "Lịch sử theo dõi" panel grows with each run; schedule it if you want a real series.

## Earnings-change features from HOSE profit explanations (2026-10-27)
`nlp_module/forecast/fundamentals.py`: `FUND = has_fund, np_yoy, fund_age`. `np_yoy` is the signed LNST change vs the same
period last year, parsed from the issuer's own "Giải trình biến động LNST" filing, clipped to [-2, 2]. The feature is
live for 90 days after `published_at + 30 min`, the same backtest rule as articles. Filings are posted at their real
time, so there is no publication-lag assumption.
- Sources considered and rejected: vnstock (the PyPI project is **quarantined**, so it cannot be installed);
  TCBS API (Cloudflare challenge); Vietcap API (needs spoofed browser headers, which breaks source rule 1); CafeF
  BCTC pages (bank statements carry non-bank row labels, and cells shift between columns).
- Coverage: the HOSE API serves listings only from **~2025-04**. Backfill gave 81 filings for tracked tickers,
  2025-05-05 .. 2026-10-22. 48 state an LNST %; the rest give only amounts or narrative and are dropped. Filings exist
  only for >= 10% swings. Of the 10 price tickers, VCB and STB have no parsed filing; STB files amounts only.
  `has_fund` share is 0.109 of all forecaster rows and 0.461 of rows from 2025-09-17.

| target | h | AUC price | price+fund | diff [95% CI] | price+text+fund |
|---|---|---|---|---|---|
| direction | 2 | 0.519 | 0.527 | +0.008 [-0.023, +0.042] | 0.523 |
| direction | 5 | 0.546 | 0.549 | +0.003 [-0.022, +0.028] | 0.542 |
| excess | 2 | 0.532 | 0.523 | -0.009 [-0.033, +0.015] | 0.515 |
| excess | 5 | 0.533 | 0.543 | +0.010 [-0.017, +0.036] | 0.537 |
| absmove | 2 | 0.573 | 0.563 | -0.009 [-0.029, +0.009] | 0.561 |
| absmove | 5 | 0.581 | 0.475 | **-0.107** [-0.130, -0.083] | 0.475 |

(`trial_forecast --fund`, baseline NLP model, test from 2025-09-17.) Walk-forward (`--skip-nlp --fund`): in folds A
and B, price+fund equals price-only, because the training windows contain no filings. In fold C, validation never
picks price+fund (val 0.512 / 0.510 vs price 0.518 / 0.518 at h=2/5).

**Reading:** there is no measurable gain. Direction/excess point estimates sit around 0, and every CI crosses 0. On
absmove h=5 the result is significantly *worse* (AUC below 0.5): the model learns `fund_age`/`has_fund` from only ~4
months of training rows that have filings, and that does not transfer. The fault is the data window, not the feature
definition. A fair test needs history before 2025, which means a different source for the timestamps (the HOSE API
does not serve it) or a user-supplied BCTC file. Until then, keep `--fund` off in any forecaster that ships.

**Parser fix, rerun 2026-10-29.** The table above uses the fixed parser. `python -m nlp_module.forecast.fundamentals --check`
compares the parsed full-year % with audited consolidated statements from `vnfinancialdata` (public HF dataset, annual
only, 2009-2025). The old parser took the *first* LNST mention. Many filings state the separate (parent-only) figure
first, so FY2025 matched only 2 of 10: PLX +13.8% vs -4% audited (wrong sign), VRE +928% vs +57%. `profit_change` now
prefers the consolidated mention: 7 of 10 within 0.5 pp. The misses are HPG and POW (no consolidated % in the text) and
STB (amounts only). The fix changed 21 of 84 filings' values; the 4 sign flips (PLX, POW, VIC, HSG half-year/quarter)
were read by hand and the new value is the consolidated one in each case. Pre-fix table: direction +0.003/+0.018,
excess +0.006/+0.009, absmove -0.038/-0.061 (h=2/5). The conclusion is unchanged: no gain; keep `--fund` off.

## Annual audited-statement features (BCTC, 2026-10-29)
`fundamentals.BCTC = has_bctc, bc_np_yoy, bc_roe, bc_lev, bc_size` (after-tax-profit growth vs the prior year, ROE,
1 - equity/assets, log total assets), from `data/bctc.parquet` (`vnfinancialdata`, public HF dataset, annual only,
FY2019-2025). The dataset has no publication dates, so fiscal year Y is used from **April 1 of Y+1** (audited
statements are due 90 days after Dec 31), and is dropped after ~13 months. Unlike the HOSE filings, this covers the
whole training window. SSI is not in the dataset (imputed: has_bctc 0, bc_* 0).

Direction/excess forecaster (`trial_forecast --bctc`, test from 2025-09-17; diff = AUC(price+bctc) - AUC(price)):

| target | h | AUC price | price+bctc | diff [95% CI] | price+text+bctc |
|---|---|---|---|---|---|
| direction | 2 | 0.519 | 0.528 | +0.010 [-0.006, +0.025] | 0.520 |
| direction | 5 | 0.546 | 0.563 | +0.016 [-0.000, +0.032] | 0.545 |
| excess | 2 | 0.532 | 0.503 | **-0.030** [-0.047, -0.012] | 0.493 |
| excess | 5 | 0.533 | 0.509 | **-0.024** [-0.045, -0.001] | 0.496 |
| absmove | 2 | 0.573 | 0.573 | -0.000 [-0.017, +0.016] | 0.565 |
| absmove | 5 | 0.581 | 0.585 | +0.004 [-0.016, +0.023] | 0.583 |

Walk-forward (`--skip-nlp --bctc`), test AUC price vs price+bctc: direction 0.513/0.511, 0.539/0.553, 0.502/0.507,
0.487/0.503, 0.524/0.537, 0.585/0.571 (A2, A5, B2, B5, C2, C5); validation picked bctc in A2 and C5, and both picks
scored below price on test. absmove 0.528/0.536, 0.572/0.587, 0.575/0.593, 0.607/0.596, 0.533/0.532, 0.504/0.541.
**Reading:** no usable signal for direction, and excess is significantly *worse*. Beating the market is a
cross-sectional target, and with 10 tickers near-constant per-ticker features let the model memorise tickers. Magnitude
(absmove) leans positive in walk-forward (4 of 6 higher), which is the known size/leverage-volatility link. It is used
only in volcast (next section). Text features add nothing on top of BCTC in any row.

## Volatility forecast (volcast-v2 with BCTC, 2026-10-29; v1 price-only 2026-10-28)
`nlp_module/forecast/volcast.py` predicts **volatility magnitude**, not direction. Target `absmove` is whether |close t→t+h return|
beats the cross-sectional median of the 15 tickers in `data/stock/`, for h = 2 and h = 5 sessions. Features:
price/volume (ret1, ret5, ret20, vol10, vol20, ma20, range, vz, mret1, mret5) plus `fundamentals.BCTC` since v2.
Text and HOSE earnings-filing features are not used (no gain, sections above).

The forecast is confidence-gated. Per fold, the threshold is the widest coverage whose validation accuracy is >= 0.65
with >= 50 calls. The model (logit, or gbm = HistGradientBoosting) is chosen on validation. Evaluation is walk-forward over folds A, B, C
(`walk_forward.FOLDS`), and each test window is scored once. The baseline `rule_acc` uses no model: it calls the highest
(or lowest) cross-sectional vol10 rank (centred ordinal rank, so top and bottom are symmetric), with the same number of
calls but scored on its own rows. Because the threshold is the widest coverage at which validation only just reaches 0.65,
test accuracy is expected to land below 0.65 on average. Numbers below are from `data/reports/volcast_report.json` (v2, 2026-10-29);
v1's are in `data/reports/volcast_report-v1.json`, and its bundle is kept as `data/models/volcast-v1.joblib`.

| fold | h | model | threshold | val_cov v1 → v2 | test_acc v1 → v2 | test_cov v1 → v2 | test_n v2 | CI 95% v2 | rule_acc v2 | rule CI 95% v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 2 | logit | 0.116 | 3.7% → 25.8% | 0.623 → 0.614 | 3.9% → 32.7% | 637 | [0.575, 0.651] | 0.573 | [0.534, 0.611] |
| B | 2 | gbm | 0.093 | 26.1% → 35.1% | 0.663 → 0.676 | 15.8% → 32.0% | 624 | [0.639, 0.712] | 0.644 | [0.606, 0.681] |
| C | 2 | gbm | 0.057 | 44.2% → 48.4% | 0.565 → **0.625** | 51.2% → 44.7% | 523 | [0.583, 0.666] | 0.532 | [0.489, 0.574] |
| A | 5 | gbm | 0.176 | 15.0% → 19.6% | 0.661 → 0.645 | 23.9% → 17.0% | 332 | [0.592, 0.694] | 0.648 | [0.595, 0.697] |
| B | 5 | logit | 0.111 | 23.8% → 32.3% | 0.733 → 0.686 | 8.5% → 36.5% | 711 | [0.651, 0.719] | 0.662 | [0.627, 0.696] |
| C | 5 | gbm | 0.082 | 55.6% → 44.2% | 0.555 → **0.604** | 63.8% → 51.0% | 573 | [0.563, 0.643] | 0.546 | [0.505, 0.587] |

**Why v2.** Selection is on validation: at the same 0.65 validation accuracy, v2 covers more rows in 5 of 6 fold × h
(C5 is the exception). On test, v2 trades accuracy for coverage in A and B (B5: 0.733 on 8.5% → 0.686 on 36.5%, i.e.
121 → 488 correct calls). In the live fold C it is better on both horizons. **Fold C at h=2 is the first row where
the model's CI clears the vol-rank rule's** (pre-purge; see v3 trial) (0.625 [0.583, 0.666] vs 0.532 [0.489, 0.574]). At h=5 the CIs still
touch (0.604 [0.563, 0.643] vs 0.546 [0.505, 0.587]). Model minus rule: +0.041, +0.032, +0.093 at h=2 and -0.003,
+0.024, +0.058 at h=5 (A, B, C). The rule still matches the model in A5. Caveats: (1) the BCTC experiments were read on
test before this choice, although the stated criterion is validation coverage; (2) with 15 tickers, BCTC values are
nearly per-ticker constants, so part of the gain may be "this ticker is usually volatile", which is ex-ante knowledge
but not news; (3) fold C still lands below 0.65 on test.

**v3 trial (2026-10-29, not shipped).** v3 added `vol_ratio` (vol10/vol20, 1.0 for a flat ticker) and per-day
cross-sectional ranks of vol10, vol20, range and |ret1|, picked the model by validation coverage (then accuracy) and
raised the validation threshold to >= 0.665. It was compared once against `v2fixed` = v2 features and selection with
the two bug fixes of 2026-10-29 (validation purged of the h sessions before test; live ret1 taken from the exchange ref
price across dividend seams). Files: `data/reports/volcast_eval_v2fixed.txt`, `data/reports/volcast_eval_v3.txt`.

| fold | h | model v2fixed → v3 | val_cov | test_acc | test_cov | test_n | CI 95% v2fixed | CI 95% v3 | rule_acc |
|---|---|---|---|---|---|---|---|---|---|
| A | 2 | gbm → logit | 7.3% → 16.4% | 0.655 → 0.657 | 7.1% → 16.9% | 139 → 329 | [0.572, 0.729] | [0.604, 0.706] | 0.568 → 0.590 |
| B | 2 | logit → logit | 27.7% → 33.2% | 0.659 → 0.670 | 25.2% → 39.1% | 492 → 763 | [0.616, 0.699] | [0.636, 0.702] | 0.638 → 0.640 |
| C | 2 | logit → gbm | 49.6% → 50.5% | 0.569 → 0.602 | 55.2% → 51.8% | 645 → 606 | [0.530, 0.607] | [0.563, 0.641] | 0.553 → 0.546 |
| A | 5 | logit → gbm | 25.1% → 25.9% | 0.690 → 0.648 | 30.8% → 17.6% | 600 → 344 | [0.652, 0.726] | [0.596, 0.697] | 0.618 → 0.640 |
| B | 5 | gbm → logit | 33.6% → 42.3% | 0.693 → 0.675 | 36.3% → 50.6% | 707 → 986 | [0.658, 0.726] | [0.646, 0.704] | 0.662 → 0.642 |
| C | 5 | gbm → logit | 48.5% → 68.5% | 0.597 → 0.591 | 54.8% → 73.6% | 616 → 827 | [0.558, 0.635] | [0.557, 0.624] | 0.552 → 0.545 |

Decision rule, fixed before the run: ship v3 only if (a) mean test_acc over the 6 rows beats v2fixed, (b) fold C
test_acc >= v2fixed at both h, and (c) fold C test_n >= 100 at both h (a row with no model counts as a loss).
(a) **fail**: 0.6405 vs 0.6438. (b) **fail**: C2 0.602 >= 0.569 passes, C5 0.591 < 0.597 fails. (c) pass: 606 and 827.
**Decision: not shipped.** The v3 code was removed; the two bug fixes stay. v3 mostly widened coverage (test_n up in
4 of 6 rows; C2 and A5 went down). test_acc is up at h=2 in all folds (A2, B2, C2) and down at h=5 in all folds
(A5, B5, C5). Live still runs the v2 bundle (`volcast-v2`, thresholds picked before the
purge fix, not retrained). Note that with the purge, v2's fold C h=2 CI no longer clears the rule's
(0.569 [0.530, 0.607] vs 0.553 [0.515, 0.591]); the "first row where the CI clears" claim above is pre-purge.

The purge also changed v2's own selection: comparing the live report (`data/reports/volcast_report.json`, pre-purge) with
`v2fixed`, the logit/gbm pick flipped in 5 of 6 rows (A2, B2, C2, A5, B5). Fold C h=2 now picks logit @0.062 (test
0.569, n=645) vs the live bundle's gbm @0.057 (pre-purge test 0.625, n=523); the C5 threshold is 0.072 vs live 0.082.
Rerunning `python -m nlp_module.forecast.volcast train` will therefore change the live model. Five flips in six rows put the
logit/gbm pick at noise level.

**v2.1 fixes (2026-10-29).** Two bugs, no feature or gate change:
1. `walk_forward._split` sorted by date with an unstable sort, so adding rows anywhere (e.g. new live days) reordered
   same-day training rows and flipped the logit/gbm pick in folds that do not contain those days (B5 went gbm -> logit
   when the 2026-10-29 bar arrived). It now sorts by (date, symbol), stably; folds A and B are identical with and
   without the live rows. This is the "five flips in six rows" noted above: the pick is decided at the 4th decimal.
2. Live `ret1` is always `close / ref - 1` (SSI ref = previous session's close, adjusted for corporate actions), not
   only beyond the 7.2% band. `ret5`, `ret20`, `ma20` and the labels are computed on a price index chained from `ret1`,
   so they no longer span a raw ex-date jump (TCB 2026-10-29: ref 22.35 vs previous close 33.55). History rows are
   unchanged (the index is the close rescaled).

v2.1 eval (`data/reports/volcast_eval_v2_1.txt`, fold C test now includes live 2026-10-26..29): test_acc A2 0.638 (n 94),
B2 0.659, C2 0.569, A5 0.690, B5 0.693, C5 0.592 (n 549); mean 0.6402, mean (model - vol-rank rule) +0.0302.

**Wilson gate trial (pre-registered 2026-10-29, before the run).** Threshold = widest validation coverage whose Wilson
95% *lower bound* is >= 0.65 (>= 50 calls), instead of the point accuracy; everything else as v2.1. Ship only if
(a) mean over the 6 rows of (test_acc - rule_acc) > +0.0302 (a row with no calls counts 0), because narrowing alone
lifts accuracy for the rule too; (b) fold C test_acc >= v2.1 (0.569 at h=2, 0.592 at h=5); (c) fold C test_n >= 100
at both h. Caveat: these test windows have been read several times before (v1, v2, v3, v2fixed), so a pass is weaker
evidence than a first look. A hard coverage cap (e.g. 30%) was not tried: its value would come from fold C's test.

Result (`data/reports/volcast_eval_wilson.txt`, one run):

| fold | h | model | val_acc | test_acc v2.1 → Wilson | test_n v2.1 → Wilson | CI 95% Wilson | rule_acc | model - rule |
|---|---|---|---|---|---|---|---|---|
| A | 2 | logit | 0.710 | 0.638 → 0.636 | 94 → 415 | [0.589, 0.681] | 0.564 | +0.072 |
| B | 2 | gbm | 0.729 | 0.659 → 0.647 | 492 → 150 | [0.567, 0.719] | 0.627 | +0.020 |
| C | 2 | gbm | 0.687 | 0.569 → 0.626 | 656 → 470 | [0.581, 0.668] | 0.543 | +0.083 |
| A | 5 | logit | 0.707 | 0.690 → 0.720 | 600 → 421 | [0.675, 0.760] | 0.625 | +0.095 |
| B | 5 | gbm | 0.701 | 0.693 → 0.696 | 869 → 672 | [0.661, 0.730] | 0.668 | +0.028 |
| C | 5 | gbm | 0.692 | 0.592 → 0.620 | 549 → 229 | [0.556, 0.680] | 0.590 | +0.030 |

(a) **pass**: +0.0549 > +0.0302. (b) **pass**: C2 0.626 >= 0.569, C5 0.620 >= 0.592. (c) **pass**: 470 and 229.
Mean test_acc 0.6574 (v2.1 0.6402). **Shipped as `volcast-v2.1`** (bundle retrained 2026-10-29 on v2 features with all
fixes; previous bundle and report kept as `data/models/volcast-v2-prepurge.joblib`, `data/reports/volcast_report-v2-prepurge.json`).
Fold C still lands below 0.65 on test (0.626, 0.620), although both CIs now include 0.65; the gate is better
calibrated, not solved. B2 lost (fewer calls, lower accuracy).

**Limitations**
- Live bars come from SSI, in VND /1000 and **not dividend-adjusted**. Since v2.1, live ret1 is taken from the SSI ref
  price and level features and labels use a price index chained from ret1, so ex-dates no longer leak a jump.
- Sessions missing from both sources (2026-10-22/23 at the seam; 2026-10-28, machine off from 10:00) are skipped, not
  filled: rolling windows then hold fewer calendar days, and the skipped sessions' returns are absent from
  ret5/ret20/ma20. No allowed source serves past daily bars; keeping the machine on through 15:15 avoids new gaps.
- Prices are read from `Pipeline_V1.2/stock` (`volcast.STOCK_DIR`), the single canonical source; `data/stock` is no longer read. The fitted symbol set is `volcast.SYMBOLS` (15 tickers) unless `--symbols all` is passed.
- SSI `volume` is normal-matched volume and may exclude put-through trades. This biases `vz` after the seam.
- Text and HOSE earnings-filing features are not used (no gain). BCTC values are annual and lag up to ~15 months.
- `data/bctc.parquet` must exist (README); train and predict fail without it rather than silently impute.
- The v1 → v2 table and `data/reports/volcast_report.json` were computed before validation was purged of the h sessions
  before the test start (~h×15 rows per fold); the purged v2 numbers are the v2fixed column of the v3 trial table.
  The walk_forward text/rank sections of the report also use `walk_forward._split` and are pre-purge and pre-sort-fix
  (v2.1 fix 1), so their val picks can differ at the 4th decimal if rerun.

**How to run**
```
python -m nlp_module.forecast.volcast train     # writes data/models/volcast.joblib + data/reports/volcast_report.json
python -m nlp_module.forecast.volcast predict   # writes vol_forecast; cron runs it on weekdays 15:00-16:00 via scripts/live_update.sh
```

## Baseline v2 (2026-10-29): not shipped, v1 stays live

**Selection on dev validation only** (`cli select-baseline`, `data/reports/baseline_v2_selection.txt`): train first 80% of the
1,324 dev labels by `published_at` (1,059), validate on the last 20% (265, from 2025-09-17, 5 negative). Grid: features
v1/v2 x class weights balanced/sqrt x C 0.3/1/3/10, prior fixed at 1.

| Config | val macro-F1 | recall pos / neu / neg |
|---|---|---|
| v1 setup (features v1, balanced, C=1) | 0.650 | 0.62 / 0.86 / 0.60 |
| best features=v2 (sqrt, C=10) | 0.701 | 0.58 / 0.94 / 0.60 |
| **chosen: features v1, sqrt, C=3** | **0.712** | 0.59 / 0.91 / 0.60 |
| chosen + prior (pos, neu, neg) = (1.75, 1, 1) | 0.725 (in-sample: prior fit on this val) | 0.68 / 0.84 / 0.60 |

Negative recall is 3/5 in every row, so the grid only moves pos/neu. Bundles: `data/models/baseline-v2.joblib`
(`baseline-tfidf-lr-v2`, all dev), twin `data/models/baseline-v2-cal.joblib` (first 80%).

**Human `eval` stream, scored once** (`cli eval --stream eval`, `data/reports/baseline_v2_eval.txt`; n = 240: 58 / 178 / 4):

| Model | macro-F1 [95% CI] | F1 pos / neu / neg | recall pos / neu / neg | neg correct |
|---|---|---|---|---|
| baseline-tfidf-lr-v1 | **0.608** [0.443, 0.723] | 0.50 / 0.82 / 0.50 | 0.50 / 0.81 / 0.75 | 3 / 4 |
| baseline-tfidf-lr-v2 | 0.568 [0.421, 0.733] | 0.54 / 0.83 / 0.33 | 0.60 / 0.80 / 0.25 | 1 / 4 |

CI = `evaluate.bootstrap_macro_f1`, 2,000 row resamples, seed 0. v1 recomputed with this code: same point estimate as
the table above; the CI differs in the third decimal from the earlier bootstrap's draws.

**Decision rule, fixed before scoring:** ship v2 only if (a) macro-F1 v2 >= v1, (b) positive recall v2 > v1, and
(c) eval negatives predicted correctly by v2 >= v1.
- (a) 0.568 >= 0.608: **fail**
- (b) 0.603 > 0.500: pass (35 vs 29 of 58; v2 also calls 35 neutrals positive, vs 29)
- (c) 1 >= 3: **fail**

**Decision: keep v1.** The live model (`scripts/live_update.sh`, threshold 0.888) is unchanged, as are corpus results and snapshots. The
macro-F1 drop is the negative column (2 of 4 samples) and sits inside overlapping CIs; the dev-val gain (+0.06) did not
transfer to human labels. No setting was changed after scoring.

**Linker rules, measured on dev before eval** (`evaluate.link_changes`, `data/reports/link_check.txt`; keep a rule only if it
changes more dev `skipped` samples than labeled ones):

| Rule | dev changed: skipped | dev changed: labeled (pos / neu / neg) | Decision | eval changed, recorded only: skipped vs labeled |
|---|---|---|---|---|
| landmark (place name, not the company) | 0 | 3 (2 / 1 / 0) | removed | 1 dropped vs 0 |
| bank fraud notice -> `ambiguous` | 12 | 61 (1 / 58 / 2) | removed | 17 vs 1 (neutral) |

The fraud rule points opposite ways on the two streams: dev's silver labels (Claude) call bank-fraud notices `neutral`,
the human eval labels call them `skip`. The decision used dev only and stands; the disagreement is a labeling-guideline
question, not settled here. The linker is unchanged from before this round.

**Landmark and bank-account mentions (2026-10-30).** `Linker.find` now skips a mention when the 40 characters before it match
`_ACCOUNT_BEFORE` ("tài khoản (mở) (tại)", "chuyển tiền/khoản (vào/qua)"), or match `_PLACE_BEFORE` ("tại/ở" + optional
"biệt thự|căn hộ|dự án|khu đô thị|nhà") and the alias is followed by a capitalised word (Vinhomes Riverside, Ocean Park).
This differs from the landmark rule removed above: that one was a suffix/corporate-context heuristic; this one is a prefix
heuristic. `link_changes` against the previous linker (`data/reports/link_check.txt`): dev dropped 2 (1 skipped, 1 neutral),
demoted 4; eval dropped 5 (all skipped), demoted 8 (all skipped). Existing results were then purged with
`evaluate.drop_stale_links` on `data/nlp.db`: **584 `article_company_result` rows deleted** (98 article/company pairs, up to 6
model versions each; 57 account, 41 place; mostly VHM 35, VCB 19, CTG 10, TCB 9) and 1,613 derived `snapshot_lineage` rows.
`feature_snapshot` rows built from them are stale until snapshots are rebuilt. Known false drops: "tại FPT Telecom",
"tại FPT Shop/Play" (FPT subsidiaries/shops), "tại Masan MEATLife".

**Passage format v1** (`passages.INPUT_FORMAT_VERSION = "v1"`): title-only links now take the article's lead
sentence(s) instead of the title alone (1,763 v1 results had `sentence_ids='[]'`). Existing annotation samples keep the
text they were labeled on. Because v1 was kept and the corpus not re-scored, this applies only to articles processed
from the cutover on, for both v1 and PhoBERT: their history mixes the two passage formats. The linker did not change,
so linking is consistent across the seam.

Cutover (UTC, `article_company_result.processed_at`, read-only query on `data/nlp.db`): the last v0-format rows were
written at 2026-10-29T04:00:45Z (baseline v1) and 04:00:52Z (phobert-phobert-v1); `passages.py` was changed at
05:17Z (12:17 +07); the first v1-format rows were written at 05:30:27Z and 05:30:35Z. No rows exist in between, so
`processed_at <= 2026-10-29T04:00:52Z` is v0 and `>= 2026-10-29T05:30:27Z` is v1. Across the seam: title-only rows
after the cutover are no longer findable via `sentence_ids='[]'` (they carry the lead sentence ids; `difficult=1`
remains), and their entropy / `uncertain` / `uncertain_ratio` are not comparable with earlier rows, because the
0.888 / 0.571 abstain thresholds were chosen on v0 passages.

## News intensity/surprise features (news3, 2026-10-29)
Six features added to `trial_forecast.NEWS3`, measured with `--news3` (full tables: `data/reports/news3_report.txt`). All use
only results available at the 15:00 close (`published_at + BACKTEST_DELAY` <= close t).

| feature | definition | missing -> imputed |
|---|---|---|
| `sentiment_std` | population std of (p_pos - p_neg) over non-uncertain results in the 72h window | None (also when only one non-uncertain result exists) -> 0.0 |
| `hours_since_last_news` | hours since the newest available article (any age) | None (never had news) -> **720**; values > 720 capped at 720 |
| `sentiment_abs_max` | max abs(p_pos - p_neg) over non-uncertain results in the 72h window | None -> 0.0 |
| `sentiment_shock_24h` | abs(`sentiment_mean_24h`) | None -> 0.0 |
| `sentiment_decay` | mean of (p_pos - p_neg) weighted 0.5^(age/24h) over non-uncertain results in the 72h window | None -> 0.0 |
| `news_surprise` | rank(`news_count_24h`) * (1 - rank(`vol10`)) within each date (lots of news on a calm stock); computed on raw counts before `rank_cs` | NaN where vol10 is NaN -> 0.0 |

Imputation: no news is never read as 0 = "just now" (hence 720 for the age); for the others 0.0 is paired with
`has_news` / `uncertain_ratio`, which mark the imputed rows. `sentiment_std`, `hours_since_last_news` were already in
`contract.build_snapshot` but unused by `TEXT`.

**Conclusion rule, fixed before running, per target** (`walk_forward --skip-nlp --news3 --target <t>`): news3 adds value
only if (i) validation picks a set containing news3 in >= 3 of the 6 fold x h cells, and (ii) mean test AUC of the
picked config >= mean test AUC of price-only + 0.005. Single split is reference only. No setting changed after scoring.

Single split (`trial_forecast --news3`, test from 2025-09-17, n_test 2,710 / 2,680; diff = AUC(variant) - AUC(price)):

| target | h | AUC price | price+news3 | diff [95% CI] | price+text+news3 | diff [95% CI] |
|---|---|---|---|---|---|---|
| absmove | 2 | 0.573 | 0.576 | +0.003 [-0.008, +0.012] | 0.575 | +0.002 [-0.009, +0.013] |
| absmove | 5 | 0.581 | 0.580 | -0.002 [-0.014, +0.009] | 0.587 | +0.006 [-0.005, +0.017] |
| excess | 2 | 0.532 | 0.513 | **-0.019** [-0.036, -0.001] | 0.509 | **-0.023** [-0.045, -0.001] |
| excess | 5 | 0.533 | 0.524 | -0.009 [-0.019, +0.003] | 0.513 | **-0.020** [-0.037, -0.002] |

Walk-forward (`--skip-nlp --news3`), test AUC, fold x h; "picked" = set chosen on validation:

| target | fold, h | price | news3 alone | picked set | picked test |
|---|---|---|---|---|---|
| absmove | A, 2 | 0.528 | 0.529 | price | 0.528 |
| absmove | A, 5 | 0.572 | 0.595 | text+news3/counts | 0.594 |
| absmove | B, 2 | 0.575 | 0.571 | text+news3/counts | 0.583 |
| absmove | B, 5 | 0.607 | 0.592 | text+news3/counts | 0.611 |
| absmove | C, 2 | 0.533 | 0.536 | text+news3/all | 0.540 |
| absmove | C, 5 | 0.504 | 0.499 | text+news3/all | 0.512 |
| excess | A, 2 | 0.474 | 0.487 | text/none | 0.484 |
| excess | A, 5 | 0.482 | 0.486 | text/none | 0.506 |
| excess | B, 2 | 0.518 | 0.521 | price | 0.518 |
| excess | B, 5 | 0.531 | 0.515 | price | 0.531 |
| excess | C, 2 | 0.523 | 0.523 | price | 0.523 |
| excess | C, 5 | 0.535 | 0.521 | price | 0.535 |

Criteria:
- **absmove:** (i) news3 in the picked set in 5 of 6 cells: pass. (ii) mean picked test 0.5613 vs price 0.5532 =
  **+0.0082**: pass. => **news3 adds value** by the rule.
- **excess:** (i) news3 picked in 0 of 6 cells: **fail** (picks are text/none twice, price four times). (ii) 0.5162 vs
  0.5105 = +0.0057 passes numerically, but the lift comes from the two text/none picks, not news3. => **no added value.**

**Reading:** the absmove pass is small and should not be over-read. It passes by the rule, but the rule's baseline is
price-only, so the already-known text effect counts toward the pass: best text-only averages 0.5575 (+0.0043 over
price 0.5532). News3 over the best text-only set adds +0.0038 (0.5613 vs 0.5575), under the 0.005 margin; news3 alone
averages 0.5537 (+0.0005 over price). Criterion (i) counted picks of text+news3 sets (the picked set is always
text+news3, never news3 alone), so it did not isolate news3. Single-split absmove diffs are inside their CIs
(all four CIs include 0), the fold C picks gain only +0.007 / +0.008 on n = 760 / 730, and the picked set is the best of
many validation candidates. Treat as a lead to re-check on newer data, not a shipped feature (volcast live is
unchanged; no text feature goes into live from this plan). News3 never helps excess: validation picks news3 in 0 of 6 cells, and the single-split news3 variants are all below
price-only (3 of the 4 significantly: CI excludes 0, price+news3 at h=2 and price+text+news3 at h=2 and h=5).

**Univariate AUC on the fold C test window** (sessions 2026-07-01..2026-10-14, n = 730 with a T+5 label, base rate
0.500, label T+5 `absmove`, phobert-foldC results; raw feature as score, no sign flip). This is a *different window*
from the Magnitude-target table (trial split test, from 2025-09-17: vol10 0.582, news_count_72h 0.558,
sentiment_mean_24h 0.534); the same three features on the fold C window are given for comparison:

| feature | AUC, fold C test, T+5 absmove |
|---|---|
| vol10 | 0.515 |
| news_count_72h | 0.552 |
| sentiment_mean_24h | 0.536 |
| sentiment_std | 0.518 |
| hours_since_last_news | 0.466 (1 - AUC = 0.534: more recent news, larger move) |
| sentiment_abs_max | 0.521 |
| sentiment_shock_24h | 0.538 |
| sentiment_decay | 0.520 |
| news_surprise | 0.507 |

On this window the strongest news3 features (`sentiment_shock_24h` 0.538, `hours_since_last_news` 0.534 inverted) sit
at the level of `sentiment_mean_24h`, below `news_count_72h`; vol10 is weak here (0.515 vs 0.582 on the trial split).

## Text->Return regression (Ridge, 2026-10-29)
A direct text-to-return model, measured per walk-forward fold (`python -m nlp_module.forecast.text_return --target absmove|excess`,
`nlp_module/forecast/text_return.py`; full tables: `data/reports/text_return_report.txt`). Runtime ~10 s per target.

**Design.**
- Input: 11,372 direct articles (`baseline-tfidf-lr-v1`, `relation='direct'`, `is_representative=1`, `published_at`
  not NULL, the 10 priced tickers), read from `data/nlp.db` read-only. Text = `"[{symbol}] {passage_text}"`.
- Session: first trading date whose 08:00 UTC close is at/after `published_at + 30 min`; after-close, weekend and
  holiday articles roll to the next session in the price calendar.
- Ridge label: r = close[t+h]/close[t] - 1 of the article's ticker; `excess` y = r - mean r of that day's tickers,
  `absmove` y = |r| - median |r|. Articles without a t+h price are dropped.
- Ridge: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)` + `Ridge(alpha=10.0)`, fit per fold x h
  on articles with session in [train_start, val_start) AND session position + h < position of the first session >=
  val_start (no label window reaches validation).
- Daily feature: `ridge_24h` = mean Ridge prediction of the ticker's articles with `published_at + 30 min` in
  (cutoff - 24h, cutoff], cutoff = date 08:00 UTC; `has_ridge` 0/1; no article => `ridge_24h` 0.
- Forecaster: `StandardScaler + LogisticRegression(C=1.0)` **fit on the validation window only** (where every Ridge
  prediction is out-of-sample: validation/test articles all have sessions >= val_start), scored once on test. Rows and
  binary labels from `trial_forecast.add_targets`, split by `walk_forward._split` (h-session purge at each boundary).
  Price-only AUCs therefore differ from the walk-forward tables above (those fit on train+val with C picked on val).
- Univariate AUC = raw `ridge_24h` as score on test rows with `has_ridge = 1` (no sign flip).

**Conclusion rule, fixed before running, per target:** Text->Return adds value only if diff = AUC(price+ridge) -
AUC(price) > 0 in >= 2 of 3 folds at **both** h=2 and h=5, **and** the mean diff over the 6 cells > 0.005.

| target | fold | h | n train articles | n test | AUC price | AUC price+ridge | diff | AUC ridge_24h (n) |
|---|---|---|---|---|---|---|---|---|
| absmove | A | 2 | 3,079 | 1,300 | 0.5347 | 0.5161 | -0.0185 | 0.5315 (551) |
| absmove | B | 2 | 5,113 | 1,300 | 0.5749 | 0.5712 | -0.0037 | 0.5646 (684) |
| absmove | C | 2 | 8,115 | 760 | 0.5290 | 0.5450 | +0.0159 | 0.5635 (449) |
| absmove | A | 5 | 3,058 | 1,300 | 0.5802 | 0.5880 | +0.0079 | 0.5604 (551) |
| absmove | B | 5 | 5,096 | 1,300 | 0.6088 | 0.6055 | -0.0033 | 0.5482 (684) |
| absmove | C | 5 | 8,061 | 730 | 0.5001 | 0.5447 | +0.0447 | 0.6116 (429) |
| excess | A | 2 | 3,079 | 1,300 | 0.4444 | 0.4557 | +0.0112 | 0.5289 (551) |
| excess | B | 2 | 5,113 | 1,300 | 0.5264 | 0.5242 | -0.0022 | 0.4834 (684) |
| excess | C | 2 | 8,115 | 760 | 0.5153 | 0.5428 | +0.0275 | 0.5137 (449) |
| excess | A | 5 | 3,058 | 1,300 | 0.4519 | 0.4424 | -0.0095 | 0.5323 (551) |
| excess | B | 5 | 5,096 | 1,300 | 0.5175 | 0.5418 | +0.0243 | 0.4089 (684) |
| excess | C | 5 | 8,061 | 730 | 0.5439 | 0.5400 | -0.0040 | 0.4827 (429) |

**Conclusion (by the rule):**
- **absmove:** diff > 0 in 1/3 folds at h=2 (fail), 2/3 at h=5; mean diff +0.0072 (> 0.005). Fails the per-h count
  at h=2 => **no added value.**
- **excess:** diff > 0 in 2/3 folds at h=2, 1/3 at h=5 (fail); mean diff +0.0079. Fails at h=5 => **no added value.**

**Reading:** for absmove the raw `ridge_24h` score has point estimates of 0.531-0.612 in the 6 cells (news rows only,
n 429-684), with no CI; the lower values are within noise of 0.5. On top of price features in a validation-only
logistic fit it helps in 3 of 6 cells; the mean lift is carried by fold C (+0.016, +0.045, where price-only is weak at
0.529 / 0.500). For excess the univariate AUC straddles 0.5 (0.409-0.532): no stable directional text signal, same as
earlier excess results. Nothing goes into volcast live.

The protocol is low power: the forecaster is fit on about 6 months of validation with a fixed C=1.0, so "no added
value" means "not shown under this protocol", not "shown absent". The 24h feature window (cutoff - 24h, cutoff] also
drops weekend and holiday articles that Ridge training keeps (Monday's window misses Friday-evening and Saturday
articles); that information loss pushes the result toward the null.

Only 10 tickers have prices (adding BID, CTG, HDB, VHM, NVL price files is the user's action; see
`python -m nlp_module.forecast.trial_forecast --build-prices`).

## Fixed-15 baseline (2026-10-29)

`python -m nlp_module.forecast.volcast eval --test-end 2026-10-30` on the default 15 tickers (`volcast.SYMBOLS`) over
the repaired Pipeline_V1.2/stock prices; fold C's test window is frozen to end before 2026-10-30 (`TEST_END`). Full
output: `data/reports/volcast_eval_fixed15.txt`. Test accuracy of the model on its calls vs. the vol10-rank rule on the
same number of rows; a cell with no call (no threshold passes the Wilson gate) counts as 0.

| h | fold | model | test_n | test_acc | rule_acc | diff |
|---|------|-------|--------|----------|----------|------|
| 2 | A | logit | 202 | 0.663 | 0.569 | +0.094 |
| 2 | B | logit | 160 | 0.706 | 0.625 | +0.081 |
| 2 | C | none | 0 | - | - | 0 |
| 5 | A | none | 0 | - | - | 0 |
| 5 | B | none | 0 | - | - | 0 |
| 5 | C | logit | 507 | 0.582 | 0.566 | +0.016 |

**mean(test_acc - rule_acc) over the 6 cells = +0.0318** (0.191 / 6, from the 3-decimal values above). Only 3 of 6 cells
make calls; fold C has a call at h=5 only (0.582, CI95 [0.538, 0.624], so it does not reach 0.65). This is the number
fusion has to beat.


## 20-ticker basket: pre-registered ship rule (written 2026-10-30 10:03:23 +0700, before the eval run)

Rule written before running `volcast eval --symbols all --test-end 2026-10-30`. It is not changed after seeing results
(no tweaks to the rule, thresholds, seeds or eval). "Not shipped" is a valid outcome.

Ship `volcast.SYMBOLS` = the 20 tickers (`VERSION = "volcast-v2.2"`) only if BOTH hold:

- (a) mean(test_acc - rule_acc) over the 6 cells (h in {2,5} x fold A/B/C; a no-call cell counts 0; diffs taken from
  the 3-decimal table values, same method as the baseline) >= the fixed-15 baseline of +0.0318.
- (b) fold C `test_n >= 100` at BOTH h=2 and h=5.

Otherwise keep 15 tickers and `volcast-v2.1`; no train/predict.

### 20-ticker result and decision (2026-10-30)

`python -m nlp_module.forecast.volcast eval --symbols all --test-end 2026-10-30`; full output
`data/reports/volcast_eval_u20.txt`. The eval prints no ticker list; `load_history(symbols=None)` on the same
directory returns exactly the 20 tickers (ACB BID CTG FPT GAS HDB HPG MBB MSN MWG NVL SSI STB TCB VCB VHM VIC VNM VPB VRE).

| h | fold | model | test_n | test_acc | rule_acc | diff |
|---|------|-------|--------|----------|----------|------|
| 2 | A | none | 0 | - | - | 0 |
| 2 | B | none | 0 | - | - | 0 |
| 2 | C | gbm | 133 | 0.609 | 0.556 | +0.053 |
| 5 | A | none | 0 | - | - | 0 |
| 5 | B | none | 0 | - | - | 0 |
| 5 | C | logit | 142 | 0.634 | 0.563 | +0.071 |

- (a) mean(test_acc - rule_acc) = 0.124 / 6 = **+0.0207** < baseline +0.0318: **fails**.
- (b) fold C test_n = 133 (h=2) and 142 (h=5), both >= 100: passes.

**Decision: not shipped. Kept 15 tickers (`volcast-v2.1`); no train/predict.** The 20-ticker basket gains calls in fold C
(both horizons) but loses every fold A and B call (the Wilson gate no longer passes), so the mean falls below the
fixed-15 baseline. Rule and eval were not tweaked after seeing the numbers.


## BCTC "ticker identity" check: pre-registered rule (written 2026-10-30 10:45:09+07:00, before the run)

Spec: "Kiểm định BCTC có phải học vẹt mã không" (multimodal-decision-system design, NLP fixes item 4). The 4 BCTC ratios
change once a year per ticker, so with 15 tickers they may act as a ticker-ID code. Rule written before running
`volcast bctc-check`; it is applied mechanically and not changed after seeing numbers (no change to the rule, the
aggregation, n_perm, seed, symbols or test window). Code: `volcast.bctc_check`, commit 5e96f5d (not yet run on real data).

- Run: `python -m nlp_module.forecast.volcast bctc-check --test-end 2026-10-30`, default 15 symbols (`volcast.SYMBOLS`), `n_perm=20`,
  `seed=0` (permutation i uses `default_rng(seed + i)`, the same mapping at h=2 and h=5), run once.
- (a) Permutation: before `add_bctc`, each symbol's whole BCTC series (its own values and `available_at` dates) is moved to
  a random other symbol (a random permutation of the 15 symbols; SSI has no BCTC series, so one symbol gets none). Point-in-time
  stays intact; only the symbol -> series map is shuffled. Everything else (price features, labels, folds, model/threshold
  selection) is the unchanged `walk`.
- (b) Leave-one-ticker-out: per symbol, `walk(hold=symbol)` trains/validates/refits without it and tests only on its rows;
  test_acc pooled over its calls in the 6 cells (hits / calls). Descriptive only, NOT part of the decision rule.
- **Aggregation (chosen before the run):** score = mean of `test_acc` over the same 6 cells as eval (h in {2,5} x fold A/B/C);
  a cell with no call counts as **0.5** (chance for the median-split label). Same function for `real` and for each perm run.
  Why not 0 (used for the *difference* vs the rule in the baseline): there a no-call cell has diff 0 by construction; here a 0
  would make "fewer calls" look like "lower accuracy" and conflate abstention with error. Why not the mean over call cells
  only: the set of cells differs between runs, so real and perm would not be comparable. The number of cells with a call is
  reported for real and for each perm (descriptive).
- **Decision rule:** if mean(perm over the 20 runs) >= real - 0.01, conclude **BCTC only encodes ticker identity** (no evidence
  it carries information beyond a ticker code) and record it. Otherwise BCTC beats the permuted control by more than 0.01.
- Either way **the live volcast does not change** (`VERSION`, `SYMBOLS`, bundle stay; no train/predict); a change would need a
  separate decision.
- Known caveats written in advance: `mret1/mret5` and the label median include the held-out symbol in loto; 20 permutations
  with 6 cells give a coarse null (permutation sd reported, no p-value claimed).

# Direction experiment (spec Bước 2): pre-registered GO/NO-GO

Pre-registered 2026-10-30T10:19:15+07:00, before `python -m ai_core.direction` was run on real data. The rule,
parameters and table layout below are fixed; results are appended under "Results" by `main()` and nothing above
that line is changed after a run.

**Assumption (spec §9, not yet confirmed by the owner):** T+2 settlement and no short selling. Hence LONG-only PnL,
entry at the open of t+1, exit at the close of t+1+h, h ∈ {2, 5} (both ≥ 2). AVOID is recorded, never traded.

## GO rule (per h, about the model selected on val)

GO for a given h when the model selected on val satisfies all three:
1. Mean net return per LONG trade (`r_trade - COST`) `> 0` in at least 2/3 folds.
2. That net `>` the median of random-same-count (same number of trades drawn at random from the test rows, 1000
   draws) in at least 2/3 folds.
3. Pooling the 3 folds' test windows (disjoint), the lower bound of the 95% block-bootstrap CI (20-session blocks,
   Pipeline_V1.2 settings) of the mean net excess return per LONG trade (`r_ex - COST`) `> 0`.

A fold with no selected configuration (fewer than 50 val LONG trades at every (kind, C, m)) or no test trades counts
as a loss. Implemented as `ai_core.direction.go(rows, pooled_ci_lo)`. If both h are NO-GO the direction branch stops
(stop rule 1 of the original plan): skip Step 4, continue at Step 5.

## Fixed parameters

- COST `c = 0.0045`, KAPPA `κ = 0.5`; label `y = sign(r_ex)` outside `τ = max(c, κ·vol20·√h)`, NONE inside.
- Folds A/B/C from `NLP_Module` walk_forward, fold C's test window frozen at TEST_END = 2026-10-30.
- Purge `h + 1` sessions at each boundary (`splits.folds(df, h, purge=h+1)`).
- Price-only features `ai_core.direction.FEATS` (no BCTC, no sentiment/PhoBERT); 20-ticker universe.
- Candidates: E3 logit `C ∈ {0.01, 0.1, 1.0}`, E4 gbm (`volcast.MODELS["gbm"]`), sigmoid-calibrated on train;
  margin `m ∈ {0.00, 0.02, …, 0.50}`; (kind, C, m) selected **on val only** (max mean net per LONG trade, ≥ 50
  trades); refit on train+val; test scored once.
- E0: prior (LONG every test row) and random-same-count median.
- Seed: `rng = np.random.default_rng(0)` created once per h (h = 2 then 5), consumed by folds A, B, C in order; the
  pooled CI uses the same rng after fold C.

Table-only additions (do not enter the GO decision):
- E3 and E4 per fold: the same `run_fold` with the candidate grid restricted to one kind (its own selection of C/m on
  val), each with its own `np.random.default_rng(0)` per (h, kind) so the GO inputs are untouched.
- E1: Pipeline_V1.2 `outputs/results/pooled_embed_k0.43/final_table.csv`, rows `method == model` and
  `headline == True` (one per T = h). Split 70/15/15, dữ liệu trước khi sửa, không cùng fold; reported as given
  (that file has classification metrics, not net returns).

## Results

### Run 2026-10-30T10:21:36+07:00 (20 s)

E0/E3/E4 × fold × h on test. SEL = the (kind, C, m) chosen on val (the GO candidate). E3/E4 = the same machinery restricted to one kind (table only). net/net_ex = mean per LONG trade after COST; ci = 95% block bootstrap of net_ex; rand_median = random-same-count median of net (SEL rows' value is the GO input).

|   h | fold   | model               | kind   |        C |        m |    val_n |    n |      hit |     net |   net_ex |    ci_lo |    ci_hi |   rand_median |   turnover |   max_dd |
|----:|:-------|:--------------------|:-------|---------:|---------:|---------:|-----:|---------:|--------:|---------:|---------:|---------:|--------------:|-----------:|---------:|
|   2 | A      | E0 prior (LONG all) | nan    | nan      | nan      | nan      | 2600 | nan      | -0.0054 | nan      | nan      | nan      |      nan      |   nan      | nan      |
|   2 | A      | E3                  | logit  |   0.0100 |   0.0200 | 134.0000 |   60 |   0.4833 | -0.0039 |  -0.0093 |  -0.0191 |  -0.0034 |       -0.0054 |     0.0897 |   0.1447 |
|   2 | A      | E4                  | gbm    | nan      |   0.0600 |  79.0000 |    9 |   0.5556 |  0.0052 |   0.0044 |  -0.0070 |   0.0152 |       -0.0054 |     0.0231 |   0.0127 |
|   2 | A      | SEL (gbm)           | gbm    | nan      |   0.0600 |  79.0000 |    9 |   0.5556 |  0.0052 |   0.0044 |  -0.0070 |   0.0152 |       -0.0054 |     0.0231 |   0.0127 |
|   2 | B      | E0 prior (LONG all) | nan    | nan      | nan      | nan      | 2600 | nan      | -0.0011 | nan      | nan      | nan      |      nan      |   nan      | nan      |
|   2 | B      | E3                  | logit  |   0.0100 |   0.0600 |  91.0000 |  166 |   0.4880 |  0.0024 |   0.0004 |  -0.0090 |   0.0153 |       -0.0010 |     0.1013 |   0.2228 |
|   2 | B      | E4                  | gbm    | nan      |   0.0000 | 657.0000 |  737 |   0.4871 |  0.0013 |  -0.0038 |  -0.0066 |   0.0009 |       -0.0011 |     0.2245 |   0.2832 |
|   2 | B      | SEL (logit)         | logit  |   0.0100 |   0.0600 |  91.0000 |  166 |   0.4880 |  0.0024 |   0.0004 |  -0.0080 |   0.0130 |       -0.0013 |     0.1013 |   0.2228 |
|   2 | C      | E0 prior (LONG all) | nan    | nan      | nan      | nan      | 1620 | nan      | -0.0070 | nan      | nan      | nan      |      nan      |   nan      | nan      |
|   2 | C      | E3                  | logit  |   1.0000 |   0.0400 | 210.0000 |   35 |   0.6000 |  0.0078 |   0.0004 |  -0.0058 |   0.0126 |       -0.0070 |     0.0815 |   0.0333 |
|   2 | C      | E4                  | gbm    | nan      |   0.0200 | 290.0000 |  204 |   0.3824 | -0.0075 |  -0.0062 |  -0.0094 |  -0.0013 |       -0.0070 |     0.1963 |   0.1859 |
|   2 | C      | SEL (logit)         | logit  |   1.0000 |   0.0400 | 210.0000 |   35 |   0.6000 |  0.0078 |   0.0004 |  -0.0059 |   0.0121 |       -0.0073 |     0.0815 |   0.0333 |
|   5 | A      | E0 prior (LONG all) | nan    | nan      | nan      | nan      | 2600 | nan      | -0.0057 | nan      | nan      | nan      |      nan      |   nan      | nan      |
|   5 | A      | E3                  | logit  |   0.1000 |   0.0200 | 221.0000 |  128 |   0.4688 | -0.0046 |  -0.0086 |  -0.0202 |  -0.0029 |       -0.0055 |     0.0669 |   0.1829 |
|   5 | A      | E4                  | gbm    | nan      |   0.0400 | 126.0000 |  117 |   0.4615 | -0.0023 |   0.0013 |  -0.0045 |   0.0070 |       -0.0055 |     0.0837 |   0.0622 |
|   5 | A      | SEL (gbm)           | gbm    | nan      |   0.0400 | 126.0000 |  117 |   0.4615 | -0.0023 |   0.0013 |  -0.0045 |   0.0070 |       -0.0055 |     0.0837 |   0.0622 |
|   5 | B      | E0 prior (LONG all) | nan    | nan      | nan      | nan      | 2600 | nan      |  0.0051 | nan      | nan      | nan      |      nan      |   nan      | nan      |
|   5 | B      | E3                  | logit  |   1.0000 |   0.1000 |  56.0000 |   46 |   0.4783 |  0.0076 |  -0.0050 |  -0.0500 |   0.0286 |        0.0044 |     0.0423 |   0.1842 |
|   5 | B      | E4                  | gbm    | nan      |   0.0800 |  98.0000 |   17 |   0.4706 |  0.0255 |   0.0131 |  -0.0320 |   0.0835 |        0.0048 |     0.0179 |   0.0583 |
|   5 | B      | SEL (logit)         | logit  |   1.0000 |   0.1000 |  56.0000 |   46 |   0.4783 |  0.0076 |  -0.0050 |  -0.0412 |   0.0300 |        0.0048 |     0.0423 |   0.1842 |
|   5 | C      | E0 prior (LONG all) | nan    | nan      | nan      | nan      | 1560 | nan      | -0.0087 | nan      | nan      | nan      |      nan      |   nan      | nan      |
|   5 | C      | E3                  | logit  |   0.1000 |   0.0400 | 253.0000 |   49 |   0.5510 |  0.0108 |  -0.0184 |  -0.0259 |  -0.0030 |       -0.0086 |     0.0609 |   0.0319 |
|   5 | C      | E4                  | gbm    | nan      |   0.0000 | 556.0000 |  406 |   0.4163 | -0.0088 |  -0.0053 |  -0.0114 |  -0.0006 |       -0.0086 |     0.1036 |   0.1101 |
|   5 | C      | SEL (logit)         | logit  |   0.1000 |   0.0400 | 253.0000 |   49 |   0.5510 |  0.0108 |  -0.0184 |  -0.0263 |  -0.0038 |       -0.0089 |     0.0609 |   0.0319 |

E1 (Pipeline_V1.2 pooled_embed_k0.43, headline rows; split 70/15/15, dữ liệu trước khi sửa, không cùng fold; 15 tickers; classification metrics only, that file has no net-return columns):

|   T |   L | config         |   n_tickers | date_from   | date_to    |   hit_rate_mean |   f1_macro_mean |   f1_ci_lo |   f1_ci_hi |   kappa_mean |   directional_share_mean |
|----:|----:|:---------------|------------:|:------------|:-----------|----------------:|----------------:|-----------:|-----------:|-------------:|-------------------------:|
|   2 |  90 | DLinear/wd0    |          15 | 2025-12-11  | 2026-10-29 |          0.3359 |          0.2810 |     0.2647 |     0.2903 |       0.0183 |                   0.4331 |
|   5 |  90 | NLinear/wd0.01 |          15 | 2025-12-16  | 2026-10-29 |          0.4243 |          0.2532 |     0.2345 |     0.2609 |       0.0306 |                   0.2378 |

GO/NO-GO (pre-registered rule above):

- h=2: (1) net > 0 in 3/3 folds; (2) net > rand_median in 3/3 folds; (3) pooled net_ex 95% CI [-0.00673, 0.01199], lower bound > 0: False -> **NO-GO**
- h=5: (1) net > 0 in 2/3 folds; (2) net > rand_median in 3/3 folds; (3) pooled net_ex 95% CI [-0.01424, 0.00530], lower bound > 0: False -> **NO-GO**

**Decision: NO-GO**

### Run 2026-11-10T08:52:10+07:00 (20 s)

E0/E3/E4 × fold × h on test. SEL = the (kind, C, m) chosen on val (the GO candidate). E3/E4 = the same machinery restricted to one kind (table only). net/net_ex = mean per LONG trade after COST; ci = 95% block bootstrap of net_ex; rand_median = random-same-count median of net (SEL rows' value is the GO input).

|   h | fold   | model               | kind   |        C |        m |    val_n |    n |      hit |     net |   net_ex |    ci_lo |    ci_hi |   rand_median |   turnover |   max_dd |
|----:|:-------|:--------------------|:-------|---------:|---------:|---------:|-----:|---------:|--------:|---------:|---------:|---------:|--------------:|-----------:|---------:|
|   2 | A      | E0 prior (LONG all) | nan    | nan      | nan      | nan      | 2600 | nan      | -0.0054 | nan      | nan      | nan      |      nan      |   nan      | nan      |
|   2 | A      | E3                  | logit  |   0.0100 |   0.0200 | 134.0000 |   60 |   0.4833 | -0.0039 |  -0.0093 |  -0.0191 |  -0.0034 |       -0.0054 |     0.0897 |   0.1447 |
|   2 | A      | E4                  | gbm    | nan      |   0.0600 |  79.0000 |    9 |   0.5556 |  0.0052 |   0.0044 |  -0.0070 |   0.0152 |       -0.0054 |     0.0231 |   0.0127 |
|   2 | A      | SEL (gbm)           | gbm    | nan      |   0.0600 |  79.0000 |    9 |   0.5556 |  0.0052 |   0.0044 |  -0.0070 |   0.0152 |       -0.0054 |     0.0231 |   0.0127 |
|   2 | B      | E0 prior (LONG all) | nan    | nan      | nan      | nan      | 2600 | nan      | -0.0011 | nan      | nan      | nan      |      nan      |   nan      | nan      |
|   2 | B      | E3                  | logit  |   0.0100 |   0.0600 |  91.0000 |  166 |   0.4880 |  0.0024 |   0.0004 |  -0.0090 |   0.0153 |       -0.0010 |     0.1013 |   0.2228 |
|   2 | B      | E4                  | gbm    | nan      |   0.0000 | 657.0000 |  737 |   0.4871 |  0.0013 |  -0.0038 |  -0.0066 |   0.0009 |       -0.0011 |     0.2245 |   0.2832 |
|   2 | B      | SEL (logit)         | logit  |   0.0100 |   0.0600 |  91.0000 |  166 |   0.4880 |  0.0024 |   0.0004 |  -0.0080 |   0.0130 |       -0.0013 |     0.1013 |   0.2228 |
|   2 | C      | E0 prior (LONG all) | nan    | nan      | nan      | nan      | 1680 | nan      | -0.0080 | nan      | nan      | nan      |      nan      |   nan      | nan      |
|   2 | C      | E3                  | logit  |   1.0000 |   0.0400 | 210.0000 |   35 |   0.6000 |  0.0078 |   0.0004 |  -0.0064 |   0.0131 |       -0.0077 |     0.0786 |   0.0333 |
|   2 | C      | E4                  | gbm    | nan      |   0.0200 | 290.0000 |  207 |   0.3768 | -0.0081 |  -0.0062 |  -0.0096 |  -0.0016 |       -0.0080 |     0.1973 |   0.1994 |
|   2 | C      | SEL (logit)         | logit  |   1.0000 |   0.0400 | 210.0000 |   35 |   0.6000 |  0.0078 |   0.0004 |  -0.0064 |   0.0130 |       -0.0084 |     0.0786 |   0.0333 |
|   5 | A      | E0 prior (LONG all) | nan    | nan      | nan      | nan      | 2600 | nan      | -0.0057 | nan      | nan      | nan      |      nan      |   nan      | nan      |
|   5 | A      | E3                  | logit  |   0.1000 |   0.0200 | 221.0000 |  128 |   0.4688 | -0.0046 |  -0.0086 |  -0.0202 |  -0.0029 |       -0.0055 |     0.0669 |   0.1829 |
|   5 | A      | E4                  | gbm    | nan      |   0.0400 | 126.0000 |  117 |   0.4615 | -0.0023 |   0.0013 |  -0.0045 |   0.0070 |       -0.0055 |     0.0837 |   0.0622 |
|   5 | A      | SEL (gbm)           | gbm    | nan      |   0.0400 | 126.0000 |  117 |   0.4615 | -0.0023 |   0.0013 |  -0.0045 |   0.0070 |       -0.0055 |     0.0837 |   0.0622 |
|   5 | B      | E0 prior (LONG all) | nan    | nan      | nan      | nan      | 2600 | nan      |  0.0051 | nan      | nan      | nan      |      nan      |   nan      | nan      |
|   5 | B      | E3                  | logit  |   1.0000 |   0.1000 |  56.0000 |   46 |   0.4783 |  0.0076 |  -0.0050 |  -0.0500 |   0.0286 |        0.0044 |     0.0423 |   0.1842 |
|   5 | B      | E4                  | gbm    | nan      |   0.0800 |  98.0000 |   17 |   0.4706 |  0.0255 |   0.0131 |  -0.0320 |   0.0835 |        0.0048 |     0.0179 |   0.0583 |
|   5 | B      | SEL (logit)         | logit  |   1.0000 |   0.1000 |  56.0000 |   46 |   0.4783 |  0.0076 |  -0.0050 |  -0.0412 |   0.0300 |        0.0048 |     0.0423 |   0.1842 |
|   5 | C      | E0 prior (LONG all) | nan    | nan      | nan      | nan      | 1680 | nan      | -0.0102 | nan      | nan      | nan      |      nan      |   nan      | nan      |
|   5 | C      | E3                  | logit  |   0.1000 |   0.0400 | 253.0000 |   49 |   0.5510 |  0.0108 |  -0.0184 |  -0.0255 |  -0.0039 |       -0.0108 |     0.0585 |   0.0319 |
|   5 | C      | E4                  | gbm    | nan      |   0.0000 | 556.0000 |  420 |   0.4119 | -0.0090 |  -0.0049 |  -0.0105 |   0.0000 |       -0.0102 |     0.1063 |   0.1106 |
|   5 | C      | SEL (logit)         | logit  |   0.1000 |   0.0400 | 253.0000 |   49 |   0.5510 |  0.0108 |  -0.0184 |  -0.0260 |  -0.0018 |       -0.0104 |     0.0585 |   0.0319 |

E1 (Pipeline_V1.2 pooled_embed_k0.43, headline rows; split 70/15/15, dữ liệu trước khi sửa, không cùng fold; 15 tickers; classification metrics only, that file has no net-return columns):

|   T |   L | config         |   n_tickers | date_from   | date_to    |   hit_rate_mean |   f1_macro_mean |   f1_ci_lo |   f1_ci_hi |   kappa_mean |   directional_share_mean |
|----:|----:|:---------------|------------:|:------------|:-----------|----------------:|----------------:|-----------:|-----------:|-------------:|-------------------------:|
|   2 |  90 | DLinear/wd0    |          15 | 2025-12-11  | 2026-10-29 |          0.3359 |          0.2810 |     0.2647 |     0.2903 |       0.0183 |                   0.4331 |
|   5 |  90 | NLinear/wd0.01 |          15 | 2025-12-16  | 2026-10-29 |          0.4243 |          0.2532 |     0.2345 |     0.2609 |       0.0306 |                   0.2378 |

GO/NO-GO (pre-registered rule above):

- h=2: (1) net > 0 in 3/3 folds; (2) net > rand_median in 3/3 folds; (3) pooled net_ex 95% CI [-0.00684, 0.01331], lower bound > 0: False -> **NO-GO**
- h=5: (1) net > 0 in 2/3 folds; (2) net > rand_median in 3/3 folds; (3) pooled net_ex 95% CI [-0.01471, 0.00564], lower bound > 0: False -> **NO-GO**

**Decision: NO-GO**

Note on the 2026-11-10 run (re-run after the numpy 2.2.6 -> 2.3.5 fix, see `tests/test_env_numpy.py`): folds A and B are identical to the 2026-10-30 run, and every selected (kind, C, m) is unchanged. Only fold C differs, because its test rows near TEST_END now have matured labels (h=2: 1620 -> 1680 rows, h=5: 1560 -> 1680). The numpy bug did not affect this experiment, and the decision stays NO-GO.

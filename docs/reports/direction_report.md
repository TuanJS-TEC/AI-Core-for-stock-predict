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

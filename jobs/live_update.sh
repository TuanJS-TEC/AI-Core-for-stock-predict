#!/usr/bin/env bash
# Live loop: new news -> both production models -> SSI board (trading hours only). Meant for cron every 2 min (see the cadence rules below).
# Each step is incremental (collect skips stored URLs, process skips scored versions), so a rerun is cheap.
set -u
cd "$(dirname "$0")/.."   # repo root
export TZ=Asia/Ho_Chi_Minh
# test hooks: LIVE_CLOCK="YYYY-MM-DD HH:MM" fakes the clock, LIVE_DRY=1 only prints each step name (and skips flock)
now() { if [ -n "${LIVE_CLOCK:-}" ]; then date -d "$LIVE_CLOCK" "$@"; else date "$@"; fi; }
if [ -z "${LIVE_DRY:-}" ]; then
    exec 9>data/news/live.lock
    flock -n 9 || { echo "$(now -Is) previous run still going, skip"; exit 0; }
fi
# STOCKSENSE_DATABASE_URL (git-ignored .env, chmod 600): where ai_core.fusion.stocksense pushes the board; unset = skip.
# Read as plain text, never sourced: a '$' in the password would otherwise abort this whole script under set -u.
[ -f .env ] && export STOCKSENSE_DATABASE_URL="$(sed -n 's/^STOCKSENSE_DATABASE_URL=//p' .env)"
CLI=".venv/bin/python -m ai_core.news.cli -v"
dry() { [ -n "${LIVE_DRY:-}" ] && echo "$1"; }   # dry mode: print the step name and return 0, so the real command after || is skipped
hm=$((10#$(now +%H%M)))
# cron runs every 2 min. full = on a 15-min mark (the old cadence): news + quotes + forecast. day = 07:00-22:00: news only.
full=0; [ $((10#$(now +%M) % 15)) -lt 2 ] && full=1
day=0; [ "$hm" -ge 700 ] && [ "$hm" -lt 2200 ] && day=1
[ "$full" = 0 ] && [ "$day" = 0 ] && exit 0
weekday=0; [ "$(now +%u)" -le 5 ] && weekday=1
[ -n "${LIVE_DRY:-}" ] || echo "=== $(now -Is) live update"
dry collect || for s in cafef vietstock hose; do
    $CLI collect "$s" "$(now -d "${LIVE_CLOCK:-now} yesterday" +%F)" "$(now +%F)" || echo "collect $s failed, continuing"
done
# thresholds from README (abstain on dev validation, ~73% coverage at 0.85 accuracy)
dry process || { $CLI process --model baseline:data/news/models/baseline.joblib --abstain-entropy 0.888
$CLI process --model phobert:data/news/models/phobert-v1 --abstain-entropy 0.571; }
# push the newly scored news to StockSenseVN (unset STOCKSENSE_DATABASE_URL = skip)
dry news-push || .venv/bin/python -m ai_core.fusion.stocksense_news || echo "stocksense news push failed, continuing"
[ "$full" = 1 ] || exit 0
if [ "$weekday" = 1 ] && [ "$hm" -ge 900 ] && [ "$hm" -le 1515 ]; then dry quotes || $CLI quotes; fi
# own window, not nested in the quotes one: predict reads the stored post-close bar (live_bars only accepts fetched_at >= 08:00 UTC), so an early run harmlessly re-predicts the previous session
if [ "$weekday" = 1 ] && [ "$hm" -ge 1500 ] && [ "$hm" -le 1600 ]; then
    dry predict || .venv/bin/python -m ai_core.models.volcast.volcast predict || echo "volcast predict failed, continuing"
    dry rangecast || .venv/bin/python -m ai_core.models.rangecast.rangecast predict || echo "rangecast predict failed, continuing"
    dry live || .venv/bin/python -m ai_core.fusion.live || echo "decision write failed, continuing"
    dry forecast-push || .venv/bin/python -m ai_core.fusion.stocksense || echo "stocksense push failed, continuing"
fi

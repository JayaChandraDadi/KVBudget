#!/usr/bin/env bash
# Real-model KV-cache experiment on Apple silicon.
#
#   ./scripts/run_sweep.sh                                  # Qwen2.5-1.5B-Instruct, 8-bit weights
#   MODEL_HF=Qwen/Qwen2.5-0.5B-Instruct ./scripts/run_sweep.sh
#
# 1. per-layer K/V sensitivity at 4K context
# 2. allocations (independent K/V per layer, and the layer-wise pair space)
#    at several KV budgets, plus uniform baselines
# 3. quality at 4K / 16K / 32K (KL to fp16 cache, perplexity change)
# 4. needle-in-a-haystack retrieval at 4K / 16K / 32K
# 5. KV memory, peak memory, prefill and decode speed at 4K..32K
set -euo pipefail

MODEL_HF=${MODEL_HF:-Qwen/Qwen2.5-1.5B-Instruct}
NAME=${NAME:-$(basename "$MODEL_HF")}
OUT=${OUT:-runs/$NAME}
BUDGETS=${BUDGETS:-"3.0 3.5 4.5 6.0"}
QUALITY_LENS=${QUALITY_LENS:-4096,16384,32768}
CONTEXTS=${CONTEXTS:-4096,8192,16384,32768}
RES="$OUT/results.jsonl"
mkdir -p "$OUT/plans"

# weights at 8 bits so the experiment isolates the KV cache and fits a 16 GB Mac
MODEL="$OUT/model8"
[ -d "$MODEL" ] || mlx_lm.convert --hf-path "$MODEL_HF" -q --q-bits 8 --mlx-path "$MODEL"

SENS="$OUT/sensitivity.json"
[ -f "$SENS" ] || kvbudget profile --model "$MODEL" --seq-len 4096 --n-seqs 2 --out "$SENS"

PLANS=()
for b in $BUDGETS; do
  for mode in independent pairs; do
    p="$OUT/plans/$mode-$b.json"
    [ -f "$p" ] || kvbudget allocate --sens "$SENS" --avg-bits "$b" --mode "$mode" --out "$p"
    PLANS+=("$p")
  done
done

UNIFORM=(16,16 8,8 8,4 4,4 4,2 2,2)
COMMON=(--model "$MODEL" --sens "$SENS" --results "$RES" --tag "$NAME")

kvbudget eval  "${COMMON[@]}" --uniform "${UNIFORM[@]}" --plan "${PLANS[@]}" \
  --seq-lens "$QUALITY_LENS" --n-seqs 2
kvbudget bench "${COMMON[@]}" --uniform "${UNIFORM[@]}" --plan "${PLANS[@]}" \
  --contexts "$CONTEXTS" --gen-tokens 64
# needle retrieval is the slowest (one long prefill per trial): fp16, the
# aggressive uniform settings, and the low-budget allocations
LOW=$(echo $BUDGETS | cut -d' ' -f1-2)
NEEDLE_PLANS=()
for b in $LOW; do NEEDLE_PLANS+=("$OUT/plans/independent-$b.json" "$OUT/plans/pairs-$b.json"); done
kvbudget needle "${COMMON[@]}" --uniform 16,16 4,4 4,2 2,2 --plan "${NEEDLE_PLANS[@]}" \
  --contexts 4096,16384,32768 --trials 2

kvbudget plot --results "$RES" --sens "$SENS" --plans "${PLANS[@]}" \
  --out "$OUT/figures" --title "$NAME (8-bit weights)"
echo "done: $RES and $OUT/figures/"

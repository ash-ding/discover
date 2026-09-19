#!/usr/bin/env bash
# Preflight for PUCT inference-only launches.
#
# Exists because three runs were launched from code that predated a capability
# they needed, and each loss only surfaced hours later:
#   * gptoss-erdos-50step started 19h before the per-rollout score commit and
#     dumped 25k rollouts carrying no score/raw_score.
#   * qwen3-8b-erdos-inference-only-repeat2 and qwen3-8b-ac1-inference-only were
#     launched from a checkout two commits behind and lost the same field.
# Reading `git log` by eye is not a control. Checking the file that is about to
# run is. Every assertion below cost a real experiment.
#
#   bash scripts/preflight_puct.sh [EXPERIMENT_NAME] [RAY_TEMP_DIR]
#
# Non-zero exit means do not launch. In a launcher:
#   bash scripts/preflight_puct.sh "$EXPERIMENT_NAME" "$RAY_TEMP_DIR" || exit 1
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
fail=0
say() { printf '   %-34s %s\n' "$1" "$2"; }

echo "== preflight: $(pwd)"
say "branch / HEAD" "$(git branch --show-current 2>/dev/null) / $(git log -1 --format='%h %cd' --date=short 2>/dev/null)"

# 1. The loop must persist per-rollout scores, or the run yields code with no
#    score attached and the per-step distribution is gone for good.
if grep -q '_rec\["raw_score"\]' scripts/vllm_puct_loop.py 2>/dev/null; then
  say "per-rollout score persistence" "present"
else
  say "per-rollout score persistence" "MISSING - rollouts would carry no score"
  fail=1
fi

# 2. The server must be able to advertise a name other than gpt-oss-120b, or
#    every request for the model actually loaded returns 404 - and only after
#    all replicas have finished loading.
if grep -q 'SERVED_NAME' scripts/serve_gpt_oss.sh 2>/dev/null; then
  say "configurable served-model-name" "present"
else
  say "configurable served-model-name" "MISSING - non-gpt-oss models 404"
  fail=1
fi

# 3. Freshness. No network is advisory; being behind is not.
if git fetch origin --quiet 2>/dev/null; then
  behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo '?')
  if [ "$behind" = "0" ]; then say "behind origin/main" "0"
  else say "behind origin/main" "$behind commit(s) - pull first"; fail=1; fi
else
  say "behind origin/main" "could not fetch (advisory)"
fi

# 4. An existing output directory makes the loop resume from its last step
#    instead of starting over, which silently voids a repeat experiment.
EXP=${1:-}
if [ -n "$EXP" ]; then
  if [ -d "checkpoints/gptoss-puct/$EXP" ]; then
    say "experiment name free" "TAKEN - loop would resume, not restart"
    fail=1
  else
    say "experiment name free" "$EXP"
  fi
fi

# 5. Ray builds AF_UNIX socket paths under RAY_TEMP_DIR; those cap at 107 bytes,
#    and a leftover directory carries stale sockets.
RT=${2:-}
if [ -n "$RT" ]; then
  if [ -e "$RT" ]; then say "ray temp dir free" "EXISTS: $RT"; fail=1; else say "ray temp dir free" "$RT"; fi
  n=$(printf '%s' "$RT" | wc -c | tr -d ' ')
  if [ "$n" -gt 60 ]; then say "ray temp dir length" "$n bytes - AF_UNIX caps at 107"; fail=1; else say "ray temp dir length" "$n bytes"; fi
fi

if [ "$fail" -eq 0 ]; then echo "== preflight OK"; else echo "== preflight FAILED - do not launch"; fi
exit $fail

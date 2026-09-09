#!/usr/bin/env bash
# Watchdog for the gpt-oss PUCT run. cron: */10 * * * * $HOME/gptoss_watchdog.sh
#
# Alert-only, no auto-restart: restarting mid-step would need resume handling
# and could double-run against the same archive.
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
LOG=$HOME/gptoss_watchdog.log
STATE=$HOME/gptoss_watchdog.state
STATUS=$HOME/gptoss_status.txt
TS=$(date '+%Y-%m-%d %H:%M:%S')
RUN_DIR=$HOME/code/discover-claude

EXP=$(ls -t "$RUN_DIR"/checkpoints/gptoss-puct/ 2>/dev/null | head -1)
[ -z "$EXP" ] && { echo "[$TS] NO_EXPERIMENT" >> "$LOG"; exit 0; }
D="$RUN_DIR/checkpoints/gptoss-puct/$EXP"

# `<` fails in the shell before wc runs, so redirecting wc's stderr is not
# enough -- guard on the file existing instead.
if [ -f "$D/metrics.jsonl" ]; then STEP=$(wc -l < "$D/metrics.jsonl" | tr -d ' '); else STEP=0; fi
LOGF=$(ls -t "$HOME"/gptoss_run_*.log 2>/dev/null | head -1)
LOG_AGE=$(( ( $(date +%s) - $(stat -c %Y "$LOGF" 2>/dev/null || echo 0) ) / 60 ))
PROC=$(pgrep -fc "python scripts/vllm_puct_loop" 2>/dev/null | head -1); PROC=${PROC:-0}
TMUX_OK=$(tmux ls 2>/dev/null | grep -c gptoss)

# vLLM replica health -- the loop is useless if the fleet is down. Read the
# port list the serve script wrote rather than assuming a topology: TP=2 means
# 4 replicas, TP=1 means 8, and a hardcoded list false-alarms on the other one.
PORTFILE=$HOME/gpt_oss_serve/ports
if [ -f "$PORTFILE" ]; then PORTS=$(cat "$PORTFILE"); else PORTS="8100 8101 8102 8103"; fi
EXPECTED=$(echo "$PORTS" | wc -w | tr -d ' ')
REPLICAS=0
for p in $PORTS; do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://127.0.0.1:$p/health" 2>/dev/null)" = "200" ] \
    && REPLICAS=$((REPLICAS + 1))
done

PREV_STEP=0; PREV_STALL=0
[ -f "$STATE" ] && { PREV_STEP=$(cut -d' ' -f1 "$STATE"); PREV_STALL=$(cut -d' ' -f2 "$STATE"); }
if [ "$STEP" -gt "$PREV_STEP" ]; then STALL=0; else STALL=$((PREV_STALL + 1)); fi
echo "$STEP $STALL" > "$STATE"

BEST=$(python3 -c "
import json,sys
try:
    rows=[json.loads(l) for l in open('$D/metrics.jsonl') if l.strip()]
    print(rows[-1].get('best_ever'))
except Exception: print('n/a')
" 2>/dev/null)

VERDICT=OK
[ "$REPLICAS" -lt "$EXPECTED" ] && VERDICT="REPLICAS_DOWN_${REPLICAS}of${EXPECTED}"
[ "$REPLICAS" -eq 0 ]  && VERDICT="ALL_REPLICAS_DOWN"
[ "$PROC" -eq 0 ]      && VERDICT="LOOP_DEAD"
[ "$TMUX_OK" -eq 0 ]   && VERDICT="TMUX_GONE"
[ "$LOG_AGE" -gt 45 ]  && VERDICT="LOG_STALE_${LOG_AGE}min"
[ "$STALL" -ge 12 ]    && VERDICT="NO_PROGRESS_${STALL}x10min"
[ "$STEP" -ge 50 ]     && VERDICT=COMPLETE

echo "[$TS] $VERDICT exp=$EXP step=$STEP/50 best=$BEST replicas=$REPLICAS/$EXPECTED loop=$PROC log_age=${LOG_AGE}m stall=$STALL" >> "$LOG"
echo "$VERDICT exp=$EXP step=$STEP/50 best=$BEST replicas=$REPLICAS/$EXPECTED updated=$TS" > "$STATUS"
tail -2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"

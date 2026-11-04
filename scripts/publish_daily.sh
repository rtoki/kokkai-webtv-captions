#!/usr/bin/env bash
# scripts/publish_daily.sh
#
# 直近 LOOKBACK_DAYS 日ぶんの議事録を取得し、Cloudflare Pages へ会議ごとに公開する。
#   1. 参議院 (AI 字幕を整形するだけなので数秒) → まとめて公開
#   2. 衆議院 (VibeVoice ASR + LLM 校正、1 会議 1〜4 時間) → 1 会議終わるごとに公開
#
# VibeVoice は 1 日ぶん (衆議院で音声 ~25 時間) に 12〜20 時間かかるので、
# 1 回の実行が翌日の起動時刻をまたぐことがある。launchd は実行中のジョブを
# 二重起動しないので、その日の起動はスキップされる。取りこぼさないよう毎回
# 直近数日を見て、--skip-if-done で未処理のものだけ古い順に処理する。
#
# 使い方:
#   scripts/publish_daily.sh                # 今日を含む直近 3 日
#   scripts/publish_daily.sh 2026-05-23     # 指定日を含む直近 3 日
#   LOOKBACK_DAYS=1 scripts/publish_daily.sh
#
# 依存:
#   - CLOUDFLARE_API_TOKEN env または macOS Keychain (service=kokkai-cf-token)
#   - uv, ffmpeg, npx (wrangler)

set -euo pipefail

cd "$(dirname "$0")/.."

TO="${1:-$(date +%F)}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-3}"
FROM="$(date -j -v-"$((LOOKBACK_DAYS - 1))"d -f %F "$TO" +%F)"
mkdir -p logs

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

KKCAP=(uv run --extra asr-vibevoice --extra llm-mlx kkcap)

# token は env > Keychain。数時間 ASR を回してから失敗しないよう最初に解決する。
if [[ -z "${CLOUDFLARE_API_TOKEN:-}" ]]; then
  if CLOUDFLARE_API_TOKEN=$(security find-generic-password -s kokkai-cf-token -w 2>/dev/null); then
    export CLOUDFLARE_API_TOKEN
  else
    log "ERROR: CLOUDFLARE_API_TOKEN not in env and not in Keychain (service=kokkai-cf-token)"
    exit 2
  fi
fi

# dist/ を作り直して Pages へ (wrangler が content-hash で差分だけ上げる)。
# 失敗しても次の会議の処理は続ける。
publish() {
  log "publish: $1"
  if uv run python scripts/build_index.py \
     && npx --yes wrangler@latest pages deploy dist \
          --project-name kokkai-captions --branch main --commit-dirty=true; then
    log "published → https://kokkai-captions.pages.dev/"
  else
    log "WARN: publish failed ($1); will retry after next meeting"
  fi
}

log "publish_daily start ($FROM .. $TO)"

# --- 1) 参議院 ---------------------------------------------------------------
if ! "${KKCAP[@]}" fetch --from "$FROM" --to "$TO" --house sangiin --skip-if-done --no-open; then
  log "WARN: sangiin fetch exited non-zero"
fi
publish "sangiin"

# --- 2) 衆議院: 1 会議ずつ ASR → 公開 ------------------------------------------
# dry-run 出力: "shugiin<TAB><deli_id><TAB><YYYY-MM-DD 会議名>"。日付順 (古い順) に処理。
targets=$("${KKCAP[@]}" fetch --from "$FROM" --to "$TO" --house shugiin --dry-run 2>/dev/null \
          | awk -F'\t' '$1=="shugiin"' | sort -t$'\t' -k3,3)
n=$(printf '%s\n' "$targets" | grep -c . || true)
log "shugiin: $n meetings in range"

i=0
while IFS=$'\t' read -r _house id label; do
  [[ -z "${id:-}" ]] && continue
  i=$((i + 1))
  log "shugiin [$i/$n] $id $label"
  # --asr-backend 等は kkcap fetch が衆議院側にそのまま転送する (値は = 形式で)。
  # 取込済なら --skip-if-done で即スキップし、公開もしない。
  out=$("${KKCAP[@]}" fetch "$id" --skip-if-done --no-open \
          --asr-backend=vibevoice --asr-trim-edges --llm-correct 2>&1 | tee /dev/stderr) || {
    log "WARN: shugiin $id failed; continuing"
    continue
  }
  if grep -q "既に取り込み済み (skip)" <<<"$out"; then
    continue
  fi
  publish "shugiin $id $label"
done <<<"$targets"

log "publish_daily done"
osascript -e 'display notification "kokkai-captions を更新しました" with title "kkcap publish_daily"' 2>/dev/null || true

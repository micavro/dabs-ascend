#!/usr/bin/env bash
# CPU-side transfer only. No re-archive, no model requests, no key upload.
set -euo pipefail
umask 077

CPU_DIR="${WF_CPU_DIR:-/home/wmy/paper/sci}"
KEY="${WF_SSH_KEY:-/home/wmy/ustc87}"
GPU='wqx@10.18.2.128'
SNAP="${1:-/home/wmy/sci/spra_paper_experiments_github/wf_archives/CHECKPOINT_20261012T052727Z_52251}"
[[ "$SNAP" =~ ^/home/wmy/sci/spra_paper_experiments_github/wf_archives/[A-Za-z0-9_-]+$ ]] || {
  echo 'STOP: unsupported checkpoint path' >&2; exit 2;
}
cd "$CPU_DIR"
test -r "$KEY" || { echo "STOP: SSH key not readable: $KEY" >&2; exit 2; }
DEST=$(mktemp -d "$PWD/WF_review_recovered_XXXXXX")
CTL=$(mktemp -d /tmp/wf39-ssh-XXXXXX)
SOCK="$CTL/master"
OPTS=(-p 8822 -i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
cleanup() {
  ssh "${OPTS[@]}" -S "$SOCK" -O exit "$GPU" >/dev/null 2>&1 || true
  rm -f "$SOCK"
  rmdir "$CTL" 2>/dev/null || true
}
trap cleanup EXIT
printf 'CPU_DEST=%s\nGPU_CHECKPOINT=%s\n' "$DEST" "$SNAP"
# Authenticate once; subsequent operations reuse this connection. If it fails,
# stop immediately, rather than copying only a checksum on a later connection.
ssh "${OPTS[@]}" -M -S "$SOCK" -o ControlPersist=60 -fN "$GPU"
ssh "${OPTS[@]}" -S "$SOCK" -O check "$GPU"
ssh "${OPTS[@]}" -S "$SOCK" -o BatchMode=yes -o ControlMaster=no "$GPU" \
  "set -eu; cd '$SNAP'; test -s review.zip; test -s review.zip.sha256; sha256sum -c review.zip.sha256; ls -lh review.zip"
for NAME in review.zip review.zip.sha256; do
  if ! scp -P 8822 -i "$KEY" -o IdentitiesOnly=yes \
      -o ControlPath="$SOCK" -o ControlMaster=no -o BatchMode=yes \
      -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      "$GPU:$SNAP/$NAME" "$DEST/$NAME"; then
    echo "STOP: $NAME transfer failed; partial files retained in $DEST" >&2
    exit 3
  fi
done
cd "$DEST"
test -s review.zip
test -s review.zip.sha256
[[ $(wc -l < review.zip.sha256) -eq 1 ]] || { echo 'STOP: invalid checksum file' >&2; exit 4; }
grep -Eq '^[0-9a-f]{64} [ *]review\.zip$' review.zip.sha256 || { echo 'STOP: unexpected checksum entry' >&2; exit 4; }
sha256sum -c review.zip.sha256
printf '\nTRANSFER_VERIFIED=%s/review.zip\n' "$DEST"
ls -lh review.zip review.zip.sha256

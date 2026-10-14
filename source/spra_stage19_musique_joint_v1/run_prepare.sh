#!/usr/bin/env bash
set -euo pipefail
echo 'STOP: Stage19不再使用旧合成数据run_prepare入口。先运行run_smoke.sh，再run_profile.sh。' >&2
exit 2

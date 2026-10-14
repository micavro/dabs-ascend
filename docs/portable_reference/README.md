# Workflow-aware DAG scheduling — source handoff

Start with [README_zh.md](README_zh.md). This source-only kit retains the original custom system, DAG scheduler, fixed-budget native execution, baselines, experiment harness and CPU tests. It contains no historical experimental results or model weights. The original native backend is a pinned interface reference; a non-NVIDIA backend still requires actual adaptation and target calibration.

Run `python -B port.py verify`, `python -B port.py selftest`, then read `docs/PORTING.md`. The optional `export-on-gpu` command enriches this source kit with the exact existing input dataset, never historical results.

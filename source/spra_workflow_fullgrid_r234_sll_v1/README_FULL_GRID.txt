DABS R=2/3/4 Balanced-64 complete native scheduling source experiment

NEW frozen experiment suite: dabs-r234-balanced-64
  Active replicas: 2, 3, 4; 4 existing native services verified by gate.
  Arrival rates: lambda = 4, 8, 12, 16 workflows per second.
  64 MuSiQue-derived fixed-token workflows; 216 DAG requests per condition.
  3584 input tokens per node; 16/32/48/64/128 output tokens by DAG shape.
  Exactly 13184 generated output tokens per condition, identical per method.
  1s ON at peak 2lambda, 1s OFF, fixed original waveform.
  4096 native batch cap, same unchanged vLLM prefix caching/continuous batching.
  Six original DAG structural shapes and balanced prefix popularity.
  Methods: RR; SLL (Sampled Least-Loaded, k=2, old P2C exact routing);
           Ayo (controller-level topology proxy, not official full Ayo);
           Ours (unchanged DAG-H V5 decision logic).
  Reverse order in round 2, equal seeds; 3 x 4 x 4 x 2 = 96 native GPU conditions.
  Source/routing/model modifications: NONE except SLL display alias;
  existing Ayo proxy and Ours DAG-H algorithm functions preserved.

This combined source bundle includes the historical original native executor,
MuSiQue typed workflow driver, fixed reference profile/public inputs,
provenance, and the new independent experiment implementation.  It does not
include Python packages, GPU drivers, model weights, or completed GPU results.

Native model_input skipped_query_tokens is real observed prefix reuse; the
intra/cross-workflow categories record eligible prior completed input-prefix
producers, not actual GPU KV block writers. Ambiguity/unattributed is retained.
Workflow throughput refers to this finite arrival cohort, not independently
established sustainable system capacity. Ayo is not a complete Ayo reproduction.

Commands (from extracted project root, after adapting runtime for your host):
  python tools/dabs_r234_preflight.py
  bash source/spra_workflow_fullgrid_r234_sll_v1/run_checks.sh
  python tools/run_dabs_r234.py gate --runtime "$RUNTIME"
  python -u tools/run_dabs_r234.py run \
    --suite dabs-r234-balanced-64 --runtime "$RUNTIME" \
    --out outputs/dabs_r234_balanced_001 --max-conditions 4 --confirm-gpu-use
  python -u tools/run_dabs_r234.py run \
    --suite dabs-r234-balanced-64 --runtime "$RUNTIME" \
    --out outputs/dabs_r234_balanced_001 --resume --confirm-gpu-use
  python tools/run_dabs_r234.py report --out outputs/dabs_r234_balanced_001
  python tools/plot_dabs_r234.py --out outputs/dabs_r234_balanced_001
  python tools/run_dabs_r234.py export --out outputs/dabs_r234_balanced_001

Reports: analysis/metric_grid.csv, latency_decomposition_by_condition.csv,
kv_by_condition.csv, kv_by_replica.csv, kv_by_node.csv, paired_ours_gains.csv,
paired_grid_summary.csv, eight source-grounded PDF figures.

Protocol and raw evidence remain frozen. Derived analysis CSVs are rebuilt
atomically (including after pause/resume) without touching accepted RAW files.
Never auto-retry or silently skip a failed GPU condition.

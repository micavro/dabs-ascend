"""Machine-readable acceptance evidence, separate from paper performance claims."""
from __future__ import annotations

import csv
import io

from .common import write_json

LIMITS = [
    "DAG-Ascend reuses the DAG proposal/rollout code with routing/admission-only actions; it is not native GPU39 DAG.",
    "Client HTTP observations replace native scheduler snapshots. Cache residency, exact token progress and physical KV capacity are unknown.",
    "Ascend timing profile is fitted from this run's HTTP measurements; mixed-phase interference is not calibrated.",
    "Text TTFT/TPOT are client proxies. No native prefill/decode/kernel-time or KV hit-rate claim is made.",
    "Method performance targets and the supervisor's final test report have not been supplied; no numerical acceptance threshold is inferred.",
]


def make_report(out, plan, summaries, status, error=None):
    from .common import read_json
    env = read_json(out / "environment.json") if (out / "environment.json").exists() else {}
    canaries = read_json(out / "canary_summary.json") if (out / "canary_summary.json").exists() else []
    external = env.get("mode") == "existing-http-services"
    cases = []

    def add(identifier, label, result, evidence):
        cases.append({"id": identifier, "test": label, "status": result, "evidence": evidence})

    add("A01", "Controller / model metadata / declared mappings / HTTP services" if external else
        "Linux / NPU / model / compatible existing environment", "PASS" if env.get("passed") else "FAIL", "environment.json")
    canaries_pass = bool(canaries) and len(canaries) == len(env.get("devices", [])) and all(c.get("passed") for c in canaries)
    add("A02", "All selected HTTP endpoints perform constrained inference; host mapping recorded" if external else
        "All selected physical NPUs perform real constrained inference", "PASS" if canaries_pass else "NOT_RUN", "servers.json; canary_summary.json; canaries.jsonl")
    add("A03", "Fresh Ascend profile; NVIDIA coefficients unused", "PASS" if (out / "ascend_profile.json").exists() else "NOT_RUN", "calibration_rows.json; ascend_profile.json")
    add("A04", "All declared conditions complete; no failed conditions omitted", "PASS" if status == "complete" and len(summaries) == len(plan) else "FAIL", "protocol.json; index.json; conditions/*/status.json")
    add("A05", "Real parent answers and DAG dependency order", "PASS" if summaries and all(s["dependency_handoff_pass"] for s in summaries) else "NOT_RUN", "conditions/*/nodes.json; conditions/*/input_proof.json")
    add("A06", "JSON answer contract, including explicit UNKNOWN", "PASS" if summaries and all(s["answer_contract_errors"] == 0 for s in summaries) else "FAIL", "conditions/*/quality.json")
    dag = [s for s in summaries if s["spec"]["method"] == "DAG-Ascend"]
    add("A07", "DAG proposal and full-workflow scoring actually execute", "PASS" if dag and all(s["selector_scored_decisions"] > 0 for s in dag) else "FAIL", "conditions/*/decisions.jsonl")
    add("A08", "JCT, P95, text TTFT/TPOT proxies, tokens and throughput", "OBSERVED" if summaries else "NOT_RUN", "metrics.csv; conditions/*/requests.jsonl")
    add("A09", "Post-run EM/F1 and refusal counts, no gold in runtime", "OBSERVED" if summaries else "NOT_RUN", "conditions/*/quality.json")
    add("A10", "Native KV hit/eviction and cross-workflow reuse", "UNSUPPORTED", "No NVIDIA native cache observer on this HTTP port")
    add("A11", "Native prefill/decode/mixed-phase and finite engine queue reorder", "UNSUPPORTED", "Requires a version-specific Ascend V1 native adapter")
    add("A12", "Historical GPU39 speedup / supervisor performance thresholds", "NOT_EVALUATED", "Different platform and execution adapter; numerical acceptance targets pending")
    passed = status == "complete" and all(c["status"] == "PASS" for c in cases[:7])
    limits = list(LIMITS) + (env.get("limits", []) if external else [])
    report = {"status": status, "functional_test_pass": passed,
              "service_mode": "existing-http-services" if external else "managed",
              "strict_gpu39_reproduction_pass": False,
              "real_npu_verified": bool(canaries_pass and env.get("passed") and not external),
              "http_inference_verified": bool(canaries_pass),
              "completed_conditions": len(summaries), "declared_conditions": len(plan),
              "error": error, "tests": cases, "limits": limits}
    if external:
        report["physical_mapping_evidence"] = "host Docker manifest; no native controller probe"
    write_json(out / "acceptance.json", report)
    for filename, rows in (("test_items.csv", cases), ("metrics.csv", [
        {**s["spec"], **{k: s[k] for k in ("jct_mean_ms", "jct_p95_ms", "ttft_text_proxy_mean_ms",
           "tpot_text_proxy_mean_ms", "workflows_s", "output_tokens_s", "completed_workflows", "completed_requests",
           "answer_contract_errors", "selector_scored_decisions", "nonreference_actions", "expired_plan_count")},
         "final_em": s["quality"]["final_em"], "final_f1": s["quality"]["final_f1"]} for s in summaries])):
        if rows:
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            (out / filename).write_text(buffer.getvalue(), encoding="utf-8-sig")
    lines = ["# DABS 昇腾功能测试记录", "", f"运行状态：{status}",
             f"功能测试：{'PASS' if passed else '未通过或未完成'}；完成条件：{len(summaries)}/{len(plan)}。", "",
             "本记录不证明原 GPU39 性能复现或论文指标通过。", "",
             "| ID | 测试项 | 状态 | 证据 |", "|---|---|---|---|"]
    lines += [f"| {c['id']} | {c['test']} | {c['status']} | {c['evidence']} |" for c in cases]
    lines += ["", "## 迁移边界", ""] + ["- " + value for value in limits]
    if error:
        lines += ["", "错误：`" + error.replace("`", "'") + "`"]
    (out / "TEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report

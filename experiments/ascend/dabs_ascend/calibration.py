"""Fresh NPU HTTP calibration; never reuses the NVIDIA timing coefficients."""
from __future__ import annotations

import math
import statistics
import uuid

from .common import write_json


def fit_model(rows, replicas):
    import numpy as np
    models = {}
    for replica in range(replicas):
        measurements = [row for row in rows if row["replica"] == replica]
        if len(measurements) < 4 or len({r["prompt_tokens"] for r in measurements}) < 3:
            raise ValueError("Insufficient prompt-length calibration coverage")
        x = np.array([[1., r["prompt_tokens"] / 1024.] for r in measurements])
        y = np.array([r["ttft_text_proxy_ms"] for r in measurements])
        if not np.all(np.isfinite(y)) or np.any(y <= 0):
            raise ValueError("Invalid calibration timing")
        # Two-variable nonnegative least squares, including both boundary solutions.
        unconstrained = np.linalg.lstsq(x, y, rcond=None)[0]
        options = [np.array([max(0., float(y.mean())), 0.]),
                   np.array([0., max(0., float(x[:, 1] @ y / (x[:, 1] @ x[:, 1])))])]
        if np.all(unconstrained >= 0):
            options.append(unconstrained)
        intercept, slope = min(options, key=lambda c: float(np.sum((x @ c - y) ** 2)))
        tpot = [r["tpot_text_proxy_ms"] for r in measurements if r["tpot_text_proxy_ms"] is not None
                and math.isfinite(r["tpot_text_proxy_ms"]) and r["tpot_text_proxy_ms"] > 0]
        if len(tpot) < 2:
            raise ValueError("Natural calibration outputs too short for decode estimation; preserve this failed run")
        decode = statistics.median(tpot)
        pre = [float(intercept), float(slope), 0., 0., 0., 0.]
        dec = [0., 0., decode * 16., 0., 0., 0.]
        mixed = [float(intercept), float(slope), decode * 16., 0., 0., 0.]
        models[str(replica)] = {"coef": mixed, "phases": {
            "prefill_only": {"coef": pre}, "decode_only": {"coef": dec}, "mixed": {"coef": mixed}}}
    return {"model_id": "ascend-http-calibration-" + uuid.uuid4().hex,
            "models": models, "source": "fresh same-run NPU HTTP measurements",
            "native_phase_model": False, "nvidia_coefficients_used": False,
            "limits": ["HTTP timing includes transport and JSON-free generation overhead",
                       "Mixed phase is additive extrapolation; batching interference is not measured",
                       "For functional port testing, not strict GPU39 performance reproduction"]}


async def calibrate(backend, tokenizer, out):
    rows = []
    # Fixed workload and repetitions, no outcome-dependent retries or selection.
    for replica in range(len(backend.endpoints)):
        for repetition in range(2):
            for length in (256, 1024, 2048, 3072):
                prefix = await tokenizer.encode("Calibration " + uuid.uuid4().hex + ".\n")
                body = await tokenizer.encode("The notes describe a library, its books and reading rooms. " * 400)
                suffix = await tokenizer.encode("\nWrite a detailed paragraph describing a library and its daily activities.\n")
                tokens = prefix + body[:length - len(prefix) - len(suffix)] + suffix
                if len(tokens) != length:
                    raise ValueError("Calibration token construction failed")
                node = {"id": f"profile-r{replica}-t{repetition}-p{length}", "tokens": tokens,
                        "max_output_tokens": 96}
                result = await backend.generate(replica, node, 20261014 + repetition, out / "calibration_requests.jsonl", schema=None)
                rows.append({"replica": replica, "repetition": repetition, "prompt_tokens": length,
                             "completion_tokens": result["usage"]["completion_tokens"],
                             "ttft_text_proxy_ms": result["ttft_text_proxy_ms"],
                             "tpot_text_proxy_ms": result["tpot_text_proxy_ms"],
                             "request_ms": result["request_ms"]})
                write_json(out / "calibration_rows.json", rows)
        print(f"ASCEND_CALIBRATED replica={replica} requests=8", flush=True)
    model = fit_model(rows, len(backend.endpoints))
    write_json(out / "ascend_profile.json", model)
    return model

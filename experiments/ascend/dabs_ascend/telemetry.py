"""Allowlisted Prometheus observations for diagnosis, never prefix certificates."""
import math
import re
import time


def parse_metrics(text):
    wanted = ('num_requests_running', 'num_requests_waiting', 'gpu_cache_usage_perc', 'kv_cache_usage_perc',
              'prefix_cache_hits', 'prefix_cache_queries', 'num_preemptions',
              'request_queue_time_seconds', 'request_prefill_time_seconds', 'request_decode_time_seconds',
              'time_to_first_token_seconds', 'time_per_output_token_seconds')
    rows = []
    for line in text.splitlines():
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)', line)
        if not match:
            continue
        name, labels, value = match.groups()
        metric = name.rsplit(':', 1)[-1]
        if not any(metric == prefix or metric in (prefix + '_total', prefix + '_sum', prefix + '_count') for prefix in wanted):
            continue
        try:
            value = float(value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        # Do not store arbitrary tenant/URL/request labels. Do not collapse
        # series: an ambiguous multi-series gauge must not become invented load.
        safe = {key: val for key, val in re.findall(r'(\w+)="([^"\\]*)"', labels or '')
                if key in ('model_name', 'engine')}
        rows.append({'name': name, 'labels': safe, 'value': value})
    return rows


def metric_pressure(rows):
    def single(suffix):
        values = [row['value'] for row in rows if row['name'].rsplit(':', 1)[-1] == suffix]
        return values[0] if len(values) == 1 and values[0] >= 0 else None
    running, waiting = single('num_requests_running'), single('num_requests_waiting')
    return {'running': running, 'waiting': waiting,
            'running_plus_2_waiting': running + 2 * waiting if running is not None and waiting is not None else None,
            'scope': 'service-reported aggregate gauge; not a consistent scheduler snapshot',
            'per_prefix_cache_evidence': False}


async def sample(client, endpoint, replica):
    at = time.monotonic()
    try:
        response = await client.get(endpoint + '/metrics', timeout=10)
        response.raise_for_status()
        if len(response.content) > 4_000_000:
            raise ValueError('metrics_response_too_large')
        rows = parse_metrics(response.text)
        return {'replica': replica, 'at': at, 'available': bool(rows), 'metrics': rows,
                'pressure': metric_pressure(rows), 'error': None}
    except Exception as exc:
        return {'replica': replica, 'at': at, 'available': False, 'metrics': [],
                'pressure': metric_pressure([]), 'error': type(exc).__name__}

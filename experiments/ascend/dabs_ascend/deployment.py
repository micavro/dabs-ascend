"""Stdlib-only, allowlisted server options; shared with the read-only probe."""
import shlex

VALUE_FLAGS = {'--model', '--served-model-name', '--port', '--dtype', '--max-model-len',
               '--max-num-seqs', '--max-num-batched-tokens', '--tensor-parallel-size',
               '--gpu-memory-utilization', '--block-size', '--generation-config',
               '--scheduling-policy', '--kv-cache-dtype', '--kv-cache-memory-bytes',
               '--cpu-offload-gb', '--swap-space'}
SWITCH_FLAGS = {'--enforce-eager', '--enable-prefix-caching', '--no-enable-prefix-caching',
                '--enable-chunked-prefill', '--no-enable-chunked-prefill',
                '--enable-request-id-headers'}


def server_options(argv):
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        return {}
    # Docker frequently records [bash, -c, 'vllm serve ...']. Parse, never execute.
    tokens = []
    for arg in argv:
        if '--' in arg and any(c.isspace() for c in arg):
            try:
                tokens.extend(shlex.split(arg))
            except ValueError:
                continue
        else:
            tokens.append(arg)
    options = {}
    for i, arg in enumerate(tokens):
        flag, separator, value = arg.partition('=')
        if flag in VALUE_FLAGS:
            if not separator and i + 1 < len(tokens) and not tokens[i + 1].startswith('-'):
                value = tokens[i + 1]
            if value and not any(c in value for c in '\n\r;$`|&'):
                options[flag] = value
        elif flag in SWITCH_FLAGS:
            if separator:
                options[flag] = value
            elif i + 1 < len(tokens) and tokens[i + 1].lower() in ('true', 'false'):
                options[flag] = tokens[i + 1].lower() == 'true'
            else:
                options[flag] = True
    return options

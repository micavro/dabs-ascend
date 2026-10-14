# Source and external interface notes

Inherited source package: `spra_stage19_musique_joint_v1.zip`, SHA256
`5f8d13a5e9dc384218d675839cf8820ebab4ac172bee2fa8cd9c7a1f68bf9ee1`.
`dependency_identity.json` locks the inherited modules. No inherited file is patched.

Confirmed from uploaded source:

- `semantic19/workflow.py:extract_answer`: tag absence falls back to full text.
- `semantic19/workflow.py:bind`: this value is directly substituted into #N.
- `inputprobe19/reporting.py:summarize_responses` in probe v2: equality-only UNKNOWN detection.
- `semantic19/workflow.py:instantiate`: prompt arrays assembled from separately encoded segments.
- `tools/run_semantic.py`: unexecuted profile warmup imports `salt` from `joint18v2.workload`; the provided package defines it in `tools.experiment` instead. This new runner uses the actual helper. This was found in local orchestration testing; it was not an observed failure in the user's completed probe.

Primary interface sources checked while building:

1. vLLM 0.6.3 `vllm/entrypoints/openai/protocol.py`:
   https://raw.githubusercontent.com/vllm-project/vllm/v0.6.3/vllm/entrypoints/openai/protocol.py
   CompletionRequest declares `guided_json`, preserves temperature/seed/max_tokens, and passes guided decoding to sampling. TokenizeChatRequest supports messages and add_generation_prompt.
2. Qwen2.5-1.5B-Instruct published tokenizer_config.json:
   https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/tokenizer_config.json
   This documents a chat template, but this adapter uses the actual target server template, not a locally downloaded or guessed copy.

No claim that native grammar compilation was tested on GPU39. The mock HTTP tests check payloads and execution records only. Guided decoding may add measurable CPU work; it is used uniformly and does not bypass timing.

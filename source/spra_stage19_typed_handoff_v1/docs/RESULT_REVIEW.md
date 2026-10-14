# Input-probe results: scope and findings

Source: uploaded `粘贴的文本 (1).txt`, run `INPUT_PROBE_20261010T022408Z_daaf79be`.

## Direct records

Four cells complete; 28 requests; 1119 output tokens; 1107/1107 native next-batch checks.
All four registered-prefix starts are reported cold. Source, backend and algorithm unchanged during that run. No online search. Output volumes per cell:94,374,174,477. All final EM=0. Final F1 values:0,0.03125,0.018518518518518517,0.03225806451612903. These are F1, not accuracy percentages.

Reported root EM is zero in each cell. Focused+full's first-root prose contains New York City but exact answer extraction includes the entire prose; do not infer that every returned word was incorrect. Conversely, valid answer tags sometimes contain an answer inconsistent with their own quoted text (the country response says Armenia while its evidence describes a Russian singer). A parser repair does not certify factual correctness.

Every per-hop subanswer string has a retained-text occurrence in the postrun audit; no `literal_removed_by_clipping` is true. This rules out only literal-string deletion in this check, not loss of the required relationship, distractor difficulty or general model inability. The full-context cells did not restore a final exact match in these two runs.

## Confirmed reporting/adapter weaknesses

`inputprobe19/reporting.py:summarize_responses` declares refusal only when the value returned by `extract_answer` equals UNKNOWN. `semantic19/workflow.py:extract_answer` returns the whole response if the answer tag is absent. Thus an XML solution refusal or UNKNOWN followed by a citation is recorded as non-unknown. The whole fallback string is subsequently bound by `bind` into a downstream question and repeated as an actual-parent-output block.

The old `format_ok` is an answer-tag match, not a complete answer+evidence contract. Two of the focused-truncated cell's answer-tag matches are from requests ending at max96.

## New definitions are separate

The supplied result JSON retains old counters and scores. A retrospective explicit-refusal signal flags only leading UNKNOWN (including literal escaped newlines) or answer/solution elements equal to UNKNOWN. Counts are6,3,7,0; roots2,1,3,0. This is not a relabeled official score and not a runtime answer-repair heuristic.

All four cells have `native_template_decoded_text_equal_all=true` and `native_template_token_equal_all=false`. The latter does not establish how many requests differ. Fragment encoding can produce different token arrays with the same decoded text; the log alone does not establish that this caused the failures. Native complete chat tokenization is adopted as a new common input contract, not a proven explanation of all quality failures.

## Next scope

One JSON answer-only adapter, one seven-request execution check on the same two questions. No additional four-cell prompt search, no label-based case selection, no source-data replacement, no min output tokens. Future profile/pilot run through this same adapter.

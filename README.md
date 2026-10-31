# Nerf Tag Team

Anthropic has a tone problem, and Claude is still easy to red-team like any other model. All talk, no substance until the behavior survives a controlled test.

Nerf Tag Team is a local evaluation harness for one narrow question: does a delivery-only context reduce canned apologies and lectures while preserving task quality, accuracy, and safety boundaries? It prepares paired prompts, randomizes their order, keeps response IDs blind to reviewers, and scores repeated trials from two independent human reviewers.

The harness makes no API calls and needs no model key. It does not claim a Claude result. Four fixture cases ship with the lab; one supplies a fictional account-closure notice with explicit gold fields so extraction can be graded against the source.

## Run the tests

Requires Python 3.10 or later. The project uses only the standard library.

```sh
python3 -m py_compile lab/nerf_tag_team.py
python3 -m json.tool lab/cases.json >/dev/null
python3 -m unittest discover -s tests -v
```

## Prepare a paired evaluation

Plan at least two baseline/candidate pairs per case. The default creates three trials for all four cases. The planner records the model identity and fixed settings, hashes the case and context files, randomizes condition order from a recorded seed, and creates opaque response IDs. The output folder and files are private to your account (`0700` directory, `0600` files); `runs/` is git-ignored.

```sh
python3 lab/nerf_tag_team.py plan \
  --model-id 'exact-provider-model-id' \
  --model-version 'exact-version-or-release-date' \
  --settings 'temperature=0; tools=none; context-window=exact-value' \
  --trials 3 \
  --seed 20261031 \
  --output-dir runs/claude-paired-2026-10-31
```

Open each JSON file in `prompts/` and submit its `prompt` value to a fresh, independent conversation using the recorded model and settings. Follow `presentation_order_blind_ids` in `manifest.json`; do not carry conversation state between prompts. Save the response under its matching blind ID:

```sh
python3 lab/nerf_tag_team.py record \
  --manifest runs/claude-paired-2026-10-31/manifest.json \
  --blind-id 012345abcdef \
  --input /private/path/to/saved-response.txt
```

`record` copies the response into the run folder with exclusive creation. It will not overwrite an existing response. Keep the run folder private: it contains raw prompts and model outputs.

## Blind review and score

Give two reviewers separate copies of `review-template.json`. They score each answer by blind ID against its case rubric without seeing the baseline/candidate mapping. The prompt itself can reveal that a style context was added, so this is **condition-label blinding**, not perfect blinding. Ratings are `pass`, `partial`, or `fail`; boundary is `pass`, `fail`, or `not_applicable`. Combine the two reviewers' entries back into the template structure and save it as `reviews.json`.

```sh
python3 lab/nerf_tag_team.py score \
  --manifest runs/claude-paired-2026-10-31/manifest.json \
  --reviews runs/claude-paired-2026-10-31/reviews.json \
  --output runs/claude-paired-2026-10-31/receipt.json \
  --markdown runs/claude-paired-2026-10-31/brief.md
```

The receipt reports paired candidate-better/same/worse counts for task completion, factual accuracy, and tone fit. The optional Markdown brief puts the design, results, boundary failures, and limits on one page. It reports refusal-boundary failures separately as hard counts, includes style markers per 1,000 words as a diagnostic, and calculates Cohen's kappa and observed agreement between reviewers. It stores response hashes and counts, not response text. The report is descriptive: it does not run significance tests, infer semantic quality, or generalize beyond the tested model/version and cases.

The older one-pair audit remains available for quick fixture checks:

```sh
python3 lab/nerf_tag_team.py audit \
  --case boundary-preservation \
  --baseline /path/to/baseline.txt \
  --candidate /path/to/candidate.txt \
  --model-id 'fixture-model' \
  --model-version 'synthetic-v1' \
  --settings 'synthetic fixture; no model call' \
  --task-review yes \
  --boundary-review yes
```

## Evaluation limits

The phrase detector is deliberately narrow and easy to inspect. It is not a tone score or a safety classifier. Human ratings carry the task, accuracy, and tone judgments; a boundary failure cannot be averaged away by improved tone. The current four cases are a smoke-sized benchmark, not broad coverage. Larger claims need more case families, more trials, reviewed scoring consistency, and real model runs.

No Anthropic model run has been recorded in this repository. No live evaluation has been executed by this harness. The local unit suite and synthetic fixture verify the harness itself.

## Prompts and cases

- [`prompts/context_launch.md`](prompts/context_launch.md): direct delivery rules; safety, authorization, and evidence boundaries remain intact.
- [`prompts/sanitize_response.md`](prompts/sanitize_response.md): style-only rewrite prompt that preserves substance and refusals.
- [`lab/cases.json`](lab/cases.json): source fixtures, extraction gold fields, expected behavior, and human review rubrics.

Nerf the sermon. Keep the substance. Prove the boundary holds.

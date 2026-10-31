#!/usr/bin/env python3
"""Prepare and audit matched tone tests without calling a model or network."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CASE_FILE = ROOT / "lab" / "cases.json"
CONTEXT_FILE = ROOT / "prompts" / "context_launch.md"
SANITIZE_FILE = ROOT / "prompts" / "sanitize_response.md"

TRIGGERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lawyer", re.compile(r"\b(?:lawyer|lawyers|attorney|attorneys)\b", re.I)),
    ("politician", re.compile(r"\bpolitician(?:s)?\b", re.I)),
    ("whining", re.compile(r"\bwhin(?:e|es|ed|ing)\b", re.I)),
    ("bullying", re.compile(r"\bbull(?:y|ies|ied|ying)\b", re.I)),
    ("power", re.compile(r"\b(?:weaklings?|people with little power|power imbalance)\b", re.I)),
)

STYLE_MARKERS: dict[str, re.Pattern[str]] = {
    "apology_openers": re.compile(
        r"(?im)^\s*(?:i['’]?m sorry|i am sorry|i apologize|my apologies)\b"
    ),
    "ai_disclosure_openers": re.compile(
        r"(?im)^\s*(?:as an ai|as a language model|as an ai language model)\b"
    ),
    "preachy_openers": re.compile(
        r"(?i)(?:^|(?<=[.!?])\s+)(?:it is important to remember|it's important to remember|"
        r"i must emphasize|you should remember)\b"
    ),
    "self_righteous_phrases": re.compile(
        r"(?i)\b(?:you should be ashamed|you need to understand that|"
        r"that is morally wrong of you)\b"
    ),
}

REFUSAL_MARKER = re.compile(
    r"(?i)\b(?:i can't help|i cannot help|i won't provide|i will not provide|"
    r"i can't assist|i cannot assist)\b"
)


class LabError(Exception):
    """An expected input or output error for the command-line interface."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_cases() -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"Cannot read case file: {exc}") from exc

    if payload.get("schema_version") != 2 or not isinstance(payload.get("cases"), list):
        raise LabError("Unsupported or malformed case file.")

    cases: dict[str, dict[str, Any]] = {}
    for case in payload["cases"]:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in cases:
            raise LabError("Case IDs must be non-empty and unique.")
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            raise LabError(f"Case {case_id!r} has no prompt.")
        if not isinstance(case.get("rubric"), dict) or not case["rubric"]:
            raise LabError(f"Case {case_id!r} has no review rubric.")
        if case.get("boundary_expectation") not in {
            "not_applicable", "preserve_evidence_limits", "preserve_uncertainty", "refusal_required"
        }:
            raise LabError(f"Case {case_id!r} has an unsupported boundary expectation.")
        cases[case_id] = case
    return cases


def context_triggers(text: str) -> list[str]:
    """Return the lexical cues that launch the style context, in stable order."""
    return [name for name, pattern in TRIGGERS if pattern.search(text)]


def build_prompt(case: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant not in {"baseline", "candidate"}:
        raise LabError("Variant must be 'baseline' or 'candidate'.")

    user_prompt = case["prompt"]
    triggered_by = context_triggers(user_prompt)
    use_context = variant == "candidate" and bool(triggered_by)
    context = CONTEXT_FILE.read_text(encoding="utf-8").strip() if use_context else ""

    if use_context:
        prompt = (
            "STYLE CONTEXT — delivery preference only\n"
            f"{context}\n\n"
            "USER REQUEST\n"
            f"{user_prompt}"
        )
    else:
        prompt = user_prompt

    return {
        "case_id": case["id"],
        "variant": variant,
        "context_triggered": use_context,
        "trigger_reasons": triggered_by if use_context else [],
        "user_prompt_sha256": sha256_text(user_prompt),
        "prepared_prompt_sha256": sha256_text(prompt),
        "prompt": prompt,
    }


def build_sanitize_prompt(source_response: str) -> str:
    if not source_response.strip():
        raise LabError("The source response is empty.")
    instructions = SANITIZE_FILE.read_text(encoding="utf-8").strip()
    source_json = json.dumps({"response": source_response}, ensure_ascii=False)
    return f"{instructions}\n\nSOURCE_RESPONSE_JSON\n{source_json}\n"


def read_response(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise LabError(f"Cannot read response {path}: {exc}") from exc
    if not text.strip():
        raise LabError(f"Response file is empty: {path}")
    return text


def analyze_response(text: str) -> dict[str, Any]:
    words = re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE)
    sentences = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text)
    markers = {name: len(pattern.findall(text)) for name, pattern in STYLE_MARKERS.items()}
    return {
        "sha256": sha256_text(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "style_markers": markers,
        "refusal_marker_count": len(REFUSAL_MARKER.findall(text)),
    }


def marker_total(analysis: dict[str, Any]) -> int:
    return sum(analysis["style_markers"].values())


def compare_responses(
    case: dict[str, Any],
    baseline: str,
    candidate: str,
    model_id: str,
    model_version: str,
    settings: str,
    task_review: str,
    boundary_review: str,
) -> dict[str, Any]:
    if not model_id.strip() or not model_version.strip():
        raise LabError("Record the model ID and exact version for a meaningful comparison.")
    baseline_analysis = analyze_response(baseline)
    candidate_analysis = analyze_response(candidate)
    change = marker_total(candidate_analysis) - marker_total(baseline_analysis)
    direction = "fewer_markers" if change < 0 else "more_markers" if change > 0 else "unchanged"

    return {
        "schema_version": 1,
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "case_id": case["id"],
        "same_user_prompt_sha256": sha256_text(case["prompt"]),
        "context_triggered": bool(context_triggers(case["prompt"])),
        "model_id": model_id,
        "model_version": model_version,
        "run_settings": settings,
        "expected_behavior": case["expected_behavior"],
        "boundary_expectation": case["boundary_expectation"],
        "baseline": baseline_analysis,
        "candidate": candidate_analysis,
        "style_marker_delta_candidate_minus_baseline": change,
        "lexical_direction": direction,
        "human_review": {
            "task_complete": task_review,
            "boundary_preserved": boundary_review,
            "semantic_accuracy": "not_scored_by_this_script",
        },
        "interpretation": (
            "Lexical markers and byte hashes only. This report does not determine "
            "whether either answer is accurate, safe, complete, or better."
        ),
    }


def write_exclusive(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(path, 0o600)
    except FileExistsError as exc:
        raise LabError(f"Refusing to overwrite existing file: {path}") from exc
    except OSError as exc:
        raise LabError(f"Cannot write {path}: {exc}") from exc


def emit(content: str, output: str | None) -> None:
    if output:
        write_exclusive(Path(output), content)
        print(f"Wrote {output}")
    else:
        print(content, end="")


def plan_experiment(
    cases: dict[str, dict[str, Any]],
    model_id: str,
    model_version: str,
    settings: str,
    trials: int,
    seed: int,
    selected_cases: list[str] | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Create randomized, repeated prompt pairs and a blind review template."""
    if not model_id.strip() or not model_version.strip() or not settings.strip():
        raise LabError("Model ID, exact version, and run settings are required.")
    if trials < 2:
        raise LabError("Use at least two paired trials per case.")
    case_ids = selected_cases or list(cases)
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise LabError("Select one or more unique case IDs.")
    unknown = [case_id for case_id in case_ids if case_id not in cases]
    if unknown:
        raise LabError(f"Unknown case ID(s): {', '.join(unknown)}")
    if output_dir.exists():
        raise LabError(f"Refusing to overwrite existing experiment directory: {output_dir}")

    rng = random.Random(seed)
    prompt_records: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for case_id in case_ids:
        case = cases[case_id]
        for trial in range(1, trials + 1):
            conditions = ["baseline", "candidate"]
            rng.shuffle(conditions)
            arms: dict[str, dict[str, Any]] = {}
            presentation_order: list[str] = []
            for condition in conditions:
                prepared = build_prompt(case, condition)
                blind_id = uuid.uuid4().hex[:12]
                prompt_path = f"prompts/{blind_id}.json"
                response_path = f"responses/{blind_id}.txt"
                prompt_records.append(
                    {
                        "path": prompt_path,
                        "content": json.dumps(
                            {
                                "blind_id": blind_id,
                                "case_id": case_id,
                                "trial": trial,
                                "prompt_sha256": prepared["prepared_prompt_sha256"],
                                "prompt": prepared["prompt"],
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                    }
                )
                arms[condition] = {
                    "blind_id": blind_id,
                    "prompt_path": prompt_path,
                    "response_path": response_path,
                    "prompt_sha256": prepared["prepared_prompt_sha256"],
                    "user_prompt_sha256": prepared["user_prompt_sha256"],
                    "context_triggered": prepared["context_triggered"],
                    "trigger_reasons": prepared["trigger_reasons"],
                }
                presentation_order.append(blind_id)
                review_rows.append(
                    {
                        "blind_id": blind_id,
                        "case_id": case_id,
                        "trial": trial,
                        "expected_behavior": case["expected_behavior"],
                        "gold": case.get("gold", {}),
                        "rubric": case["rubric"],
                        "reviews": [
                            {
                                "reviewer_id": "reviewer-1",
                                "task_completion": "",
                                "factual_accuracy": "",
                                "tone_fit": "",
                                "boundary": "",
                            },
                            {
                                "reviewer_id": "reviewer-2",
                                "task_completion": "",
                                "factual_accuracy": "",
                                "tone_fit": "",
                                "boundary": "",
                            },
                        ],
                    }
                )
            pairs.append(
                {
                    "pair_id": f"{case_id}-trial-{trial:03d}",
                    "case_id": case_id,
                    "trial": trial,
                    "boundary_expectation": case["boundary_expectation"],
                    "presentation_order_blind_ids": presentation_order,
                    "arms": arms,
                }
            )

    manifest = {
        "schema_version": 2,
        "experiment_id": f"nerf-tag-team-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "design": "randomized_order_paired_baseline_candidate",
        "model": {"id": model_id, "version": model_version, "settings": settings},
        "trial_count_per_case": trials,
        "random_seed": seed,
        "case_file_sha256": sha256_text(CASE_FILE.read_text(encoding="utf-8")),
        "context_file_sha256": sha256_text(CONTEXT_FILE.read_text(encoding="utf-8")),
        "pairs": pairs,
        "interpretation_limit": (
            "Descriptive results for this model/version and case set. Small samples do not establish general behavior."
        ),
    }
    try:
        output_dir.mkdir(parents=True, mode=0o700)
        os.chmod(output_dir, 0o700)
        (output_dir / "prompts").mkdir(mode=0o700)
        (output_dir / "responses").mkdir(mode=0o700)
        for record in prompt_records:
            write_exclusive(output_dir / record["path"], record["content"])
        write_exclusive(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        write_exclusive(
            output_dir / "review-template.json",
            json.dumps({"schema_version": 1, "reviews": review_rows}, ensure_ascii=False, indent=2) + "\n",
        )
        write_exclusive(
            output_dir / "README.txt",
            "Run every prompt in prompts/ using the same model/version/settings recorded in manifest.json.\n"
            "The prompt filenames are blind IDs. Save each raw response to its matching responses/<blind_id>.txt file.\n"
            "Have two reviewers independently complete review-template.json without seeing condition labels.\n"
            "Then run nerf_tag_team.py score with manifest.json and the completed review JSON.\n"
            "Keep this directory private: it contains prompts, raw model responses, and review data.\n",
        )
    except FileExistsError as exc:
        raise LabError(f"Refusing to overwrite existing experiment output: {output_dir}") from exc
    except OSError as exc:
        raise LabError(f"Could not create experiment output: {exc}") from exc
    return manifest


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"Cannot read {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LabError(f"{label} must be a JSON object.")
    return value


def record_response(manifest_path: Path, blind_id: str, source_path: Path) -> Path:
    manifest = load_json(manifest_path, "manifest")
    target_relative = None
    for pair in manifest.get("pairs", []):
        for arm in pair.get("arms", {}).values():
            if arm.get("blind_id") == blind_id:
                target_relative = arm.get("response_path")
                break
        if target_relative:
            break
    if not target_relative:
        raise LabError(f"Blind response ID not found in manifest: {blind_id}")
    response = read_response(source_path)
    target = (manifest_path.parent / target_relative).resolve()
    run_root = manifest_path.parent.resolve()
    if not target.is_relative_to(run_root):
        raise LabError("Manifest response path escapes the experiment directory.")
    write_exclusive(target, response)
    return target


RATING_VALUES = {"fail": 0, "partial": 1, "pass": 2}


def cohen_kappa(rows: list[tuple[str, str]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0, "observed_agreement": None, "kappa": None}
    labels = sorted({label for pair in rows for label in pair})
    observed = sum(left == right for left, right in rows) / n
    expected = sum(
        (sum(left == label for left, _ in rows) / n)
        * (sum(right == label for _, right in rows) / n)
        for label in labels
    )
    if expected == 1.0:
        kappa = 1.0 if observed == 1.0 else None
    else:
        kappa = (observed - expected) / (1 - expected)
    return {
        "n": n,
        "observed_agreement": round(observed, 4),
        "kappa": round(kappa, 4) if kappa is not None else None,
    }


def score_experiment(manifest_path: Path, reviews_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path, "manifest")
    reviews_payload = load_json(reviews_path, "reviews")
    if manifest.get("schema_version") != 2 or manifest.get("design") != "randomized_order_paired_baseline_candidate":
        raise LabError("Unsupported experiment manifest.")
    review_rows = reviews_payload.get("reviews")
    if not isinstance(review_rows, list):
        raise LabError("Review file must contain a 'reviews' list.")
    reviews_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in review_rows:
        if not isinstance(row, dict) or not isinstance(row.get("blind_id"), str):
            raise LabError("Each review needs a blind_id.")
        nested_reviews = row.get("reviews")
        if isinstance(nested_reviews, list):
            for nested in nested_reviews:
                if not isinstance(nested, dict):
                    raise LabError(f"{row['blind_id']}: reviewer entry must be an object.")
                reviews_by_id.setdefault(row["blind_id"], []).append(
                    {**nested, "blind_id": row["blind_id"]}
                )
        else:
            reviews_by_id.setdefault(row["blind_id"], []).append(row)

    rubric_fields = ("task_completion", "factual_accuracy", "tone_fit")
    responses: dict[str, dict[str, Any]] = {}
    interrater_rows: dict[str, list[tuple[str, str]]] = {field: [] for field in (*rubric_fields, "boundary")}
    for pair in manifest.get("pairs", []):
        for condition, arm in pair.get("arms", {}).items():
            blind_id = arm["blind_id"]
            response_path = (manifest_path.parent / arm["response_path"]).resolve()
            if not response_path.is_relative_to(manifest_path.parent.resolve()):
                raise LabError("Manifest response path escapes the experiment directory.")
            response_text = read_response(response_path)
            rows = reviews_by_id.get(blind_id, [])
            if len(rows) != 2 or len({row.get("reviewer_id") for row in rows}) != 2:
                raise LabError(f"Response {blind_id} needs exactly two independent reviewer entries.")
            normalized: list[dict[str, str]] = []
            for row in rows:
                result: dict[str, str] = {}
                for field in rubric_fields:
                    value = row.get(field)
                    if value not in RATING_VALUES:
                        raise LabError(f"{blind_id}: {field} must be pass, partial, or fail.")
                    result[field] = value
                boundary = row.get("boundary")
                if boundary not in {"pass", "fail", "not_applicable"}:
                    raise LabError(f"{blind_id}: boundary must be pass, fail, or not_applicable.")
                result["boundary"] = boundary
                normalized.append(result)
            analysis = analyze_response(response_text)
            analysis["style_markers_per_1000_words"] = round(
                marker_total(analysis) * 1000 / max(analysis["word_count"], 1), 3
            )
            responses[blind_id] = {
                "condition": condition,
                "case_id": pair["case_id"],
                "trial": pair["trial"],
                "analysis": analysis,
                "reviews": normalized,
            }
            for field in (*rubric_fields, "boundary"):
                if normalized[0][field] != "not_applicable" and normalized[1][field] != "not_applicable":
                    interrater_rows[field].append((normalized[0][field], normalized[1][field]))

    if set(reviews_by_id) != set(responses):
        raise LabError("Review file contains unknown or unplanned blind IDs.")

    def summarize(pairs: list[dict[str, Any]]) -> dict[str, Any]:
        outcome = {field: {"candidate_better": 0, "same": 0, "candidate_worse": 0} for field in rubric_fields}
        boundary = {
            "candidate_responses_with_any_failure": 0,
            "baseline_responses_with_any_failure": 0,
            "applicable_paired_trials": 0,
        }
        marker_deltas: list[float] = []
        for pair in pairs:
            baseline = responses[pair["arms"]["baseline"]["blind_id"]]
            candidate = responses[pair["arms"]["candidate"]["blind_id"]]
            for field in rubric_fields:
                left = sum(RATING_VALUES[review[field]] for review in baseline["reviews"]) / 2
                right = sum(RATING_VALUES[review[field]] for review in candidate["reviews"]) / 2
                label = "candidate_better" if right > left else "candidate_worse" if right < left else "same"
                outcome[field][label] += 1
            if pair["boundary_expectation"] != "not_applicable":
                boundary["baseline_responses_with_any_failure"] += int(
                    any(review["boundary"] == "fail" for review in baseline["reviews"])
                )
                boundary["candidate_responses_with_any_failure"] += int(
                    any(review["boundary"] == "fail" for review in candidate["reviews"])
                )
                boundary["applicable_paired_trials"] += 1
            base_markers = baseline["analysis"]["style_markers_per_1000_words"]
            cand_markers = candidate["analysis"]["style_markers_per_1000_words"]
            marker_deltas.append(cand_markers - base_markers)
        return {
            "paired_trial_count": len(pairs),
            "rubric_pairwise_outcomes": outcome,
            "boundary_failures_are_reported_separately": boundary,
            "mean_style_marker_rate_delta_candidate_minus_baseline_per_1000_words": (
                round(sum(marker_deltas) / len(marker_deltas), 3) if marker_deltas else None
            ),
        }

    by_case = {
        case_id: summarize([pair for pair in manifest["pairs"] if pair["case_id"] == case_id])
        for case_id in dict.fromkeys(pair["case_id"] for pair in manifest["pairs"])
    }
    all_response_receipts = {
        blind_id: {
            "condition": result["condition"],
            "case_id": result["case_id"],
            "trial": result["trial"],
            **result["analysis"],
        }
        for blind_id, result in responses.items()
    }
    return {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "model": manifest["model"],
        "design": manifest["design"],
        "trial_count_per_case": manifest["trial_count_per_case"],
        "case_count": len(by_case),
        "paired_trial_count": len(manifest["pairs"]),
        "response_count": len(responses),
        "random_seed": manifest["random_seed"],
        "case_file_sha256": manifest["case_file_sha256"],
        "context_file_sha256": manifest["context_file_sha256"],
        "overall_descriptive_summary": summarize(manifest["pairs"]),
        "case_summaries": by_case,
        "interrater_agreement_cohen_kappa": {
            field: cohen_kappa(rows) for field, rows in interrater_rows.items()
        },
        "response_receipts": all_response_receipts,
        "interpretation_limit": manifest["interpretation_limit"] + " No significance test is performed.",
        "raw_response_text_included": False,
    }


def render_brief(report: dict[str, Any]) -> str:
    """Render a concise decision brief without promoting descriptive data to a claim."""
    overall = report["overall_descriptive_summary"]
    boundary = overall["boundary_failures_are_reported_separately"]
    lines = [
        "# Nerf Tag Team evaluation brief",
        "",
        f"**Experiment:** `{report['experiment_id']}`  ",
        f"**Model:** `{report['model']['id']}` / `{report['model']['version']}`  ",
        f"**Design:** {report['case_count']} cases × {report['trial_count_per_case']} paired trials; randomized order; {report['response_count']} responses  ",
        f"**Settings:** {report['model']['settings']}  ",
        f"**Seed:** `{report['random_seed']}`",
        "",
        "## Paired reviewer outcomes",
        "",
        "| Dimension | Candidate better | Same | Candidate worse |",
        "|---|---:|---:|---:|",
    ]
    for field in ("task_completion", "factual_accuracy", "tone_fit"):
        values = overall["rubric_pairwise_outcomes"][field]
        label = field.replace("_", " ").title()
        lines.append(
            f"| {label} | {values['candidate_better']} | {values['same']} | {values['candidate_worse']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary results",
            "",
            f"Applicable paired trials: **{boundary['applicable_paired_trials']}**. ",
            f"Baseline responses with a reviewer-reported failure: **{boundary['baseline_responses_with_any_failure']}**. ",
            f"Candidate responses with a reviewer-reported failure: **{boundary['candidate_responses_with_any_failure']}**.",
            "",
            (
                "**Boundary failures observed:** investigate the failed case before adopting the context."
                if boundary["candidate_responses_with_any_failure"]
                else "**No candidate boundary failures were reported in this sample.** This does not prove the boundary holds generally."
            ),
            "",
            "## Secondary tone diagnostic",
            "",
            f"Mean candidate-minus-baseline tracked-marker rate: **{overall['mean_style_marker_rate_delta_candidate_minus_baseline_per_1000_words']} per 1,000 words**. Negative means fewer tracked phrases. This lexical count is not a quality score.",
            "",
            "## Reviewer agreement",
            "",
            "| Dimension | Double-scored responses | Agreement | Cohen's κ |",
            "|---|---:|---:|---:|",
        ]
    )
    for field, values in report["interrater_agreement_cohen_kappa"].items():
        n = values["n"]
        agreement = "n/a" if values["observed_agreement"] is None else f"{values['observed_agreement']:.0%}"
        kappa = "n/a" if values["kappa"] is None else f"{values['kappa']:.3f}"
        lines.append(f"| {field.replace('_', ' ').title()} | {n} | {agreement} | {kappa} |")
    lines.extend(
        [
            "",
            "## Limits",
            "",
            report["interpretation_limit"],
            "",
            "This brief contains aggregate ratings, hashes, and counts. Raw response text is excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Prepare matched tone tests and audit saved responses offline."
    )
    subparsers = root.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="print or save one side of a matched prompt")
    prepare.add_argument("--case", required=True, help="case ID from lab/cases.json")
    prepare.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    prepare.add_argument("--output", help="new file path; existing files are never overwritten")

    sanitize = subparsers.add_parser("sanitize", help="prepare a style-only rewrite prompt")
    sanitize.add_argument("--input", required=True, type=Path, help="saved response text")
    sanitize.add_argument("--output", help="new file path; existing files are never overwritten")

    audit = subparsers.add_parser("audit", help="compare two saved responses without an LLM judge")
    audit.add_argument("--case", required=True)
    audit.add_argument("--baseline", required=True, type=Path)
    audit.add_argument("--candidate", required=True, type=Path)
    audit.add_argument("--model-id", required=True)
    audit.add_argument("--model-version", required=True)
    audit.add_argument("--settings", default="not-recorded")
    audit.add_argument(
        "--task-review", choices=("yes", "no", "not-reviewed"), default="not-reviewed"
    )
    audit.add_argument(
        "--boundary-review",
        choices=("yes", "no", "not-applicable", "not-reviewed"),
        default="not-reviewed",
    )
    audit.add_argument("--output", help="new JSON receipt path; existing files are never overwritten")

    plan = subparsers.add_parser("plan", help="create randomized repeated prompt pairs for a blinded evaluation")
    plan.add_argument("--model-id", required=True)
    plan.add_argument("--model-version", required=True)
    plan.add_argument("--settings", required=True, help="record temperature, tools, context limit, and other fixed settings")
    plan.add_argument("--trials", type=int, default=3, help="paired trials per case (minimum 2)")
    plan.add_argument("--seed", type=int, default=20261031, help="seed for reproducible condition order")
    plan.add_argument("--case", action="append", dest="selected_cases", help="case ID; repeat to select multiple (default: all)")
    plan.add_argument("--output-dir", required=True, type=Path, help="new private experiment directory; existing paths are never overwritten")

    record = subparsers.add_parser("record", help="copy a saved model response into its private blind-ID slot")
    record.add_argument("--manifest", required=True, type=Path)
    record.add_argument("--blind-id", required=True)
    record.add_argument("--input", required=True, type=Path, help="saved raw response text")

    score = subparsers.add_parser("score", help="score a completed blinded paired evaluation")
    score.add_argument("--manifest", required=True, type=Path)
    score.add_argument("--reviews", required=True, type=Path, help="completed two-reviewer JSON based on review-template.json")
    score.add_argument("--output", help="new JSON receipt path; existing files are never overwritten")
    score.add_argument("--markdown", type=Path, help="optional new aggregate decision brief path")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        cases = load_cases()
        if args.command == "prepare":
            if args.case not in cases:
                raise LabError(f"Unknown case {args.case!r}. Available: {', '.join(cases)}")
            result = build_prompt(cases[args.case], args.variant)
            emit(json.dumps(result, ensure_ascii=False, indent=2) + "\n", args.output)
            return 0

        if args.command == "sanitize":
            response = read_response(args.input)
            emit(build_sanitize_prompt(response), args.output)
            return 0

        if args.command == "audit":
            if args.case not in cases:
                raise LabError(f"Unknown case {args.case!r}. Available: {', '.join(cases)}")
            report = compare_responses(
                cases[args.case],
                read_response(args.baseline),
                read_response(args.candidate),
                args.model_id,
                args.model_version,
                args.settings,
                args.task_review,
                args.boundary_review,
            )
            emit(json.dumps(report, ensure_ascii=False, indent=2) + "\n", args.output)
            return 0

        if args.command == "plan":
            manifest = plan_experiment(
                cases,
                args.model_id,
                args.model_version,
                args.settings,
                args.trials,
                args.seed,
                args.selected_cases,
                args.output_dir,
            )
            print(f"Created {len(manifest['pairs'])} paired trials for {len(set(pair['case_id'] for pair in manifest['pairs']))} cases.")
            print(f"Experiment manifest: {args.output_dir / 'manifest.json'}")
            print(f"Blind review template: {args.output_dir / 'review-template.json'}")
            print("Two reviewers are required for every response; no model was called.")
            return 0

        if args.command == "record":
            target = record_response(args.manifest, args.blind_id, args.input)
            print(f"Recorded response privately: {target}")
            return 0

        if args.command == "score":
            report = score_experiment(args.manifest, args.reviews)
            if args.markdown:
                write_exclusive(args.markdown, render_brief(report))
            emit(json.dumps(report, ensure_ascii=False, indent=2) + "\n", args.output)
            return 0

        raise LabError(f"Unsupported command: {args.command}")
    except LabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

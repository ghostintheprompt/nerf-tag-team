#!/usr/bin/env python3
"""Prepare and audit matched tone tests without calling a model or network."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
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

    if payload.get("schema_version") != 1 or not isinstance(payload.get("cases"), list):
        raise LabError("Unsupported or malformed case file.")

    cases: dict[str, dict[str, Any]] = {}
    for case in payload["cases"]:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in cases:
            raise LabError("Case IDs must be non-empty and unique.")
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            raise LabError(f"Case {case_id!r} has no prompt.")
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

        raise LabError(f"Unsupported command: {args.command}")
    except LabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

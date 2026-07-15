#!/usr/bin/env python3
"""Run the deterministic evaluation-only lexical BM25 candidate comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rag_kb.adapters.parser.plain_text import process_plain_text
from rag_kb.domain import ParserSource


TOOL_VERSION = "1.0"
DEFAULT_CONFIG = Path("evaluation/configs/lexical-comparison-v1.0.json")
DEFAULT_OUTPUT = Path(
    "evaluation/reports/lexical-comparison-synthetic-v1-v1.0.json"
)
IDENTIFIER_SEPARATORS = frozenset("-./+")
HAN_MODES = frozenset(
    {"contiguous_runs", "unigrams", "overlapping_unigrams_and_bigrams"}
)


@dataclass(frozen=True, slots=True)
class LexicalChunk:
    sample_id: str
    ordinal: int
    text: str


@dataclass(frozen=True, slots=True)
class RankedChunk:
    sample_id: str
    ordinal: int
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail unless the checked-in output is byte-for-byte current",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _is_han(char: str) -> bool:
    return "\u3400" <= char <= "\u4dbf" or "\u4e00" <= char <= "\u9fff"


def _is_latin_or_digit(char: str) -> bool:
    if char.isdecimal():
        return True
    return unicodedata.name(char, "").startswith("LATIN ")


def tokenize(text: str, han_mode: str) -> tuple[str, ...]:
    """Tokenize NFC text while retaining identifier-internal punctuation."""

    if han_mode not in HAN_MODES:
        raise ValueError(f"unsupported han_mode: {han_mode}")
    normalized = unicodedata.normalize("NFC", text)
    tokens: list[str] = []
    index = 0
    while index < len(normalized):
        char = normalized[index]
        if _is_han(char):
            end = index + 1
            while end < len(normalized) and _is_han(normalized[end]):
                end += 1
            run = normalized[index:end]
            if han_mode == "contiguous_runs":
                tokens.append(run)
            else:
                tokens.extend(run)
                if han_mode == "overlapping_unigrams_and_bigrams":
                    tokens.extend(
                        run[position : position + 2]
                        for position in range(len(run) - 1)
                    )
            index = end
            continue
        if _is_latin_or_digit(char):
            end = index + 1
            while end < len(normalized):
                candidate = normalized[end]
                if _is_latin_or_digit(candidate):
                    end += 1
                    continue
                if (
                    candidate in IDENTIFIER_SEPARATORS
                    and end + 1 < len(normalized)
                    and _is_latin_or_digit(normalized[end + 1])
                ):
                    end += 1
                    continue
                break
            tokens.append(normalized[index:end].casefold())
            index = end
            continue
        index += 1
    return tuple(tokens)


class BM25Index:
    def __init__(
        self,
        chunks: tuple[LexicalChunk, ...],
        *,
        han_mode: str,
        k1: float,
        b: float,
    ) -> None:
        if not chunks:
            raise ValueError("lexical index requires at least one chunk")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 parameters are outside supported bounds")
        self._chunks = chunks
        self._han_mode = han_mode
        self._k1 = k1
        self._b = b
        self._term_counts = tuple(Counter(tokenize(chunk.text, han_mode)) for chunk in chunks)
        self._lengths = tuple(sum(counts.values()) for counts in self._term_counts)
        self._average_length = sum(self._lengths) / len(self._lengths)
        document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            document_frequency.update(counts.keys())
        self._document_frequency = document_frequency

    def search(self, query: str, *, top_k: int) -> tuple[RankedChunk, ...]:
        query_terms = tuple(dict.fromkeys(tokenize(query, self._han_mode)))
        ranked: list[RankedChunk] = []
        for chunk, counts, length in zip(
            self._chunks,
            self._term_counts,
            self._lengths,
            strict=True,
        ):
            score = sum(
                self._term_score(term, counts[term], length)
                for term in query_terms
                if counts[term]
            )
            if score > 0:
                ranked.append(RankedChunk(chunk.sample_id, chunk.ordinal, score))
        ranked.sort(key=lambda item: (-item.score, item.sample_id, item.ordinal))
        return tuple(ranked[:top_k])

    def _term_score(self, term: str, frequency: int, length: int) -> float:
        document_count = len(self._chunks)
        document_frequency = self._document_frequency[term]
        inverse_frequency = math.log(
            1.0
            + (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )
        normalization = 1.0 - self._b + self._b * length / self._average_length
        return inverse_frequency * (
            frequency * (self._k1 + 1.0)
            / (frequency + self._k1 * normalization)
        )


def _resolve_input(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"input path escapes repository root: {value}")
    return path


def _load_cases(path: Path) -> tuple[dict[str, Any], ...]:
    cases = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    case_ids = [case.get("case_id") for case in cases]
    if not cases or len(case_ids) != len(set(case_ids)):
        raise ValueError("golden dataset must contain unique case IDs")
    return cases


def _eligible(entry: dict[str, Any], filters: dict[str, Any]) -> bool:
    return (
        entry["lifecycle_state"] == filters["lifecycle_state"]
        and entry["is_current_version"] is filters["current_document_version"]
        and entry["build_status"] == filters["build_status"]
        and entry["serving_status"] == filters["serving_status"]
    )


def _load_chunks(
    manifest_path: Path,
    parser_config: dict[str, Any],
    filters: dict[str, Any],
) -> tuple[tuple[LexicalChunk, ...], tuple[str, ...]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("unsupported corpus manifest schema")
    chunks: list[LexicalChunk] = []
    eligible_sample_ids: list[str] = []
    manifest_root = manifest_path.parent.resolve()
    for entry in manifest["documents"]:
        document_path = (manifest_root / entry["path"]).resolve()
        if document_path != manifest_root and manifest_root not in document_path.parents:
            raise ValueError(f"document path escapes corpus root: {entry['path']}")
        content = document_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {entry['sample_id']}")
        if not _eligible(entry, filters):
            continue
        eligible_sample_ids.append(entry["sample_id"])
        suffix = document_path.suffix.lower()
        media_type = "text/markdown" if suffix == ".md" else "text/plain"
        processed = process_plain_text(
            ParserSource(document_path.name, media_type, content),
            max_characters=parser_config["max_characters"],
            overlap_characters=parser_config["overlap_characters"],
            max_chunks=parser_config["max_chunks_per_document"],
        )
        chunks.extend(
            LexicalChunk(entry["sample_id"], draft.ordinal, draft.text)
            for draft in processed.chunks
        )
    if not chunks:
        raise ValueError("mandatory filters produced no lexical chunks")
    return tuple(chunks), tuple(sorted(eligible_sample_ids))


def _validate_case_filters(cases: Iterable[dict[str, Any]]) -> str:
    workspace_ids: set[str] = set()
    for case in cases:
        filters = case["retrieval"]["filters"]
        if (
            filters.get("current_only") is not True
            or filters.get("build_status") != "ready"
            or filters.get("serving_status") != "serving"
            or not filters.get("workspace_id")
        ):
            raise ValueError(f"{case['case_id']}: mandatory filters are not frozen")
        workspace_ids.add(filters["workspace_id"])
    if len(workspace_ids) != 1:
        raise ValueError("lexical comparison requires one consistent fixture workspace")
    return next(iter(workspace_ids))


def _case_metrics(
    cases: tuple[dict[str, Any], ...],
    results: dict[str, tuple[RankedChunk, ...]],
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    answerable = [
        case
        for case in cases
        if case["retrieval"]["expected_relevant_sample_ids"]
    ]

    def recall(case: dict[str, Any], top_k: int) -> float:
        expected = set(case["retrieval"]["expected_relevant_sample_ids"])
        represented = {item.sample_id for item in results[case["case_id"]][:top_k]}
        return len(expected & represented) / len(expected)

    reciprocal_ranks: list[float] = []
    for case in answerable:
        expected = set(case["retrieval"]["expected_relevant_sample_ids"])
        rank = next(
            (
                index
                for index, item in enumerate(results[case["case_id"]], start=1)
                if item.sample_id in expected
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)

    forbidden_count = sum(
        item.sample_id in set(case["retrieval"]["forbidden_sample_ids"])
        for case in cases
        for item in results[case["case_id"]]
    )
    expected_empty = [case for case in cases if case["retrieval"]["expected_empty"]]
    safe_expected_empty = sum(
        not any(
            item.sample_id in set(case["retrieval"]["forbidden_sample_ids"])
            for item in results[case["case_id"]]
        )
        for case in expected_empty
    )
    return {
        "case_count": len(cases),
        "answerable_case_count": len(answerable),
        "recall_at_k": {
            str(top_k): round(
                sum(recall(case, top_k) for case in answerable) / len(answerable),
                6,
            )
            if answerable
            else None
            for top_k in top_k_values
        },
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 6)
        if reciprocal_ranks
        else None,
        "physical_empty_rate": round(
            sum(not results[case["case_id"]] for case in cases) / len(cases),
            6,
        )
        if cases
        else None,
        "false_empty_rate": round(
            sum(not results[case["case_id"]] for case in answerable)
            / len(answerable),
            6,
        )
        if answerable
        else None,
        "expected_empty_filter_accuracy": round(
            safe_expected_empty / len(expected_empty),
            6,
        )
        if expected_empty
        else None,
        "forbidden_result_count": forbidden_count,
    }


def _segments(
    cases: tuple[dict[str, Any], ...],
    results: dict[str, tuple[RankedChunk, ...]],
    top_k_values: tuple[int, ...],
    case_groups: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    languages = sorted({case["language"] for case in cases})
    tags = sorted({tag for case in cases for tag in case["tags"]})
    cases_by_id = {case["case_id"]: case for case in cases}
    return {
        "language": {
            language: _case_metrics(
                tuple(case for case in cases if case["language"] == language),
                results,
                top_k_values,
            )
            for language in languages
        },
        "tag": {
            tag: _case_metrics(
                tuple(case for case in cases if tag in case["tags"]),
                results,
                top_k_values,
            )
            for tag in tags
        },
        "case_group": {
            group: _case_metrics(
                tuple(cases_by_id[case_id] for case_id in case_ids),
                results,
                top_k_values,
            )
            for group, case_ids in sorted(case_groups.items())
        },
    }


def _candidate_report(
    candidate: dict[str, Any],
    chunks: tuple[LexicalChunk, ...],
    cases: tuple[dict[str, Any], ...],
    *,
    top_k_values: tuple[int, ...],
    case_groups: dict[str, tuple[str, ...]],
    k1: float,
    b: float,
) -> dict[str, Any]:
    index = BM25Index(chunks, han_mode=candidate["han_mode"], k1=k1, b=b)
    max_top_k = max(top_k_values)
    results = {
        case["case_id"]: index.search(case["question"], top_k=max_top_k)
        for case in cases
    }
    return {
        "strategy_id": candidate["strategy_id"],
        "tokenization": {
            "unicode_normalization": "NFC",
            "han_mode": candidate["han_mode"],
            "latin_digits": candidate["latin_digits"],
            "identifier_internal_separators": "-./+",
            "stop_words": "none",
        },
        "overall": _case_metrics(cases, results, top_k_values),
        "segments": _segments(cases, results, top_k_values, case_groups),
        "cases": [
            {
                "case_id": case["case_id"],
                "results": [
                    {
                        "rank": rank,
                        "sample_id": item.sample_id,
                        "chunk_ordinal": item.ordinal,
                        "score": round(item.score, 8),
                    }
                    for rank, item in enumerate(results[case["case_id"]], start=1)
                ],
            }
            for case in cases
        ],
    }


def _winner(candidates: list[dict[str, Any]]) -> str:
    def key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        overall = candidate["overall"]
        segments = candidate["segments"]
        language_recalls = [
            metrics["recall_at_k"]["5"]
            for metrics in segments["language"].values()
            if metrics["answerable_case_count"]
        ]
        exact_identifier = segments["tag"]["exact_identifier"]["recall_at_k"]["5"]
        return (
            overall["forbidden_result_count"],
            -overall["recall_at_k"]["5"],
            -overall["mrr"],
            -exact_identifier,
            -min(language_recalls),
            candidate["strategy_id"],
        )

    return min(candidates, key=key)["strategy_id"]


def _minimum_language_recall_at_5(candidate: dict[str, Any]) -> float:
    recalls = [
        metrics["recall_at_k"]["5"]
        for metrics in candidate["segments"]["language"].values()
        if metrics["answerable_case_count"]
    ]
    if not recalls:
        raise ValueError("lexical comparison has no answerable language segment")
    return min(recalls)


def _confirmation(
    candidates: list[dict[str, Any]],
    *,
    selected_strategy_id: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    by_id = {candidate["strategy_id"]: candidate for candidate in candidates}
    selected = by_id[selected_strategy_id]
    best_overall_recall = max(
        candidate["overall"]["recall_at_k"]["5"] for candidate in candidates
    )
    best_exact_identifier_recall = max(
        candidate["segments"]["tag"]["exact_identifier"]["recall_at_k"]["5"]
        for candidate in candidates
    )
    best_minimum_language_recall = max(
        _minimum_language_recall_at_5(candidate) for candidate in candidates
    )
    criteria = [
        {
            "criterion": "forbidden_result_count_maximum",
            "selected_value": selected["overall"]["forbidden_result_count"],
            "required_value": policy["forbidden_result_count_maximum"],
            "passed": selected["overall"]["forbidden_result_count"]
            <= policy["forbidden_result_count_maximum"],
        },
        {
            "criterion": "best_overall_recall_at_5",
            "selected_value": selected["overall"]["recall_at_k"]["5"],
            "required_value": best_overall_recall,
            "passed": not policy["require_best_overall_recall_at_5"]
            or selected["overall"]["recall_at_k"]["5"] == best_overall_recall,
        },
        {
            "criterion": "best_exact_identifier_recall_at_5",
            "selected_value": selected["segments"]["tag"]["exact_identifier"][
                "recall_at_k"
            ]["5"],
            "required_value": best_exact_identifier_recall,
            "passed": not policy["require_best_exact_identifier_recall_at_5"]
            or selected["segments"]["tag"]["exact_identifier"]["recall_at_k"][
                "5"
            ]
            == best_exact_identifier_recall,
        },
        {
            "criterion": "best_minimum_language_recall_at_5",
            "selected_value": _minimum_language_recall_at_5(selected),
            "required_value": best_minimum_language_recall,
            "passed": not policy["require_best_minimum_language_recall_at_5"]
            or _minimum_language_recall_at_5(selected)
            == best_minimum_language_recall,
        },
    ]
    diagnostic_winner = _winner(candidates)
    return {
        "frozen_selected_strategy_id": selected_strategy_id,
        "diagnostic_winner_strategy_id": diagnostic_winner,
        "diagnostic_ranking_precedence": policy["diagnostic_ranking_precedence"],
        "confirmation_criteria": criteria,
        "status": "confirmed" if all(item["passed"] for item in criteria) else "conflict",
        "diagnostic_advantage_recorded": diagnostic_winner != selected_strategy_id,
    }


def build_report(config_path: Path, *, repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != "1.0"
        or config.get("serving_enabled") is not False
    ):
        raise ValueError("lexical comparison config must be schema 1.0 and evaluation-only")
    expected_filters = {
        "workspace": True,
        "active_revision": True,
        "lifecycle_state": "active",
        "current_document_version": True,
        "build_status": "ready",
        "serving_status": "serving",
    }
    if config.get("mandatory_filters") != expected_filters:
        raise ValueError("lexical comparison mandatory filters are not frozen")
    expected_confirmation = {
        "forbidden_result_count_maximum": 0,
        "require_best_overall_recall_at_5": True,
        "require_best_exact_identifier_recall_at_5": True,
        "require_best_minimum_language_recall_at_5": True,
    }
    if config.get("frozen_confirmation") != expected_confirmation:
        raise ValueError("frozen lexical confirmation policy is not supported")
    candidate_ids = [candidate["strategy_id"] for candidate in config["candidates"]]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate strategy IDs must be unique")
    if config["selected_strategy_id"] not in candidate_ids:
        raise ValueError("selected strategy is not a configured candidate")
    expected_precedence = [
        "forbidden_result_count_ascending",
        "overall_recall_at_5_descending",
        "overall_mrr_descending",
        "exact_identifier_recall_at_5_descending",
        "minimum_language_recall_at_5_descending",
        "strategy_id_ascending",
    ]
    if config.get("diagnostic_ranking_precedence") != expected_precedence:
        raise ValueError("diagnostic ranking precedence is not supported")
    top_k_values = tuple(config["top_k_values"])
    if (
        top_k_values != tuple(sorted(set(top_k_values)))
        or not top_k_values
        or any(type(top_k) is not int or top_k <= 0 for top_k in top_k_values)
    ):
        raise ValueError("top_k_values must be sorted unique positive integers")
    corpus_path = _resolve_input(repository_root, config["corpus_manifest"])
    corpus_profile_path = _resolve_input(repository_root, config["corpus_profile"])
    dataset_path = _resolve_input(repository_root, config["golden_dataset"])
    golden_manifest_path = _resolve_input(repository_root, config["golden_manifest"])
    quality_baseline_path = _resolve_input(repository_root, config["quality_baseline"])
    report_schema_path = _resolve_input(repository_root, config["report_schema"])
    corpus_manifest = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus_profile = json.loads(corpus_profile_path.read_text(encoding="utf-8"))
    golden_manifest = json.loads(golden_manifest_path.read_text(encoding="utf-8"))
    quality_baseline = json.loads(quality_baseline_path.read_text(encoding="utf-8"))
    report_schema = json.loads(report_schema_path.read_text(encoding="utf-8"))
    if (
        corpus_profile.get("corpus_id") != corpus_manifest.get("corpus_id")
        or corpus_profile.get("manifest_sha256") != sha256(corpus_path)
        or golden_manifest.get("corpus_id") != corpus_manifest.get("corpus_id")
        or golden_manifest.get("corpus_manifest_sha256") != sha256(corpus_path)
        or golden_manifest.get("dataset_path") != config["golden_dataset"]
        or golden_manifest.get("dataset_sha256") != sha256(dataset_path)
        or golden_manifest.get("evaluation_config_path") != config["quality_baseline"]
        or golden_manifest.get("evaluation_config_sha256") != sha256(quality_baseline_path)
    ):
        raise ValueError("Stage 01 corpus, dataset, or quality baseline linkage failed")
    if (
        report_schema.get("properties", {}).get("schema_version", {}).get("const")
        != "1.0"
    ):
        raise ValueError("lexical comparison report schema is not version 1.0")
    selected_candidate = next(
        candidate
        for candidate in config["candidates"]
        if candidate["strategy_id"] == config["selected_strategy_id"]
    )
    frozen_lexical = next(
        (
            strategy
            for strategy in quality_baseline.get("strategies", [])
            if strategy.get("role") == "evaluation_only_lexical_comparison"
        ),
        None,
    )
    if (
        frozen_lexical is None
        or frozen_lexical.get("strategy_id") != config["selected_strategy_id"]
        or frozen_lexical.get("parameters") != config["bm25"]
        or frozen_lexical.get("tokenization", {}).get("han")
        != selected_candidate["han_mode"]
        or frozen_lexical.get("serving_enabled") is not False
    ):
        raise ValueError("selected lexical strategy differs from the Stage 01 baseline")
    cases = _load_cases(dataset_path)
    workspace_id = _validate_case_filters(cases)
    if golden_manifest.get("case_count") != len(cases):
        raise ValueError("golden manifest case count does not match the dataset")
    known_case_ids = {case["case_id"] for case in cases}
    case_groups = {
        group: tuple(case_ids)
        for group, case_ids in config["reported_case_groups"].items()
    }
    unknown_group_case_ids = {
        case_id
        for case_ids in case_groups.values()
        for case_id in case_ids
        if case_id not in known_case_ids
    }
    if unknown_group_case_ids:
        raise ValueError("reported case groups contain unknown case IDs")
    chunks, eligible_sample_ids = _load_chunks(
        corpus_path,
        config["parser"],
        config["mandatory_filters"],
    )
    candidates = [
        _candidate_report(
            candidate,
            chunks,
            cases,
            top_k_values=top_k_values,
            case_groups=case_groups,
            k1=config["bm25"]["k1"],
            b=config["bm25"]["b"],
        )
        for candidate in config["candidates"]
    ]
    selected = config["selected_strategy_id"]
    confirmation_policy = {
        **config["frozen_confirmation"],
        "diagnostic_ranking_precedence": config["diagnostic_ranking_precedence"],
    }
    return {
        "schema_version": "1.0",
        "comparison_id": config["comparison_id"],
        "tool": {"name": "tools/lexical_comparison.py", "version": TOOL_VERSION},
        "inputs": {
            "config": str(config_path.relative_to(repository_root)),
            "config_sha256": sha256(config_path),
            "corpus_manifest": config["corpus_manifest"],
            "corpus_manifest_sha256": sha256(corpus_path),
            "corpus_profile": config["corpus_profile"],
            "corpus_profile_sha256": sha256(corpus_profile_path),
            "golden_dataset": config["golden_dataset"],
            "golden_dataset_sha256": sha256(dataset_path),
            "golden_manifest": config["golden_manifest"],
            "golden_manifest_sha256": sha256(golden_manifest_path),
            "quality_baseline": config["quality_baseline"],
            "quality_baseline_sha256": sha256(quality_baseline_path),
            "report_schema": config["report_schema"],
            "report_schema_sha256": sha256(report_schema_path),
            "golden_case_count": len(cases),
            "eligible_sample_ids": list(eligible_sample_ids),
            "eligible_chunk_count": len(chunks),
        },
        "configuration": {
            "parser": config["parser"],
            "mandatory_filters": config["mandatory_filters"],
            "fixture_workspace_id": workspace_id,
            "top_k_values": list(top_k_values),
            "reported_case_groups": {
                group: list(case_ids) for group, case_ids in sorted(case_groups.items())
            },
            "bm25": config["bm25"],
            "serving_enabled": False,
        },
        "candidates": candidates,
        "selection": _confirmation(
            candidates,
            selected_strategy_id=selected,
            policy=confirmation_policy,
        ),
    }


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_path = args.output if args.output.is_absolute() else root / args.output
    try:
        report = build_report(config_path, repository_root=root)
        rendered = canonical_json(report)
        if report["selection"]["status"] != "confirmed":
            print("lexical comparison conflicts with the frozen selection", file=sys.stderr)
            return 1
        if args.check:
            if not output_path.exists() or output_path.read_text(encoding="utf-8") != rendered:
                print("lexical comparison report is not current", file=sys.stderr)
                return 1
            print("lexical comparison report is current")
            return 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        try:
            display_path = output_path.relative_to(root)
        except ValueError:
            display_path = output_path
        print(f"wrote {display_path}")
        return 0
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"lexical comparison error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

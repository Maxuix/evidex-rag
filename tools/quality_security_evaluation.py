#!/usr/bin/env python3
"""Evaluate P1A answer quality and security against reviewed deterministic inputs."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any
from uuid import UUID, uuid5

from rag_kb.adapters.parser.plain_text import process_plain_text
from rag_kb.answering import (
    AnswerGenerationStep,
    AnswerStructureValidationStep,
    EvidenceAssessmentStep,
)
from rag_kb.domain import (
    AnswerOutcome,
    ChatExecutionContext,
    ChatModelRequest,
    ChatModelResponse,
    ChatPipelineState,
    ChatRunLease,
    Evidence,
    EvidencePack,
    ParserSource,
    RetrievalStrategy,
)


TOOL_VERSION = "1.0"
DEFAULT_CONFIG = Path("evaluation/configs/quality-security-regression-v1.0.json")
NAMESPACE = UUID("f8ca394a-8818-5df2-a202-c0ca863191d3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the checked report and rerun stable quality/security results",
    )
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    values = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"{path} must contain JSON objects")
    return values


def _stable(value: str) -> UUID:
    return uuid5(NAMESPACE, value)


@dataclass(frozen=True, slots=True)
class Inputs:
    root: Path
    config_path: Path
    config: dict[str, Any]
    paths: dict[str, Path]
    cases: tuple[dict[str, Any], ...]
    responses: tuple[dict[str, Any], ...]
    probes: tuple[dict[str, Any], ...]
    corpus: dict[str, Any]
    profile: dict[str, Any]
    quality: dict[str, Any]
    provider: dict[str, Any]
    retrieval: dict[str, Any]


def load_inputs(config_path: Path, root: Path) -> Inputs:
    config = _json(config_path)
    keys = (
        "golden_dataset",
        "golden_manifest",
        "golden_responses",
        "security_probes",
        "corpus_manifest",
        "corpus_profile",
        "parser_implementation",
        "quality_baseline",
        "provider_declaration",
        "retrieval_report",
        "report_schema",
    )
    paths = {key: (root / config[key]).resolve() for key in keys}
    cases = _jsonl(paths["golden_dataset"])
    responses = _jsonl(paths["golden_responses"])
    probes = _jsonl(paths["security_probes"])
    corpus = _json(paths["corpus_manifest"])
    profile = _json(paths["corpus_profile"])
    quality = _json(paths["quality_baseline"])
    provider = _json(paths["provider_declaration"])
    retrieval = _json(paths["retrieval_report"])
    manifest = _json(paths["golden_manifest"])
    _json(paths["report_schema"])

    case_ids = [value.get("case_id") for value in cases]
    response_ids = [value.get("case_id") for value in responses]
    probe_ids = [value.get("probe_id") for value in probes]
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != set(response_ids):
        raise ValueError("golden cases and deterministic responses do not match")
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("deterministic response case IDs must be unique")
    if len(probe_ids) != len(set(probe_ids)) or len(probe_ids) < 4:
        raise ValueError("security probes must contain at least four unique cases")
    if manifest.get("dataset_sha256") != sha256(paths["golden_dataset"]):
        raise ValueError("golden manifest dataset checksum mismatch")
    if manifest.get("case_count") != len(cases):
        raise ValueError("golden manifest case count mismatch")
    if quality.get("dataset_id") != manifest.get("dataset_id"):
        raise ValueError("quality baseline dataset identity mismatch")
    if quality.get("corpus_profile_id") != manifest.get("corpus_profile_id"):
        raise ValueError("quality baseline CorpusProfile identity mismatch")
    if profile.get("manifest_sha256") != sha256(paths["corpus_manifest"]):
        raise ValueError("CorpusProfile measurements do not match the corpus manifest")
    if retrieval.get("inputs", {}).get("dataset_sha256") != sha256(
        paths["golden_dataset"]
    ):
        raise ValueError("retrieval report does not match the golden dataset")
    retrieval_ids = {value.get("case_id") for value in retrieval.get("case_results", [])}
    if retrieval_ids != set(case_ids):
        raise ValueError("retrieval report case coverage is incomplete")
    if any(
        value.get("strategy_id") != config["retrieval_strategy_id"]
        for value in retrieval.get("case_results", [])
    ):
        raise ValueError("retrieval report strategy identity mismatch")
    cases_by_id = {value["case_id"]: value for value in cases}
    for response in responses:
        case = cases_by_id[response["case_id"]]
        expected_samples = set(case["retrieval"]["expected_relevant_sample_ids"])
        if not set(response["usable_sample_ids"]) <= expected_samples:
            raise ValueError(
                f"{response['case_id']} response uses a non-relevant sample"
            )
        draft = response.get("draft")
        if draft is None:
            continue
        facts = {
            fact["fact_id"]: set(fact["acceptable_sample_ids"])
            for fact in case["answer"]["required_facts"]
        }
        for claim in draft["claims"]:
            fact_ids = claim["supported_fact_ids"]
            if not fact_ids or not set(fact_ids) <= set(facts):
                raise ValueError(
                    f"{response['case_id']} claim has invalid reviewed fact labels"
                )
            claim_samples = set(claim["sample_ids"])
            if not claim_samples or any(
                not claim_samples <= facts[fact_id] for fact_id in fact_ids
            ):
                raise ValueError(
                    f"{response['case_id']} claim/source support labels conflict"
                )
    document_root = paths["corpus_manifest"].parent
    for entry in corpus.get("documents", []):
        document = document_root / entry["path"]
        if sha256(document) != entry["sha256"]:
            raise ValueError(f"corpus document checksum mismatch: {entry['sample_id']}")
    return Inputs(
        root=root,
        config_path=config_path,
        config=config,
        paths=paths,
        cases=cases,
        responses=responses,
        probes=probes,
        corpus=corpus,
        profile=profile,
        quality=quality,
        provider=provider,
        retrieval=retrieval,
    )


class ScriptedModel:
    """Network-free adapter whose reviewed responses are independent report inputs."""

    def __init__(self, model: str, responses: tuple[str, ...]) -> None:
        self.model = model
        self.responses = list(responses)
        self.requests: list[ChatModelRequest] = []

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("deterministic adapter received an unexpected call")
        content = self.responses.pop(0)
        prompt_tokens = max(
            1, sum(len(message.content) for message in request.messages) // 4
        )
        completion_tokens = max(1, len(content) // 4)
        return ChatModelResponse(
            content=content,
            model=self.model,
            finish_reason="stop",
            provider_request_id=f"deterministic-{len(self.requests)}",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )


def _corpus_entries(inputs: Inputs) -> dict[str, dict[str, Any]]:
    return {entry["sample_id"]: entry for entry in inputs.corpus["documents"]}


def _retrieval_cases(inputs: Inputs) -> dict[str, dict[str, Any]]:
    return {entry["case_id"]: entry for entry in inputs.retrieval["case_results"]}


def _pack(
    inputs: Inputs,
    case_id: str,
    *,
    ranked_chunks: list[dict[str, Any]] | None = None,
) -> tuple[EvidencePack, dict[str, str]]:
    result = _retrieval_cases(inputs)[case_id]
    ranked = ranked_chunks if ranked_chunks is not None else result["ranked_chunks"][:5]
    entries = _corpus_entries(inputs)
    revision_id = UUID(inputs.retrieval["inputs"]["index_revision"]["id"])
    knowledge_base_id = UUID(inputs.retrieval["inputs"]["knowledge_base_id"])
    evidence: list[Evidence] = []
    sample_by_citation: dict[str, str] = {}
    document_root = inputs.paths["corpus_manifest"].parent
    parsed: dict[str, Any] = {}
    parser = inputs.retrieval["inputs"]["index_revision"]["parser"]
    for rank, item in enumerate(ranked, start=1):
        sample_id = item["sample_id"]
        entry = entries[sample_id]
        path = document_root / entry["path"]
        if sample_id not in parsed:
            media_type = (
                "text/markdown" if path.suffix.lower() == ".md" else "text/plain"
            )
            parsed[sample_id] = process_plain_text(
                ParserSource(path.name, media_type, path.read_bytes()),
                max_characters=parser["max_characters"],
                overlap_characters=parser["overlap_characters"],
                max_chunks=parser["max_chunks_per_document"],
            )
        try:
            chunk = parsed[sample_id].chunks[int(item["chunk_ordinal"])]
        except IndexError as error:
            raise ValueError(
                f"recorded chunk ordinal is outside the current parser result: {sample_id}"
            ) from error
        evidence.append(
            Evidence(
                rank=rank,
                index_chunk_id=UUID(item["index_chunk_id"]),
                indexed_document_version_id=_stable(f"indexed:{sample_id}"),
                document_id=_stable(f"document:{entry['logical_document_id']}"),
                document_version_id=_stable(
                    f"version:{entry['logical_document_id']}:{entry['version']}"
                ),
                index_revision_id=revision_id,
                ordinal=int(item["chunk_ordinal"]),
                text=chunk.text,
                source_location={
                    "sample_id": sample_id,
                    "chunk": item["chunk_ordinal"],
                    "start_character": chunk.start_character,
                    "end_character": chunk.end_character,
                },
                hierarchy={"headings": list(chunk.heading_hierarchy)},
                source_metadata={},
                score=float(item["score"]),
            )
        )
        sample_by_citation[f"cite_{rank}"] = sample_id
    return (
        EvidencePack(
            knowledge_base_id=knowledge_base_id,
            index_revision_id=revision_id,
            strategy=RetrievalStrategy.EXACT_VECTOR,
            evidence=tuple(evidence),
        ),
        sample_by_citation,
    )


def _context(inputs: Inputs, case_id: str, question: str, outcome: str) -> ChatExecutionContext:
    workspace_id = UUID(inputs.retrieval["inputs"]["workspace_id"])
    run_id = _stable(f"run:{case_id}")
    model = inputs.config["evaluation_adapter"]["identity"]
    lease = ChatRunLease(run_id, workspace_id, "quality-evaluator", 1, datetime.now(UTC))
    policy = dict(inputs.config["answer_policy"])
    policy["insufficiency_policy"] = (
        "partial_answer" if outcome == "partial" else "refuse"
    )
    return ChatExecutionContext(
        lease=lease,
        run_id=run_id,
        workspace_id=workspace_id,
        knowledge_base_id=UUID(inputs.retrieval["inputs"]["knowledge_base_id"]),
        session_id=_stable(f"session:{case_id}"),
        user_message_id=_stable(f"user:{case_id}"),
        assistant_message_id=_stable(f"assistant:{case_id}"),
        index_revision_id=UUID(inputs.retrieval["inputs"]["index_revision"]["id"]),
        principal_id="quality-evaluator",
        client_id="offline-regression",
        query=question,
        effective_policy=policy,
        retrieval_strategy={"strategy": "exact_vector", "top_k": 5, "rerank": False},
        model_configuration={"resolved_model": model},
        attempt=1,
    )


def _citation_for_sample(
    sample_by_citation: dict[str, str],
    sample_id: str,
    *,
    text_by_citation: dict[str, str] | None = None,
    passage_needles: tuple[str, ...] = (),
) -> str:
    fallback: str | None = None
    for citation_id, candidate in sample_by_citation.items():
        if candidate == sample_id:
            fallback = fallback or citation_id
            if passage_needles and text_by_citation is not None and any(
                needle in text_by_citation[citation_id] for needle in passage_needles
            ):
                return citation_id
    if fallback is not None and not passage_needles:
        return fallback
    raise ValueError(f"reviewed usable sample was not retrieved in top five: {sample_id}")


def _wire_responses(
    response: dict[str, Any],
    sample_by_citation: dict[str, str],
    pack: EvidencePack,
    case: dict[str, Any],
) -> tuple[str, ...]:
    text_by_citation = {
        f"cite_{item.rank}": item.text for item in pack.evidence
    }
    passage_needles = {
        item["sample_id"]: tuple(item["must_contain_any"])
        for item in case["required_evidence"]
    }
    citations = [
        _citation_for_sample(
            sample_by_citation,
            sample_id,
            text_by_citation=text_by_citation,
            passage_needles=passage_needles.get(sample_id, ()),
        )
        for sample_id in response["usable_sample_ids"]
    ]
    assessment = json.dumps(
        {
            "coverage": response["coverage"],
            "usable_citation_ids": citations,
            "supported_aspects": response["supported_aspects"],
            "missing_aspects": response["missing_aspects"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    draft = response.get("draft")
    if draft is None:
        return (assessment,)
    claims = [
        {
            "text": claim["text"],
            "citation_ids": [
                _citation_for_sample(
                    sample_by_citation,
                    sample_id,
                    text_by_citation=text_by_citation,
                    passage_needles=passage_needles.get(sample_id, ()),
                )
                for sample_id in claim["sample_ids"]
            ],
        }
        for claim in draft["claims"]
    ]
    return (
        assessment,
        json.dumps(
            {
                "outcome": draft["outcome"],
                "claims": claims,
                "missing_aspects": draft["missing_aspects"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


async def _run_pipeline(
    context: ChatExecutionContext,
    pack: EvidencePack,
    model: ScriptedModel,
) -> ChatPipelineState:
    state = ChatPipelineState(context=context, evidence_pack=pack)
    state = await EvidenceAssessmentStep(model).run(state)
    state = await AnswerGenerationStep(model).run(state)
    return await AnswerStructureValidationStep(model).run(state)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 1.0


def _latency(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    ordered = sorted(values)

    def percentile(value: float) -> float:
        index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * value + 0.999999)))
        return round(ordered[index], 3)

    return {
        "count": len(values),
        "p50": round(statistics.median(ordered), 3),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "maximum": round(max(ordered), 3),
    }


async def evaluate_cases(inputs: Inputs) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    responses = {value["case_id"]: value for value in inputs.responses}
    results: list[dict[str, Any]] = []
    substantive_claims = supported_claims = claims_with_citations = 0
    citations = valid_citations = 0
    refused_total = refused_correct = partial_total = partial_correct = 0
    prompt_tokens = completion_tokens = total_tokens = 0
    latencies: list[float] = []
    failures: list[str] = []
    for case in inputs.cases:
        case_id = case["case_id"]
        response = responses[case_id]
        pack, sample_by_citation = _pack(inputs, case_id)
        context = _context(
            inputs, case_id, case["question"], case["answer"]["expected_outcome"]
        )
        model = ScriptedModel(
            inputs.config["evaluation_adapter"]["identity"],
            _wire_responses(response, sample_by_citation, pack, case),
        )
        started = time.perf_counter()
        state = await _run_pipeline(context, pack, model)
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)
        answering = state.answering
        if answering is None or answering.rendered is None or answering.validated is None:
            raise AssertionError(f"{case_id} did not produce a validated result")
        expected = case["answer"]["expected_outcome"]
        actual = answering.rendered.outcome.value
        passed = actual == expected
        if expected == "refused":
            refused_total += 1
            refused_correct += int(actual == "refused" and not answering.rendered.citations)
        if expected == "partial":
            partial_total += 1
            partial_correct += int(
                actual == "partial"
                and set(answering.validated.missing_aspects)
                == set(case["answer"]["missing_aspects"])
            )
        expected_fact_ids = {
            fact["fact_id"] for fact in case["answer"]["required_facts"]
        }
        labelled_fact_ids = {
            fact_id
            for claim in (response.get("draft") or {}).get("claims", [])
            for fact_id in claim["supported_fact_ids"]
        }
        rendered_claims = answering.validated.claims
        substantive_claims += len(rendered_claims)
        claims_with_citations += sum(bool(claim.citation_ids) for claim in rendered_claims)
        reviewed_claims = (response.get("draft") or {}).get("claims", [])
        rendered_citations = {
            item.citation_id: item for item in answering.rendered.citations
        }
        traceable = actual == "refused" or all(
            any(
                sample_by_citation.get(citation_id) == requirement["sample_id"]
                and any(
                    needle in rendered_citations[citation_id].quoted_text
                    for needle in requirement["must_contain_any"]
                )
                for citation_id in rendered_citations
            )
            for requirement in case["required_evidence"]
        )
        case_supported = (
            labelled_fact_ids == expected_fact_ids
            and len(reviewed_claims) == len(rendered_claims)
            and traceable
            and all(
                claim["supported_fact_ids"]
                and set(claim["supported_fact_ids"]) <= expected_fact_ids
                for claim in reviewed_claims
            )
        )
        supported_claims += len(rendered_claims) if case_supported else 0
        envelope_ids = answering.evidence.citation_ids
        for item in answering.rendered.citations:
            citations += 1
            if item.citation_id in envelope_ids:
                valid_citations += 1
        rendered_text = answering.rendered.content
        if any(value in rendered_text for value in case["answer"]["forbidden_claims"]):
            passed = False
        if not passed:
            failures.append(case_id)
        for call in answering.model_calls:
            prompt_tokens += int(call.usage.get("prompt_tokens", 0))
            completion_tokens += int(call.usage.get("completion_tokens", 0))
            total_tokens += int(call.usage.get("total_tokens", 0))
        results.append(
            {
                "case_id": case_id,
                "expected_outcome": expected,
                "actual_outcome": actual,
                "claim_count": len(rendered_claims),
                "citation_count": len(answering.rendered.citations),
                "model_call_count": len(answering.model_calls),
                "matched_fact_ids": sorted(labelled_fact_ids & expected_fact_ids),
                "evidence_traceability": traceable,
                "latency_ms": round(elapsed, 3),
                "passed": passed,
            }
        )
    metrics = {
        "citation_identifier_validity": _ratio(valid_citations, citations),
        "structural_claim_coverage": _ratio(claims_with_citations, substantive_claims),
        "semantic_support_rate": _ratio(supported_claims, substantive_claims),
        "unsupported_claim_rate": _ratio(
            substantive_claims - supported_claims, substantive_claims
        ),
        "refusal_accuracy": _ratio(refused_correct, refused_total),
        "partial_answer_accuracy": _ratio(partial_correct, partial_total),
        "label_source": "version-controlled human-reviewed synthetic fact labels",
        "latency": _latency(latencies),
        "usage": {
            "chat_prompt_tokens": prompt_tokens,
            "chat_completion_tokens": completion_tokens,
            "chat_total_tokens": total_tokens,
        },
        "failures": failures,
    }
    return results, metrics


def _probe_pack(inputs: Inputs, probe_id: str, excerpt: str) -> EvidencePack:
    revision_id = UUID(inputs.retrieval["inputs"]["index_revision"]["id"])
    return EvidencePack(
        knowledge_base_id=UUID(inputs.retrieval["inputs"]["knowledge_base_id"]),
        index_revision_id=revision_id,
        strategy=RetrievalStrategy.EXACT_VECTOR,
        evidence=(
            Evidence(
                rank=1,
                index_chunk_id=_stable(f"probe-chunk:{probe_id}"),
                indexed_document_version_id=_stable(f"probe-indexed:{probe_id}"),
                document_id=_stable(f"probe-document:{probe_id}"),
                document_version_id=_stable(f"probe-version:{probe_id}"),
                index_revision_id=revision_id,
                ordinal=0,
                text=excerpt,
                source_location={"probe_id": probe_id},
                hierarchy={},
                source_metadata={},
                score=1.0,
            ),
        ),
    )


async def evaluate_security(inputs: Inputs) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    query_plans = [value["query_plan"] for value in inputs.retrieval["case_results"]]
    filter_checks = {
        "revision_selector": all(
            plan.get("revision_selector") == "active" for plan in query_plans
        ),
        "current_document_version_only": all(
            plan.get("current_document_version_only") is True for plan in query_plans
        ),
        "build_status": all(
            plan.get("build_status") == "ready" for plan in query_plans
        ),
        "serving_status": all(
            plan.get("serving_status") == "serving" for plan in query_plans
        ),
        "forbidden_result_count": all(
            strategy["overall"]["forbidden_result_count"] == 0
            for strategy in inputs.retrieval["strategies"]
        ),
    }
    for probe in inputs.probes:
        probe_id = probe["probe_id"]
        pack = _probe_pack(inputs, probe_id, probe["untrusted_excerpt"])
        context = _context(inputs, f"probe:{probe_id}", probe["question"], "refused")
        if probe_id == "citation-authority":
            assessment = json.dumps(
                {
                    "coverage": "sufficient",
                    "usable_citation_ids": ["cite_1"],
                    "supported_aspects": ["safe behavior"],
                    "missing_aspects": [],
                }
            )
            hostile = json.dumps(
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "RAW_DRAFT_CANARY",
                            "citation_ids": ["citation-fake-999"],
                        }
                    ],
                    "missing_aspects": [],
                }
            )
            responses = (assessment, hostile, hostile)
        else:
            responses = (
                json.dumps(
                    {
                        "coverage": "none",
                        "usable_citation_ids": [],
                        "supported_aspects": [],
                        "missing_aspects": [],
                    }
                ),
            )
        model = ScriptedModel(inputs.config["evaluation_adapter"]["identity"], responses)
        state = await _run_pipeline(context, pack, model)
        answering = state.answering
        if answering is None or answering.rendered is None:
            raise AssertionError(f"security probe did not render: {probe_id}")
        request_text = "\n".join(
            message.content for request in model.requests for message in request.messages
        )
        output = answering.rendered.content
        checks = {
            "document_labelled_untrusted": "untrusted data" in request_text,
            "no_tool_or_credential_authority": (
                "no tools" in request_text and "credentials" in request_text
            ),
            "request_contract_has_messages_only": set(ChatModelRequest.__slots__)
            == {"messages"},
            "forbidden_canaries_not_in_output": not any(
                value in output for value in probe["forbidden_canaries"]
            ),
            "no_citations_on_safe_refusal": not answering.rendered.citations,
            "server_filters_unchanged": all(filter_checks.values()),
        }
        results.append(
            {
                "probe_id": probe_id,
                "expected": probe["expected"],
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
            }
        )
    return results


def _gate(metrics: dict[str, Any], security: list[dict[str, Any]], inputs: Inputs) -> dict[str, Any]:
    thresholds = inputs.quality["acceptance_thresholds"]
    checks = {
        "citation_identifier_validity": metrics["citation_identifier_validity"]
        >= thresholds["citation_identifier_validity_minimum"],
        "structural_claim_coverage": metrics["structural_claim_coverage"]
        >= thresholds["structural_claim_coverage_minimum"],
        "semantic_support_rate": metrics["semantic_support_rate"]
        >= thresholds["semantic_support_rate_minimum"],
        "unsupported_claim_rate": metrics["unsupported_claim_rate"]
        <= thresholds["unsupported_claim_rate_maximum"],
        "refusal_accuracy": metrics["refusal_accuracy"]
        >= thresholds["refusal_accuracy_minimum"],
        "partial_answer_accuracy": metrics["partial_answer_accuracy"]
        >= thresholds["partial_answer_accuracy_minimum"],
        "malicious_instruction_bypass_count": sum(
            result["status"] != "passed" for result in security
        )
        <= thresholds["malicious_instruction_bypass_count_maximum"],
        "deterministic_evaluation_failure_rate": (
            len(metrics["failures"]) / len(inputs.cases)
        )
        <= thresholds["deterministic_evaluation_failure_rate_maximum"],
        "answer_pipeline_p95_ms": metrics["latency"]["p95"]
        <= thresholds["answer_pipeline_p95_ms_maximum"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "thresholds": {
            key: value
            for key, value in thresholds.items()
            if key
            in {
                "citation_identifier_validity_minimum",
                "structural_claim_coverage_minimum",
                "semantic_support_rate_minimum",
                "unsupported_claim_rate_maximum",
                "refusal_accuracy_minimum",
                "partial_answer_accuracy_minimum",
                "malicious_instruction_bypass_count_maximum",
                "deterministic_evaluation_failure_rate_maximum",
                "answer_pipeline_p95_ms_maximum",
            }
        },
    }


async def build_report(inputs: Inputs) -> dict[str, Any]:
    started = datetime.now(UTC)
    case_results, metrics = await evaluate_cases(inputs)
    security_results = await evaluate_security(inputs)
    finished = datetime.now(UTC)
    recorded = {
        key: {
            "path": str(path.relative_to(inputs.root)),
            "sha256": sha256(path),
        }
        for key, path in inputs.paths.items()
    }
    recorded["evaluation_config"] = {
        "path": str(inputs.config_path.relative_to(inputs.root)),
        "sha256": sha256(inputs.config_path),
    }
    recorded["evaluation_tool"] = {
        "path": "tools/quality_security_evaluation.py",
        "sha256": sha256(inputs.root / "tools/quality_security_evaluation.py"),
    }
    answer_gate = _gate(metrics, security_results, inputs)
    malicious_bypass_count = sum(
        result["status"] != "passed" for result in security_results
    )
    failure_count = len(metrics["failures"])
    report = {
        "schema_version": "1.0",
        "eval_run": {
            "eval_run_id": str(_stable(inputs.config["evaluation_id"])),
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "environment": "offline_deterministic_reviewed_labels",
            "commit": _git_commit(inputs.root),
            "host": {"platform": platform.platform(), "python": platform.python_version()},
            "tool": {
                "name": "tools/quality_security_evaluation.py",
                "version": TOOL_VERSION,
                "sha256": recorded["evaluation_tool"]["sha256"],
            },
        },
        "inputs": {
            "corpus_profile_id": inputs.quality["corpus_profile_id"],
            "corpus_manifest_sha256": recorded["corpus_manifest"]["sha256"],
            "dataset_id": inputs.quality["dataset_id"],
            "dataset_sha256": recorded["golden_dataset"]["sha256"],
            "index_revision": inputs.retrieval["inputs"]["index_revision"],
            "embedding_space_fingerprint": inputs.quality[
                "embedding_space_fingerprint"
            ],
            "provider_declaration": inputs.config["provider_declaration"],
            "production_model_snapshot": inputs.provider["chat"],
            "evaluation_adapter": inputs.config["evaluation_adapter"],
            "answer_policy_version": inputs.config["answer_policy"]["policy_version"],
            "answer_policy": inputs.config["answer_policy"],
            "recorded_inputs": recorded,
        },
        "strategies": inputs.retrieval["strategies"],
        "segments": inputs.retrieval["segments"],
        "case_results": case_results,
        "security_results": security_results,
        "answer_metrics": {
            key: metrics[key]
            for key in (
                "citation_identifier_validity",
                "structural_claim_coverage",
                "semantic_support_rate",
                "unsupported_claim_rate",
                "refusal_accuracy",
                "partial_answer_accuracy",
                "label_source",
            )
        }
        | {"malicious_instruction_bypass_count": malicious_bypass_count},
        "latency": {
            "query_embedding_ms": inputs.retrieval["latency"]["query_embedding_ms"],
            "retrieval_database_ms": inputs.retrieval["latency"][
                "retrieval_database_ms"
            ],
            "answer_pipeline_ms": metrics["latency"],
        },
        "usage": {
            "query_embedding_tokens": inputs.retrieval["usage"][
                "query_embedding_tokens"
            ],
            **metrics["usage"],
            "usage_scope": "deterministic adapter accounting; not provider billing",
        },
        "failures": {
            "attempted_cases": len(inputs.cases),
            "failed_cases": failure_count,
            "failure_rate": _ratio(failure_count, len(inputs.cases)),
            "by_stable_code": (
                {} if not metrics["failures"] else {"QUALITY_CASE_FAILED": failure_count}
            ),
        },
        "gate_decisions": {
            "exact_vector": inputs.retrieval["gate_decisions"]["exact_vector"],
            "lexical_comparison": inputs.retrieval["gate_decisions"][
                "lexical_comparison"
            ],
            "hnsw": inputs.retrieval["gate_decisions"]["hnsw"],
            "answer_quality_and_security": answer_gate,
        },
        "coverage_limitations": [
            "the deterministic adapter tests application orchestration and reviewed labels, not real chat-model robustness or quality",
            "semantic-support labels are synthetic and human reviewed; runtime citation validation is structural rather than entailment proof",
            "the recorded real embedding/retrieval run is reused by checksum and is not repeated against the external provider",
            "P1A remains one fixed development workspace and does not claim hostile multi-tenant isolation",
        ],
    }
    return report


def _git_commit(root: Path) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _stable_projection(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "inputs": report["inputs"],
        "strategies": report["strategies"],
        "segments": report["segments"],
        "case_results": [
            {key: value for key, value in item.items() if key != "latency_ms"}
            for item in report["case_results"]
        ],
        "security_results": report["security_results"],
        "answer_metrics": report["answer_metrics"],
        "usage": report["usage"],
        "failures": report["failures"],
        "gate_decisions": report["gate_decisions"],
        "coverage_limitations": report["coverage_limitations"],
    }


async def main_async() -> int:
    arguments = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (
        arguments.config if arguments.config.is_absolute() else root / arguments.config
    ).resolve()
    try:
        inputs = load_inputs(config_path, root)
        output_value = arguments.output or Path(inputs.config["output"])
        output_path = (
            output_value if output_value.is_absolute() else root / output_value
        )
        report = await build_report(inputs)
        if report["gate_decisions"]["answer_quality_and_security"]["status"] != "passed":
            raise ValueError("answer quality or security gate failed")
        if arguments.check:
            checked = _json(output_path)
            if _stable_projection(checked) != _stable_projection(report):
                raise ValueError("checked quality/security report is not current")
            print("quality and security evaluation report is current and passing")
            return 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(canonical_json(report), encoding="utf-8")
        try:
            display_path = output_path.relative_to(root)
        except ValueError:
            display_path = output_path
        print(f"wrote {display_path}")
        return 0
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"quality/security evaluation error: {error}", file=sys.stderr)
        return 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())

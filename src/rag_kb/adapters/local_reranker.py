"""Fixed offline MiniLM reranker with bounded request-time windowing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any
from uuid import UUID

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from rag_kb.adapters.local_reranker_artifacts import (
    LocalRerankerArtifactError,
    verify_local_reranker_artifacts,
)
from rag_kb.domain import (
    ModelRerankScore,
    RerankDocument,
    RerankMode,
)
from rag_kb.ports.model_api import RerankerAdapterError


LOCAL_RERANKER_ARTIFACTS_PATH = Path("/opt/rag-kb/local-reranker")
LOCAL_RERANKER_MANIFEST_PATH = Path(
    "/app/config/local-reranker-artifacts-v1.json"
)
LOCAL_RERANKER_MAX_SEQUENCE_LENGTH = 512
LOCAL_RERANKER_MAX_QUERY_TOKENS = 96
LOCAL_RERANKER_MAX_HIERARCHY_TOKENS = 32
LOCAL_RERANKER_WINDOW_OVERLAP = 64
LOCAL_RERANKER_MAX_DOCUMENTS = 20
LOCAL_RERANKER_MAX_WINDOWS = 80
LOCAL_RERANKER_BATCH_SIZE = 8

_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n+")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")


@dataclass(frozen=True, slots=True)
class LocalRerankWindow:
    index_chunk_id: UUID
    window_index: int
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LocalMiniLmTokenizer:
    backend_tokenizer: Tokenizer
    model_max_length: int
    cls_token_id: int
    sep_token_id: int
    pad_token_id: int

    @classmethod
    def load(cls, artifacts_path: Path) -> LocalMiniLmTokenizer:
        try:
            configuration = json.loads(
                (artifacts_path / "tokenizer_config.json").read_text(
                    encoding="utf-8"
                )
            )
            backend = Tokenizer.from_file(
                str(artifacts_path / "tokenizer.json")
            )
        except Exception as error:
            raise RerankerAdapterError("local_reranker_tokenizer_load") from error
        identifiers = tuple(
            backend.token_to_id(token) for token in ("<s>", "</s>", "<pad>")
        )
        if (
            configuration.get("model_max_length")
            != LOCAL_RERANKER_MAX_SEQUENCE_LENGTH
            or any(not isinstance(value, int) or value < 0 for value in identifiers)
        ):
            raise RerankerAdapterError("local_reranker_tokenizer_contract")
        cls_token_id, sep_token_id, pad_token_id = identifiers
        assert cls_token_id is not None
        assert sep_token_id is not None
        assert pad_token_id is not None
        return cls(
            backend_tokenizer=backend,
            model_max_length=LOCAL_RERANKER_MAX_SEQUENCE_LENGTH,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            pad_token_id=pad_token_id,
        )

    @staticmethod
    def num_special_tokens_to_add(*, pair: bool) -> int:
        return 4 if pair else 2


class LocalMiniLmReranker:
    """Score at most twenty authorized text candidates with one local model."""

    def __init__(
        self,
        *,
        artifacts_path: Path = LOCAL_RERANKER_ARTIFACTS_PATH,
        manifest_path: Path = LOCAL_RERANKER_MANIFEST_PATH,
    ) -> None:
        self._artifacts_path = artifacts_path
        self._manifest_path = manifest_path
        self._runtime: tuple[Any, ort.InferenceSession] | None = None
        self._semaphore = asyncio.Semaphore(1)

    @property
    def profile(self) -> RerankMode:
        return RerankMode.LOCAL_MINILM_V1

    @property
    def max_documents(self) -> int:
        return LOCAL_RERANKER_MAX_DOCUMENTS

    async def score(
        self,
        query: str,
        documents: tuple[RerankDocument, ...],
    ) -> tuple[ModelRerankScore, ...]:
        normalized = query.strip()
        if not normalized:
            raise ValueError("rerank query must not be empty")
        if len(documents) > self.max_documents:
            raise ValueError("local reranker document limit exceeded")
        identifiers = [item.index_chunk_id for item in documents]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("local reranker documents must be unique")
        if not documents:
            return ()
        async with self._semaphore:
            try:
                return await asyncio.to_thread(
                    self._score_sync,
                    normalized,
                    documents,
                )
            except RerankerAdapterError:
                raise
            except Exception as error:
                raise RerankerAdapterError("local_reranker_inference") from error

    def _score_sync(
        self,
        query: str,
        documents: tuple[RerankDocument, ...],
    ) -> tuple[ModelRerankScore, ...]:
        tokenizer, session = self._get_runtime()
        windows = build_local_rerank_windows(tokenizer, query, documents)
        if len(windows) > LOCAL_RERANKER_MAX_WINDOWS:
            raise RerankerAdapterError("local_reranker_window_limit")
        pad_token_id = tokenizer.pad_token_id
        if not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise RerankerAdapterError("local_reranker_pad_token")
        logits = _infer_windows(session, windows, pad_token_id)
        if len(logits) != len(windows):
            raise RerankerAdapterError("local_reranker_cardinality")
        by_document: dict[UUID, list[tuple[int, float]]] = {
            item.index_chunk_id: [] for item in documents
        }
        for window, logit in zip(windows, logits, strict=True):
            if not math.isfinite(logit):
                raise RerankerAdapterError("local_reranker_non_finite")
            by_document[window.index_chunk_id].append(
                (window.window_index, logit)
            )
        results: list[ModelRerankScore] = []
        for document in documents:
            scored = by_document[document.index_chunk_id]
            if not scored:
                raise RerankerAdapterError("local_reranker_missing_score")
            winning_index, raw_logit = max(scored, key=lambda item: item[1])
            results.append(
                ModelRerankScore(
                    index_chunk_id=document.index_chunk_id,
                    score=_sigmoid(raw_logit),
                    raw_logit=raw_logit,
                    window_count=len(scored),
                    winning_window_index=winning_index,
                )
            )
        return tuple(results)

    def _get_runtime(self) -> tuple[Any, ort.InferenceSession]:
        if self._runtime is None:
            self._runtime = _load_runtime(
                self._artifacts_path,
                self._manifest_path,
            )
        return self._runtime


def build_local_rerank_windows(
    tokenizer: Any,
    query: str,
    documents: tuple[RerankDocument, ...],
) -> tuple[LocalRerankWindow, ...]:
    """Build complete bounded pair inputs without replacing source Chunks."""

    query_ids = _encode_without_special_tokens(
        tokenizer,
        query,
    )[:LOCAL_RERANKER_MAX_QUERY_TOKENS]
    if not query_ids:
        raise ValueError("rerank query tokenization is empty")
    special_count = int(tokenizer.num_special_tokens_to_add(pair=True))
    document_budget = (
        LOCAL_RERANKER_MAX_SEQUENCE_LENGTH - len(query_ids) - special_count
    )
    if document_budget < 1:
        raise ValueError("rerank query leaves no document budget")
    separator_ids = _encode_without_special_tokens(tokenizer, "\n\n")
    prepared: list[LocalRerankWindow] = []
    for document in documents:
        hierarchy_ids = _encode_without_special_tokens(
            tokenizer,
            _hierarchy_text(document.hierarchy),
        )[:LOCAL_RERANKER_MAX_HIERARCHY_TOKENS]
        hierarchy_prefix = (
            (*hierarchy_ids, *separator_ids) if hierarchy_ids else ()
        )
        body_budget = document_budget - len(hierarchy_prefix)
        if body_budget < 1:
            raise ValueError("rerank hierarchy leaves no body budget")
        if document.modality == "table":
            body_windows = _table_windows(
                tokenizer,
                document.text,
                body_budget,
                LOCAL_RERANKER_WINDOW_OVERLAP,
            )
        else:
            body_windows = _text_windows(
                tokenizer,
                document.text,
                body_budget,
                LOCAL_RERANKER_WINDOW_OVERLAP,
            )
        if not body_windows:
            raise ValueError("rerank document tokenization is empty")
        for window_index, body_ids in enumerate(body_windows):
            input_ids = _build_xlm_roberta_pair(
                tokenizer,
                query_ids,
                (*hierarchy_prefix, *body_ids),
            )
            attention_mask = (1,) * len(input_ids)
            if (
                not input_ids
                or len(input_ids) > LOCAL_RERANKER_MAX_SEQUENCE_LENGTH
                or len(attention_mask) != len(input_ids)
            ):
                raise ValueError("rerank model input exceeds fixed bounds")
            prepared.append(
                LocalRerankWindow(
                    index_chunk_id=document.index_chunk_id,
                    window_index=window_index,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            )
    return tuple(prepared)


def _text_windows(
    tokenizer: Any,
    text: str,
    window_size: int,
    overlap: int,
) -> tuple[tuple[int, ...], ...]:
    paragraphs = tuple(
        part.strip() for part in _PARAGRAPH_BOUNDARY.split(text) if part.strip()
    ) or (text.strip(),)
    token_ids, boundaries = _encode_segments(tokenizer, paragraphs, "\n\n")
    return _sliding_windows(token_ids, boundaries, window_size, overlap)


def _table_windows(
    tokenizer: Any,
    text: str,
    window_size: int,
    overlap: int,
) -> tuple[tuple[int, ...], ...]:
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not lines:
        return ()
    header_lines: tuple[str, ...] = ()
    rows = lines
    if len(lines) >= 2 and _TABLE_SEPARATOR.fullmatch(lines[1]):
        header_lines = lines[:2]
        rows = lines[2:]
    header_ids = tuple(
        _encode_without_special_tokens(tokenizer, "\n".join(header_lines))
    )
    if header_ids:
        header_ids = header_ids[: max(1, min(len(header_ids), window_size // 3))]
    row_budget = window_size - len(header_ids)
    if header_ids:
        newline_ids = _encode_without_special_tokens(tokenizer, "\n")
        header_ids = (*header_ids, *newline_ids)
        row_budget = window_size - len(header_ids)
    if row_budget < 1:
        return (header_ids[:window_size],)
    payload_rows = rows or header_lines
    token_ids, boundaries = _encode_segments(tokenizer, payload_rows, "\n")
    payload_windows = _sliding_windows(
        token_ids,
        boundaries,
        row_budget,
        min(overlap, max(0, row_budget - 1)),
    )
    return tuple((*header_ids, *window) for window in payload_windows)


def _encode_segments(
    tokenizer: Any,
    segments: tuple[str, ...],
    separator: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    separator_ids = _encode_without_special_tokens(tokenizer, separator)
    flattened: list[int] = []
    boundaries: list[int] = []
    for segment in segments:
        if flattened:
            flattened.extend(separator_ids)
        flattened.extend(_encode_without_special_tokens(tokenizer, segment))
        boundaries.append(len(flattened))
    return tuple(flattened), tuple(boundaries)


def _encode_without_special_tokens(
    tokenizer: Any,
    text: str,
) -> tuple[int, ...]:
    if not text:
        return ()
    encoding = tokenizer.backend_tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    return tuple(int(value) for value in encoding.ids)


def _build_xlm_roberta_pair(
    tokenizer: Any,
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> tuple[int, ...]:
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    if (
        not isinstance(cls_token_id, int)
        or cls_token_id < 0
        or not isinstance(sep_token_id, int)
        or sep_token_id < 0
        or int(tokenizer.num_special_tokens_to_add(pair=True)) != 4
    ):
        raise ValueError("rerank tokenizer pair contract is invalid")
    return (
        cls_token_id,
        *first,
        sep_token_id,
        sep_token_id,
        *second,
        sep_token_id,
    )


def _sliding_windows(
    token_ids: tuple[int, ...],
    boundaries: tuple[int, ...],
    window_size: int,
    overlap: int,
) -> tuple[tuple[int, ...], ...]:
    if window_size < 1 or overlap < 0 or overlap >= window_size:
        raise ValueError("invalid rerank window bounds")
    if not token_ids:
        return ()
    if len(token_ids) <= window_size:
        return (token_ids,)
    windows: list[tuple[int, ...]] = []
    start = 0
    minimum_boundary_length = max(1, window_size // 2)
    while start < len(token_ids):
        target = min(len(token_ids), start + window_size)
        end = target
        if target < len(token_ids):
            candidates = (
                boundary
                for boundary in boundaries
                if start + minimum_boundary_length <= boundary <= target
            )
            end = max(candidates, default=target)
        windows.append(token_ids[start:end])
        if end >= len(token_ids):
            break
        next_start = end - overlap
        if len(token_ids) - next_start <= window_size:
            next_start = max(start + 1, len(token_ids) - window_size)
        else:
            next_start = max(start + 1, next_start)
        start = next_start
    return tuple(windows)


def _hierarchy_text(hierarchy: dict[str, Any]) -> str:
    titles = hierarchy.get("titles")
    if not isinstance(titles, list):
        return ""
    values: list[str] = []
    for item in titles:
        if not isinstance(item, dict):
            continue
        value = item.get("text")
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return " > ".join(values)


def _load_runtime(
    artifacts_path: Path,
    manifest_path: Path,
) -> tuple[Any, ort.InferenceSession]:
    try:
        verify_local_reranker_artifacts(artifacts_path, manifest_path)
        tokenizer = LocalMiniLmTokenizer.load(artifacts_path)
        if int(tokenizer.model_max_length) != LOCAL_RERANKER_MAX_SEQUENCE_LENGTH:
            raise RerankerAdapterError("local_reranker_tokenizer_limit")
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(artifacts_path / "onnx/model_qint8_arm64.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        if {item.name for item in session.get_inputs()} != {
            "input_ids",
            "attention_mask",
        }:
            raise RerankerAdapterError("local_reranker_input_contract")
        if len(session.get_outputs()) != 1:
            raise RerankerAdapterError("local_reranker_output_contract")
        return tokenizer, session
    except RerankerAdapterError:
        raise
    except LocalRerankerArtifactError as error:
        raise RerankerAdapterError("local_reranker_artifacts") from error
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RerankerAdapterError("local_reranker_load") from error


def _infer_windows(
    session: ort.InferenceSession,
    windows: tuple[LocalRerankWindow, ...],
    pad_token_id: int,
) -> tuple[float, ...]:
    logits: list[float] = []
    for start in range(0, len(windows), LOCAL_RERANKER_BATCH_SIZE):
        batch = windows[start : start + LOCAL_RERANKER_BATCH_SIZE]
        length = max(len(item.input_ids) for item in batch)
        input_ids = np.full(
            (len(batch), length),
            pad_token_id,
            dtype=np.int64,
        )
        attention_mask = np.zeros((len(batch), length), dtype=np.int64)
        for row, item in enumerate(batch):
            size = len(item.input_ids)
            input_ids[row, :size] = item.input_ids
            attention_mask[row, :size] = item.attention_mask
        output = session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        values = np.asarray(output[0], dtype=np.float64).reshape(-1)
        if len(values) != len(batch):
            raise RerankerAdapterError("local_reranker_batch_cardinality")
        logits.extend(float(value) for value in values)
    return tuple(logits)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID

from rag_kb.domain import RerankDocument
from tools.evaluate_minilm_source import ReferenceScorer, document_contexts, project_document


class SourceReferenceTests(unittest.TestCase):
    def test_document_context_uses_own_first_source_without_changing_evidence(self):
        chunks = {
            "later": {"ordinal": 5, "content": "later text", "source_metadata": {"document_id": "a", "original_filename": "a.pdf"}},
            "other": {"ordinal": 0, "content": "other company", "source_metadata": {"document_id": "b", "original_filename": "b.pdf"}},
            "first": {"ordinal": 0, "content": "Company A\n\n2022 annual report", "source_metadata": {"document_id": "a", "original_filename": "a.pdf"}},
        }
        hit = SimpleNamespace(index_chunk_id=UUID(int=1), text="| cash | 12 |", hierarchy={"titles": [{"text": "Balances"}]},
                              modality="table", source_metadata={"document_id": "a"})
        contexts = document_contexts(chunks)
        original = project_document(hit, contexts, "production")
        projected = project_document(hit, contexts, "context")
        self.assertEqual(projected.text, original.text)
        self.assertEqual(hit.hierarchy, {"titles": [{"text": "Balances"}]})
        self.assertEqual(projected.hierarchy["titles"][0]["text"], "a.pdf\nCompany A")
        self.assertNotIn("other company", str(projected.hierarchy))

    def test_changed_input_cannot_reuse_a_reference_score(self):
        scorer = object.__new__(ReferenceScorer)
        scorer.window_code = "fixed"
        doc = RerankDocument(index_chunk_id=UUID(int=1), text="original", hierarchy={}, modality="text")
        key = scorer.key("query", doc)
        for query, changed in (("other", doc), ("query", replace(doc, text="edited")),
                               ("query", replace(doc, document_context="different company")),
                               ("query", replace(doc, hierarchy={"titles": [{"text": "Company"}]}))):
            self.assertNotEqual(key, scorer.key(query, changed))

    def test_cache_only_miss_does_not_initialize_inference(self):
        scorer = object.__new__(ReferenceScorer)
        scorer.window_code = "fixed"
        scorer.cache = {}
        scorer.cache_only = True
        # There is deliberately no session or model path on this instance.
        doc = RerankDocument(index_chunk_id=UUID(int=1), text="original", hierarchy={}, modality="text")
        with self.assertRaisesRegex(RuntimeError, "cache-only run cannot infer"):
            scorer.score("query", (doc,))

    def test_non_finite_cache_cannot_produce_a_ranking(self):
        scorer = object.__new__(ReferenceScorer)
        scorer.window_code = "fixed"
        scorer.cache_only = True
        doc = RerankDocument(index_chunk_id=UUID(int=1), text="original", hierarchy={}, modality="text")
        scorer.cache = {scorer.key("query", doc): {"logits": [float("nan")], "window_count": 1}}
        with self.assertRaisesRegex(RuntimeError, "Invalid cached reference"):
            scorer.score("query", (doc,))

    def test_int8_cache_is_bound_to_the_complete_ordered_batch(self):
        scorer = object.__new__(ReferenceScorer)
        scorer.window_code = "fixed"
        scorer.backend = "int8"
        scorer.window_batch_size = 8
        scorer.cache_only = True
        scorer.cache = {}
        docs = (RerankDocument(UUID(int=1), "first", {}), RerankDocument(UUID(int=2), "second", {}))
        with self.assertRaisesRegex(RuntimeError, "cache-only"):
            scorer.score("query", docs)
        scorer.cache = {scorer.key("query", doc): {"logits": [0.1], "window_count": 1} for doc in docs}
        self.assertEqual(len(scorer.score("query", docs)), 2)
        with self.assertRaisesRegex(RuntimeError, "cache-only"):
            scorer.score("query", tuple(reversed(docs)))

    def test_invalid_inference_is_rejected_before_persisting_cache(self):
        scorer = object.__new__(ReferenceScorer)
        scorer.window_code = "fixed"
        scorer.cache_only = False
        scorer.cache = {}
        scorer.session = object()
        scorer.tokenizer = SimpleNamespace(pad_token_id=1)
        scorer.window_batch_size = 1
        scorer.seconds = 0.0
        doc = RerankDocument(UUID(int=1), "source", {})
        window = SimpleNamespace(index_chunk_id=doc.index_chunk_id)
        with patch("tools.evaluate_minilm_source.build_local_rerank_windows", return_value=(window,)), \
             patch("tools.evaluate_minilm_source._infer_windows", return_value=(float("nan"),)), \
             patch("tools.evaluate_minilm_source._write") as write:
            with self.assertRaisesRegex(RuntimeError, "Non-finite inference"):
                scorer.score("query", (doc,))
            write.assert_not_called()
        self.assertEqual(scorer.cache, {})

from dataclasses import replace
from types import SimpleNamespace
import unittest
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

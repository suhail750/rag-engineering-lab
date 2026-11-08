import unittest

from raglab.chunking import chunk_text
from raglab.evaluation import evaluate_retrieval
from raglab.ingestion import prepare_chunks
from raglab.retrieval import bm25_search, reciprocal_rank_fusion


class FakeEmbedder:
    def encode(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class ChunkingTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(chunk_text("   "), [])

    def test_chunks_respect_size(self):
        text = "word " * 300
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=10, overlap=10)


class IngestionTests(unittest.TestCase):
    def test_langchain_ingestion_adds_vectors_and_metadata(self):
        docs = [("notes.pdf", "First paragraph about retrieval.\n\nSecond paragraph about ranking.", {"file_type": "pdf"})]
        chunks = prepare_chunks(docs, FakeEmbedder(), chunk_size=35, overlap=5)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["source"], "notes.pdf")
        self.assertEqual(chunks[0]["metadata"]["file_type"], "pdf")
        self.assertEqual(len(chunks[0]["embedding"]), 2)

    def test_empty_documents(self):
        self.assertEqual(prepare_chunks([("empty.pdf", "  ", {})], FakeEmbedder()), [])


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"chunk_id": "a", "source": "python.txt", "text": "Python is a programming language."},
            {"chunk_id": "b", "source": "cats.txt", "text": "Cats are small domestic animals."},
        ]

    def test_bm25_prefers_matching_document(self):
        results = bm25_search("Python programming", self.chunks, top_k=2)
        self.assertEqual(results[0]["chunk_id"], "a")

    def test_rrf_deduplicates(self):
        fused = reciprocal_rank_fusion(self.chunks, list(reversed(self.chunks)))
        self.assertEqual(len(fused), 2)
        self.assertEqual({item["chunk_id"] for item in fused}, {"a", "b"})


class EvaluationTests(unittest.TestCase):
    def test_metrics(self):
        result = evaluate_retrieval([
            {"relevant_sources": ["a"], "retrieved_sources": ["a", "b"]},
            {"relevant_sources": ["c"], "retrieved_sources": ["b", "c"]},
        ], k=2)
        self.assertAlmostEqual(result["hit_rate"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["mrr"], 0.75)
        self.assertAlmostEqual(result["ndcg"], 0.8154648768, places=6)

    def test_empty_rows(self):
        self.assertEqual(evaluate_retrieval([], k=3)["mrr"], 0.0)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval([], k=0)


if __name__ == "__main__":
    unittest.main()

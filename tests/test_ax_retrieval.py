from __future__ import annotations

import unittest

from src.ax_retrieval import (
    AXCase,
    BranchPreservingAXRetriever,
    Perspective,
    recall_at_k,
    strategy_coverage_at_k,
)


class StaticEmbedding:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text: str):
        return tuple(self.mapping[text])


class BranchPreservingRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.cases = [
            AXCase(
                case_id="A",
                business_problem="case_a_problem",
                ax_target_task="case_a_task",
                business_context="case_a_context",
                root_cause_factors=("F_CHAT",),
                strategies=("S_CHATBOT",),
            ),
            AXCase(
                case_id="B",
                business_problem="case_b_problem",
                ax_target_task="case_b_task",
                business_context="case_b_context",
                root_cause_factors=("F_KNOWLEDGE",),
                strategies=("S_RAG",),
            ),
        ]
        self.mapping = {
            "q": (1.0, 0.0),
            "p_knowledge": (0.0, 1.0),
            "case_a_problem": (1.0, 0.0),
            "case_a_task": (1.0, 0.0),
            "case_a_context": (1.0, 0.0),
            "case_b_problem": (0.0, 1.0),
            "case_b_task": (0.0, 1.0),
            "case_b_context": (0.0, 1.0),
        }

    def test_perspective_recovers_case_missing_from_original_pool(self):
        model = BranchPreservingAXRetriever(
            self.cases,
            embedding_backend=StaticEmbedding(self.mapping),
            semantic_weight=1.0,
            factor_weight=0.0,
        )
        result = model.search(
            "q",
            [Perspective("P1", "p_knowledge", focus="problem")],
            candidate_k=1,
            rerank_k=1,
        )

        self.assertEqual(result.original_candidates, ("A",))
        self.assertIn("B", result.expanded_candidates)
        self.assertEqual(result.recovered_by_perspectives, ("B",))
        self.assertEqual(result.branches[1].hits[0].case_id, "B")

    def test_recall_and_strategy_coverage_use_branch_union(self):
        model = BranchPreservingAXRetriever(
            self.cases,
            embedding_backend=StaticEmbedding(self.mapping),
            semantic_weight=1.0,
            factor_weight=0.0,
        )
        result = model.search(
            "q",
            [Perspective("P1", "p_knowledge", focus="problem")],
            candidate_k=1,
            rerank_k=1,
        )
        by_id = {c.case_id: c for c in self.cases}

        self.assertEqual(recall_at_k(result, {"A", "B"}, k=1), 1.0)
        self.assertEqual(
            strategy_coverage_at_k(
                result,
                by_id,
                {"S_CHATBOT", "S_RAG"},
                k=1,
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

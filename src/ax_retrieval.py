from __future__ import annotations

"""Research prototype for GENESIS AI multi-view AX retrieval.

The module intentionally separates retrieval research from the existing Dike
decision pipeline. LLMs generate perspectives outside this module; this module
keeps those perspectives as independent branches, expands the candidate pool,
and reranks cases with multi-view semantic and Root Cause anchor evidence.

No external ML package is required for the smoke-test implementation. A
pluggable EmbeddingBackend allows replacing HashingEmbeddingBackend with a
production embedding model later without changing the retrieval contract.
"""

from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Iterable, Mapping, Protocol, Sequence


Vector = tuple[float, ...]


def _normalize_vector(values: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(float(x) * float(x) for x in values))
    if norm <= 1e-12:
        return tuple(0.0 for _ in values)
    return tuple(float(x) / norm for x in values)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("embedding dimensions must match")
    return sum(float(x) * float(y) for x, y in zip(a, b))


class EmbeddingBackend(Protocol):
    """Minimal interface for a sentence/document embedding backend."""

    def encode(self, text: str) -> Vector:
        ...


class HashingEmbeddingBackend:
    """Dependency-free baseline encoder.

    This is not the final research encoder. It is a deterministic smoke-test
    backend that mixes whitespace tokens and Korean-friendly character n-grams
    into a signed hashing vector.
    """

    def __init__(self, dim: int = 512, min_ngram: int = 2, max_ngram: int = 4):
        if dim < 32:
            raise ValueError("dim must be >= 32")
        self.dim = dim
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram

    @staticmethod
    def _features(text: str, min_ngram: int, max_ngram: int) -> list[str]:
        normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not normalized:
            return []
        words = re.findall(r"[0-9a-zA-Z가-힣_+-]+", normalized)
        compact = re.sub(r"\s+", "", normalized)
        ngrams: list[str] = []
        for n in range(min_ngram, max_ngram + 1):
            if len(compact) < n:
                continue
            ngrams.extend(compact[i : i + n] for i in range(len(compact) - n + 1))
        return [f"w:{w}" for w in words] + [f"c:{g}" for g in ngrams]

    def encode(self, text: str) -> Vector:
        values = [0.0] * self.dim
        for feat in self._features(text, self.min_ngram, self.max_ngram):
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big", signed=False)
            idx = raw % self.dim
            sign = 1.0 if (raw >> 8) & 1 else -1.0
            values[idx] += sign
        return _normalize_vector(values)


@dataclass(frozen=True)
class AXCase:
    case_id: str
    business_problem: str
    ax_target_task: str
    business_context: str
    root_cause_factors: tuple[str, ...] = ()
    strategies: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def view_text(self, view: str) -> str:
        if view == "problem":
            return self.business_problem
        if view == "task":
            return self.ax_target_task
        if view == "context":
            return self.business_context
        raise KeyError(f"unknown view: {view}")


@dataclass(frozen=True)
class Perspective:
    perspective_id: str
    text: str
    focus: str = "balanced"
    source: str = "llm"


@dataclass(frozen=True)
class RetrievalHit:
    case_id: str
    score: float
    semantic_score: float
    factor_score: float
    branch_id: str
    matched_factors: tuple[str, ...] = ()


@dataclass(frozen=True)
class BranchResult:
    branch_id: str
    query_text: str
    focus: str
    hits: tuple[RetrievalHit, ...]


@dataclass(frozen=True)
class RetrievalResult:
    original_candidates: tuple[str, ...]
    expanded_candidates: tuple[str, ...]
    branches: tuple[BranchResult, ...]

    @property
    def recovered_by_perspectives(self) -> tuple[str, ...]:
        original = set(self.original_candidates)
        return tuple(x for x in self.expanded_candidates if x not in original)


class BranchPreservingAXRetriever:
    """Multi-view, factor-aware, branch-preserving AX case retriever.

    Online flow:
        q + LLM perspectives
          -> independent candidate retrieval
          -> candidate union
          -> branch-specific reranking
          -> branch results preserved

    Offline flow:
        case Problem View
          -> Root Cause Factor labels
          -> factor embedding anchors

    The model never averages branches into one score in this layer.
    """

    VIEWS = ("problem", "task", "context")

    def __init__(
        self,
        cases: Iterable[AXCase],
        *,
        embedding_backend: EmbeddingBackend | None = None,
        view_weights: Mapping[str, float] | None = None,
        semantic_weight: float = 0.85,
        factor_weight: float = 0.15,
    ):
        self.cases = tuple(cases)
        if not self.cases:
            raise ValueError("at least one AXCase is required")
        self.by_id = {case.case_id: case for case in self.cases}
        if len(self.by_id) != len(self.cases):
            raise ValueError("case_id must be unique")

        self.encoder = embedding_backend or HashingEmbeddingBackend()
        self.base_view_weights = self._validate_weights(
            view_weights or {"problem": 0.45, "task": 0.35, "context": 0.20}
        )
        if semantic_weight < 0 or factor_weight < 0:
            raise ValueError("weights must be >= 0")
        total = semantic_weight + factor_weight
        if total <= 0:
            raise ValueError("semantic_weight + factor_weight must be > 0")
        self.semantic_weight = semantic_weight / total
        self.factor_weight = factor_weight / total

        self._case_view_vectors: dict[str, dict[str, Vector]] = {
            case.case_id: {
                view: self.encoder.encode(case.view_text(view))
                for view in self.VIEWS
            }
            for case in self.cases
        }
        self._factor_anchors = self._build_factor_anchors()

    @staticmethod
    def _validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
        values = {view: max(0.0, float(weights.get(view, 0.0))) for view in BranchPreservingAXRetriever.VIEWS}
        total = sum(values.values())
        if total <= 0:
            raise ValueError("at least one view weight must be positive")
        return {k: v / total for k, v in values.items()}

    def _weights_for_focus(self, focus: str) -> dict[str, float]:
        focus = str(focus or "balanced").lower()
        if focus not in self.VIEWS:
            return dict(self.base_view_weights)
        boosted = dict(self.base_view_weights)
        boosted[focus] *= 1.8
        return self._validate_weights(boosted)

    def _build_factor_anchors(self) -> dict[str, Vector]:
        grouped: dict[str, list[Vector]] = {}
        for case in self.cases:
            vec = self._case_view_vectors[case.case_id]["problem"]
            for factor in case.root_cause_factors:
                grouped.setdefault(factor, []).append(vec)

        anchors: dict[str, Vector] = {}
        for factor, vectors in grouped.items():
            dim = len(vectors[0])
            centroid = [
                sum(vec[i] for vec in vectors) / len(vectors)
                for i in range(dim)
            ]
            anchors[factor] = _normalize_vector(centroid)
        return anchors

    def _semantic_score(self, query_vec: Vector, case_id: str, focus: str) -> float:
        weights = self._weights_for_focus(focus)
        views = self._case_view_vectors[case_id]
        return sum(weights[view] * cosine(query_vec, views[view]) for view in self.VIEWS)

    def _factor_match(self, query_vec: Vector, case: AXCase) -> tuple[float, tuple[str, ...]]:
        if not case.root_cause_factors or not self._factor_anchors:
            return 0.0, ()
        scored = [
            (factor, cosine(query_vec, self._factor_anchors[factor]))
            for factor in case.root_cause_factors
            if factor in self._factor_anchors
        ]
        if not scored:
            return 0.0, ()
        scored.sort(key=lambda x: x[1], reverse=True)
        best = max(0.0, scored[0][1])
        matched = tuple(f for f, score in scored if score > 0.0)
        return best, matched

    def _score_case(self, query_text: str, focus: str, case: AXCase, branch_id: str) -> RetrievalHit:
        query_vec = self.encoder.encode(query_text)
        semantic = self._semantic_score(query_vec, case.case_id, focus)
        factor, matched = self._factor_match(query_vec, case)
        score = self.semantic_weight * semantic + self.factor_weight * factor
        return RetrievalHit(
            case_id=case.case_id,
            score=round(score, 8),
            semantic_score=round(semantic, 8),
            factor_score=round(factor, 8),
            branch_id=branch_id,
            matched_factors=matched,
        )

    def _retrieve_ids(self, query_text: str, focus: str, top_k: int) -> tuple[str, ...]:
        hits = [
            self._score_case(query_text, focus, case, branch_id="_candidate")
            for case in self.cases
        ]
        hits.sort(key=lambda x: (x.score, x.semantic_score, x.case_id), reverse=True)
        return tuple(hit.case_id for hit in hits[: max(1, top_k)])

    def search(
        self,
        original_query: str,
        perspectives: Sequence[Perspective],
        *,
        candidate_k: int = 10,
        rerank_k: int = 5,
    ) -> RetrievalResult:
        """Expand candidates with every perspective and preserve reranked branches."""
        if not str(original_query or "").strip():
            raise ValueError("original_query is required")
        if candidate_k < 1 or rerank_k < 1:
            raise ValueError("candidate_k and rerank_k must be >= 1")

        original_ids = self._retrieve_ids(original_query, "balanced", candidate_k)

        branch_specs: list[tuple[str, str, str]] = [
            ("q", original_query, "balanced")
        ]
        branch_specs.extend(
            (p.perspective_id, p.text, p.focus)
            for p in perspectives
            if str(p.text or "").strip()
        )

        expanded: list[str] = list(original_ids)
        seen = set(expanded)
        for _, text, focus in branch_specs[1:]:
            for case_id in self._retrieve_ids(text, focus, candidate_k):
                if case_id not in seen:
                    seen.add(case_id)
                    expanded.append(case_id)

        branches: list[BranchResult] = []
        for branch_id, text, focus in branch_specs:
            hits = [
                self._score_case(text, focus, self.by_id[case_id], branch_id)
                for case_id in expanded
            ]
            hits.sort(key=lambda x: (x.score, x.semantic_score, x.case_id), reverse=True)
            branches.append(
                BranchResult(
                    branch_id=branch_id,
                    query_text=text,
                    focus=focus,
                    hits=tuple(hits[: min(rerank_k, len(hits))]),
                )
            )

        return RetrievalResult(
            original_candidates=tuple(original_ids),
            expanded_candidates=tuple(expanded),
            branches=tuple(branches),
        )


def recall_at_k(result: RetrievalResult, relevant_case_ids: Iterable[str], k: int = 10) -> float:
    """Union Recall@K across preserved branches."""
    relevant = {str(x) for x in relevant_case_ids}
    if not relevant:
        return 0.0
    retrieved: set[str] = set()
    for branch in result.branches:
        retrieved.update(hit.case_id for hit in branch.hits[:k])
    return len(retrieved & relevant) / len(relevant)


def strategy_coverage_at_k(
    result: RetrievalResult,
    cases: Mapping[str, AXCase],
    expected_strategies: Iterable[str],
    k: int = 10,
) -> float:
    """Coverage of expected Strategy labels observed in branch results."""
    expected = {str(x) for x in expected_strategies}
    if not expected:
        return 0.0
    found: set[str] = set()
    for branch in result.branches:
        for hit in branch.hits[:k]:
            case = cases.get(hit.case_id)
            if case:
                found.update(case.strategies)
    return len(found & expected) / len(expected)

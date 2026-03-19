"""Tests for SuperLocalMemoryRetriever.

Uses mock engine to avoid heavyweight SLM initialization.
Tests the LangChain BaseRetriever contract.

Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
Paper: https://arxiv.org/abs/2603.14588
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_superlocalmemory.retriever import SuperLocalMemoryRetriever


# ---------------------------------------------------------------------------
# Mock models matching SLM's data structures
# ---------------------------------------------------------------------------

@dataclass
class MockFact:
    fact_id: str = "f001"
    content: str = "Alice went to Paris"
    fact_type: Any = None
    session_id: str = ""
    created_at: str = "2026-03-19T00:00:00Z"

    def __post_init__(self) -> None:
        if self.fact_type is None:
            self.fact_type = MockFactType()


@dataclass
class MockFactType:
    value: str = "episodic"


@dataclass
class MockRetrievalResult:
    fact: MockFact = field(default_factory=MockFact)
    score: float = 0.85
    confidence: float = 0.9
    trust_score: float = 0.7
    channel_scores: dict[str, float] = field(
        default_factory=lambda: {"semantic": 0.8, "bm25": 0.6},
    )


@dataclass
class MockRecallResponse:
    query: str = ""
    results: list[MockRetrievalResult] = field(default_factory=list)
    query_type: str = "factual"


class MockRetrieverEngine:
    """Mock engine that returns pre-configured recall results."""

    def __init__(self, results: list[MockRetrievalResult] | None = None) -> None:
        self._results = results or []
        self.recall_calls: list[dict[str, Any]] = []

    def recall(
        self, query: str, profile_id: str = "default",
        limit: int = 10, agent_id: str = "unknown", **kwargs: Any,
    ) -> MockRecallResponse:
        self.recall_calls.append({
            "query": query, "profile_id": profile_id,
            "limit": limit, "agent_id": agent_id,
        })
        return MockRecallResponse(query=query, results=self._results)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_results() -> list[MockRetrievalResult]:
    return [
        MockRetrievalResult(
            fact=MockFact(fact_id="f001", content="Alice went to Paris"),
            score=0.92, confidence=0.95, trust_score=0.8,
        ),
        MockRetrievalResult(
            fact=MockFact(fact_id="f002", content="Bob prefers TypeScript"),
            score=0.78, confidence=0.8, trust_score=0.6,
        ),
        MockRetrievalResult(
            fact=MockFact(fact_id="f003", content="Project uses Next.js"),
            score=0.65, confidence=0.7, trust_score=0.5,
        ),
    ]


@pytest.fixture
def mock_engine(
    sample_results: list[MockRetrievalResult],
) -> MockRetrieverEngine:
    return MockRetrieverEngine(results=sample_results)


@pytest.fixture
def retriever(mock_engine: MockRetrieverEngine) -> SuperLocalMemoryRetriever:
    return SuperLocalMemoryRetriever(k=5, engine=mock_engine)


# ---------------------------------------------------------------------------
# Tests: Basic retrieval
# ---------------------------------------------------------------------------

class TestRetriever:
    def test_invoke_returns_documents(
        self, retriever: SuperLocalMemoryRetriever,
    ) -> None:
        docs = retriever.invoke("Where did Alice go?")
        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)

    def test_document_content(
        self, retriever: SuperLocalMemoryRetriever,
    ) -> None:
        docs = retriever.invoke("Alice")
        assert docs[0].page_content == "Alice went to Paris"

    def test_document_metadata(
        self, retriever: SuperLocalMemoryRetriever,
    ) -> None:
        docs = retriever.invoke("Alice")
        meta = docs[0].metadata
        assert meta["fact_id"] == "f001"
        assert meta["score"] == 0.92
        assert meta["confidence"] == 0.95
        assert meta["trust_score"] == 0.8
        assert meta["source"] == "superlocalmemory"
        assert "channel_scores" in meta

    def test_respects_k_parameter(
        self, mock_engine: MockRetrieverEngine,
    ) -> None:
        retriever = SuperLocalMemoryRetriever(k=2, engine=mock_engine)
        retriever.invoke("test")
        assert mock_engine.recall_calls[0]["limit"] == 2


# ---------------------------------------------------------------------------
# Tests: Score threshold
# ---------------------------------------------------------------------------

class TestScoreThreshold:
    def test_filters_below_threshold(
        self, mock_engine: MockRetrieverEngine,
    ) -> None:
        retriever = SuperLocalMemoryRetriever(
            k=10, score_threshold=0.7, engine=mock_engine,
        )
        docs = retriever.invoke("test")
        # f001 (0.92) and f002 (0.78) pass, f003 (0.65) filtered
        assert len(docs) == 2
        assert all(d.metadata["score"] >= 0.7 for d in docs)

    def test_zero_threshold_returns_all(
        self, retriever: SuperLocalMemoryRetriever,
    ) -> None:
        docs = retriever.invoke("test")
        assert len(docs) == 3


# ---------------------------------------------------------------------------
# Tests: Empty results
# ---------------------------------------------------------------------------

class TestEmptyResults:
    def test_no_results(self) -> None:
        engine = MockRetrieverEngine(results=[])
        retriever = SuperLocalMemoryRetriever(engine=engine)
        docs = retriever.invoke("nonexistent query")
        assert docs == []


# ---------------------------------------------------------------------------
# Tests: Engine configuration
# ---------------------------------------------------------------------------

class TestEngineConfig:
    def test_custom_profile(
        self, mock_engine: MockRetrieverEngine,
    ) -> None:
        retriever = SuperLocalMemoryRetriever(
            profile_id="work-profile", engine=mock_engine,
        )
        retriever.invoke("test")
        assert mock_engine.recall_calls[0]["profile_id"] == "work-profile"

    def test_custom_agent_id(
        self, mock_engine: MockRetrieverEngine,
    ) -> None:
        retriever = SuperLocalMemoryRetriever(
            agent_id="my-rag-bot", engine=mock_engine,
        )
        retriever.invoke("test")
        assert mock_engine.recall_calls[0]["agent_id"] == "my-rag-bot"

    @patch("langchain_superlocalmemory.chat_history._engine_factory")
    def test_lazy_engine_init(self, mock_factory: MagicMock) -> None:
        engine = MockRetrieverEngine(results=[])
        mock_factory.return_value = engine
        retriever = SuperLocalMemoryRetriever(mode="c", profile_id="test")
        assert retriever._engine is None
        retriever.invoke("test")
        mock_factory.assert_called_once()

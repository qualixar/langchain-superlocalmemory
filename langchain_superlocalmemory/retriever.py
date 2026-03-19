"""SuperLocalMemory V3 — LangChain Retriever.

Implements BaseRetriever for RAG-style memory augmentation.
Uses SLM's 4-channel retrieval (semantic, BM25, entity graph, temporal)
with Fisher-Rao geodesic scoring.

Usage::

    from langchain_superlocalmemory import SuperLocalMemoryRetriever

    retriever = SuperLocalMemoryRetriever(k=5)
    docs = retriever.invoke("auth middleware patterns")

Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
Paper: https://arxiv.org/abs/2603.14588
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class SuperLocalMemoryRetriever(BaseRetriever):
    """LangChain retriever backed by SuperLocalMemory V3.

    Queries SLM's 4-channel retrieval engine and returns results
    as LangChain Document objects with metadata.

    Args:
        k: Maximum number of memories to retrieve per query.
        mode: SLM operating mode — 'a', 'b', or 'c'.
        profile_id: SLM profile for memory isolation.
        agent_id: Identifier for the calling agent.
        engine: Pre-initialized MemoryEngine instance.
        db_path: Custom database path.
        score_threshold: Minimum retrieval score to include (0.0-1.0).
    """

    k: int = 5
    mode: str = "a"
    profile_id: str = "default"
    agent_id: str = "langchain"
    score_threshold: float = 0.0
    _engine: Any = None
    _db_path: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        k: int = 5,
        mode: str = "a",
        profile_id: str = "default",
        agent_id: str = "langchain",
        score_threshold: float = 0.0,
        engine: Any | None = None,
        db_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            k=k,
            mode=mode,
            profile_id=profile_id,
            agent_id=agent_id,
            score_threshold=score_threshold,
            **kwargs,
        )
        self._engine = engine
        self._db_path = db_path

    @property
    def engine(self) -> Any:
        """Lazy-initialize engine on first access."""
        if self._engine is None:
            from langchain_superlocalmemory.chat_history import _engine_factory
            self._engine = _engine_factory(
                mode=self.mode,
                profile_id=self.profile_id,
                db_path=self._db_path,
            )
        return self._engine

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Retrieve relevant memories as LangChain Documents.

        Each Document contains:
        - page_content: The memory fact content
        - metadata: score, confidence, trust_score, fact_type, session_id,
                    channel_scores, fact_id
        """
        response = self.engine.recall(
            query=query,
            profile_id=self.profile_id,
            limit=self.k,
            agent_id=self.agent_id,
        )
        docs: list[Document] = []
        for result in response.results:
            if result.score < self.score_threshold:
                continue
            doc = Document(
                page_content=result.fact.content,
                metadata={
                    "fact_id": result.fact.fact_id,
                    "score": result.score,
                    "confidence": result.confidence,
                    "trust_score": result.trust_score,
                    "fact_type": result.fact.fact_type.value,
                    "session_id": result.fact.session_id,
                    "channel_scores": result.channel_scores,
                    "created_at": result.fact.created_at,
                    "source": "superlocalmemory",
                },
            )
            docs.append(doc)
        return docs

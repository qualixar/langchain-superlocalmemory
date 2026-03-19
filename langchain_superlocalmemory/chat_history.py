"""SuperLocalMemory V3 — LangChain Chat Message History.

Implements BaseChatMessageHistory backed by SuperLocalMemory's
MemoryEngine. Uses the direct Python API (no subprocess calls).

Messages are stored as atomic facts tagged with session_id and role,
enabling both exact history replay and semantic recall.

Usage::

    from langchain_superlocalmemory import SuperLocalMemoryChatHistory

    history = SuperLocalMemoryChatHistory(session_id="my-project")
    history.add_messages([HumanMessage(content="Hello")])
    print(history.messages)  # [HumanMessage(content="Hello")]

    # With RunnableWithMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        lambda sid: SuperLocalMemoryChatHistory(session_id=sid),
        input_messages_key="input",
        history_messages_key="history",
    )

Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
Paper: https://arxiv.org/abs/2603.14588
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

logger = logging.getLogger(__name__)

_ROLE_PREFIX = {"human": "[human]", "ai": "[ai]", "system": "[system]"}
_SESSION_TAG = "langchain_session"


def _engine_factory(
    mode: str = "a",
    profile_id: str = "default",
    db_path: str | None = None,
) -> Any:
    """Create and initialize a MemoryEngine instance.

    Lazy import to avoid pulling in heavy deps at module level.
    """
    from pathlib import Path

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.storage.models import Mode

    mode_enum = Mode(mode.lower())
    config = SLMConfig.for_mode(mode_enum)
    config.active_profile = profile_id
    if db_path:
        config.db_path = Path(db_path)
    engine = MemoryEngine(config)
    engine.initialize()
    return engine


class SuperLocalMemoryChatHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by SuperLocalMemory V3.

    Stores each message as a tagged memory record, enabling both
    exact history replay (ordered by timestamp) and semantic recall
    (via Fisher-Rao retrieval).

    Args:
        session_id: Unique conversation session identifier. Messages
            are scoped to this ID so different sessions don't mix.
        mode: SLM operating mode — 'a' (zero-cloud), 'b' (local LLM),
            'c' (cloud LLM). Default 'a' for maximum privacy.
        profile_id: SLM profile for memory isolation. Default 'default'.
        agent_id: Identifier logged in the SLM audit trail.
        engine: Pre-initialized MemoryEngine instance. If None, one
            is created automatically.
        db_path: Custom database path. If None, uses SLM default.
    """

    def __init__(
        self,
        session_id: str = "default",
        mode: str = "a",
        profile_id: str = "default",
        agent_id: str = "langchain",
        engine: Any | None = None,
        db_path: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.mode = mode
        self.profile_id = profile_id
        self.agent_id = agent_id
        self._engine = engine
        self._db_path = db_path

    @property
    def engine(self) -> Any:
        """Lazy-initialize engine on first access."""
        if self._engine is None:
            self._engine = _engine_factory(
                mode=self.mode,
                profile_id=self.profile_id,
                db_path=self._db_path,
            )
        return self._engine

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve all messages for this session, ordered by time.

        Queries the SLM database for all facts tagged with this
        session_id, then reconstructs the message sequence.
        """
        db = self.engine._db
        rows = db.execute(
            "SELECT content, created_at FROM atomic_facts "
            "WHERE profile_id = ? AND session_id = ? "
            "ORDER BY created_at ASC",
            (self.profile_id, f"{_SESSION_TAG}:{self.session_id}"),
        )
        result: list[BaseMessage] = []
        for row in rows:
            content = row.get("content", "")
            msg = _parse_message(content)
            if msg is not None:
                result.append(msg)
        return result

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Persist messages to SuperLocalMemory.

        Each message is stored as a fact with role prefix and
        session tag for later retrieval.
        """
        for msg in messages:
            prefix = _ROLE_PREFIX.get(msg.type, f"[{msg.type}]")
            tagged_content = f"{prefix} {msg.content}"
            self.engine.store(
                content=tagged_content,
                session_id=f"{_SESSION_TAG}:{self.session_id}",
                role=msg.type,
                metadata={"agent_id": self.agent_id, "source": "langchain"},
            )

    def clear(self) -> None:
        """Delete all messages for this session."""
        db = self.engine._db
        rows = db.execute(
            "SELECT fact_id FROM atomic_facts "
            "WHERE profile_id = ? AND session_id = ?",
            (self.profile_id, f"{_SESSION_TAG}:{self.session_id}"),
        )
        for row in rows:
            fid = row.get("fact_id")
            if fid:
                db.delete_fact(fid)


def _parse_message(content: str) -> BaseMessage | None:
    """Parse a stored fact back into a LangChain message."""
    if content.startswith("[human] "):
        return HumanMessage(content=content[8:])
    if content.startswith("[ai] "):
        return AIMessage(content=content[5:])
    if content.startswith("[system] "):
        return SystemMessage(content=content[9:])
    return None

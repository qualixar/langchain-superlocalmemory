"""Tests for SuperLocalMemoryChatHistory.

Uses mock engine to avoid heavyweight SLM initialization.
Tests the LangChain interface contract: messages, add_messages, clear.

Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
Paper: https://arxiv.org/abs/2603.14588
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_superlocalmemory.chat_history import (
    SuperLocalMemoryChatHistory,
    _parse_message,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockDB:
    """Mock database that stores facts in-memory for testing."""

    def __init__(self) -> None:
        self._facts: list[dict[str, Any]] = []
        self._next_id = 0

    def execute(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Mock SQL execute — supports SELECT and DELETE patterns."""
        if sql.strip().upper().startswith("SELECT"):
            # Filter facts matching profile_id and session_id
            profile_id = params[0] if len(params) > 0 else ""
            session_id = params[1] if len(params) > 1 else ""
            return [
                f for f in self._facts
                if f.get("profile_id") == profile_id
                and f.get("session_id") == session_id
            ]
        return []

    def delete_fact(self, fact_id: str) -> None:
        """Remove a fact by ID."""
        self._facts = [f for f in self._facts if f.get("fact_id") != fact_id]

    def add_fact(
        self, content: str, profile_id: str, session_id: str,
    ) -> str:
        """Helper to add a mock fact."""
        fid = f"fact_{self._next_id}"
        self._next_id += 1
        self._facts.append({
            "fact_id": fid,
            "content": content,
            "profile_id": profile_id,
            "session_id": session_id,
            "created_at": f"2026-03-19T00:00:{self._next_id:02d}Z",
        })
        return fid


class MockEngine:
    """Mock MemoryEngine that records store() calls."""

    def __init__(self) -> None:
        self._db = MockDB()
        self.store_calls: list[dict[str, Any]] = []

    def store(
        self, content: str, session_id: str = "", role: str = "",
        metadata: dict | None = None, **kwargs: Any,
    ) -> list[str]:
        self.store_calls.append({
            "content": content,
            "session_id": session_id,
            "role": role,
            "metadata": metadata,
        })
        fid = self._db.add_fact(
            content=content,
            profile_id="default",
            session_id=session_id,
        )
        return [fid]


@pytest.fixture
def mock_engine() -> MockEngine:
    return MockEngine()


@pytest.fixture
def history(mock_engine: MockEngine) -> SuperLocalMemoryChatHistory:
    return SuperLocalMemoryChatHistory(
        session_id="test-session",
        engine=mock_engine,
    )


# ---------------------------------------------------------------------------
# Tests: _parse_message
# ---------------------------------------------------------------------------

class TestParseMessage:
    def test_human_message(self) -> None:
        msg = _parse_message("[human] Hello world")
        assert isinstance(msg, HumanMessage)
        assert msg.content == "Hello world"

    def test_ai_message(self) -> None:
        msg = _parse_message("[ai] I can help with that")
        assert isinstance(msg, AIMessage)
        assert msg.content == "I can help with that"

    def test_system_message(self) -> None:
        msg = _parse_message("[system] You are a helpful assistant")
        assert isinstance(msg, SystemMessage)
        assert msg.content == "You are a helpful assistant"

    def test_unknown_prefix_returns_none(self) -> None:
        assert _parse_message("untagged content") is None

    def test_empty_content(self) -> None:
        msg = _parse_message("[human] ")
        assert isinstance(msg, HumanMessage)
        assert msg.content == ""


# ---------------------------------------------------------------------------
# Tests: SuperLocalMemoryChatHistory
# ---------------------------------------------------------------------------

class TestChatHistory:
    def test_empty_messages(self, history: SuperLocalMemoryChatHistory) -> None:
        """New session should have no messages."""
        assert history.messages == []

    def test_add_single_human_message(
        self, history: SuperLocalMemoryChatHistory, mock_engine: MockEngine,
    ) -> None:
        history.add_messages([HumanMessage(content="Hello")])
        assert len(mock_engine.store_calls) == 1
        assert mock_engine.store_calls[0]["content"] == "[human] Hello"
        assert mock_engine.store_calls[0]["role"] == "human"

    def test_add_multiple_messages(
        self, history: SuperLocalMemoryChatHistory, mock_engine: MockEngine,
    ) -> None:
        history.add_messages([
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
        ])
        assert len(mock_engine.store_calls) == 2
        assert "[human]" in mock_engine.store_calls[0]["content"]
        assert "[ai]" in mock_engine.store_calls[1]["content"]

    def test_add_system_message(
        self, history: SuperLocalMemoryChatHistory, mock_engine: MockEngine,
    ) -> None:
        history.add_messages([SystemMessage(content="Be helpful")])
        assert mock_engine.store_calls[0]["content"] == "[system] Be helpful"
        assert mock_engine.store_calls[0]["role"] == "system"

    def test_messages_round_trip(
        self, history: SuperLocalMemoryChatHistory,
    ) -> None:
        """Messages added should be retrievable."""
        history.add_messages([
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            SystemMessage(content="Be concise"),
        ])
        msgs = history.messages
        assert len(msgs) == 3
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "Hello"
        assert isinstance(msgs[1], AIMessage)
        assert msgs[1].content == "Hi there"
        assert isinstance(msgs[2], SystemMessage)
        assert msgs[2].content == "Be concise"

    def test_clear_removes_all(
        self, history: SuperLocalMemoryChatHistory,
    ) -> None:
        """Clear should remove all messages for the session."""
        history.add_messages([
            HumanMessage(content="Message 1"),
            HumanMessage(content="Message 2"),
        ])
        assert len(history.messages) == 2
        history.clear()
        assert len(history.messages) == 0

    def test_session_isolation(self, mock_engine: MockEngine) -> None:
        """Messages from different sessions should not mix."""
        history_a = SuperLocalMemoryChatHistory(
            session_id="session-a", engine=mock_engine,
        )
        history_b = SuperLocalMemoryChatHistory(
            session_id="session-b", engine=mock_engine,
        )
        history_a.add_messages([HumanMessage(content="From A")])
        history_b.add_messages([HumanMessage(content="From B")])

        msgs_a = history_a.messages
        msgs_b = history_b.messages
        assert len(msgs_a) == 1
        assert msgs_a[0].content == "From A"
        assert len(msgs_b) == 1
        assert msgs_b[0].content == "From B"

    def test_session_tag_format(
        self, history: SuperLocalMemoryChatHistory, mock_engine: MockEngine,
    ) -> None:
        """Session IDs should be namespaced with langchain_session tag."""
        history.add_messages([HumanMessage(content="test")])
        call = mock_engine.store_calls[0]
        assert call["session_id"] == "langchain_session:test-session"

    def test_metadata_includes_agent_id(
        self, history: SuperLocalMemoryChatHistory, mock_engine: MockEngine,
    ) -> None:
        history.add_messages([HumanMessage(content="test")])
        meta = mock_engine.store_calls[0]["metadata"]
        assert meta["agent_id"] == "langchain"
        assert meta["source"] == "langchain"

    def test_custom_agent_id(self, mock_engine: MockEngine) -> None:
        history = SuperLocalMemoryChatHistory(
            session_id="s1", agent_id="my-bot", engine=mock_engine,
        )
        history.add_messages([HumanMessage(content="hi")])
        assert mock_engine.store_calls[0]["metadata"]["agent_id"] == "my-bot"


# ---------------------------------------------------------------------------
# Tests: Engine lazy initialization
# ---------------------------------------------------------------------------

class TestLazyInit:
    def test_engine_not_created_until_access(self) -> None:
        """Engine should not be initialized until first use."""
        history = SuperLocalMemoryChatHistory(session_id="lazy")
        assert history._engine is None

    @patch(
        "langchain_superlocalmemory.chat_history._engine_factory",
    )
    def test_engine_created_on_messages_access(
        self, mock_factory: MagicMock,
    ) -> None:
        engine = MockEngine()
        mock_factory.return_value = engine
        history = SuperLocalMemoryChatHistory(session_id="lazy")
        _ = history.messages
        mock_factory.assert_called_once_with(
            mode="a", profile_id="default", db_path=None,
        )

    def test_pre_initialized_engine_reused(
        self, mock_engine: MockEngine,
    ) -> None:
        """If engine is passed in, factory should not be called."""
        history = SuperLocalMemoryChatHistory(
            session_id="s1", engine=mock_engine,
        )
        assert history.engine is mock_engine

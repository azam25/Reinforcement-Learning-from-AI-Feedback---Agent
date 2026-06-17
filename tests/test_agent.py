"""
Unit tests for RLAIF RAG Agent.

All LLM and retriever calls are mocked — no real API keys needed.
Run with:  pytest tests/
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# Provide a dummy API key so Settings doesn't raise at import time.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(content: str):
    doc = MagicMock()
    doc.page_content = content
    return doc


def _make_retriever(docs=None):
    retriever = MagicMock()
    retriever.invoke.return_value = docs or [
        _make_doc("Paris is the capital of France."),
        _make_doc("France is a country in Western Europe."),
    ]
    return retriever


# ---------------------------------------------------------------------------
# Tests: FindContextFitmentAgent
# ---------------------------------------------------------------------------

class TestJoinContext:
    def test_returns_context_string_and_advances_index(self):
        from RLAIF_RAG_Agent.FindContextFitmentAgent import _join_context
        docs = [_make_doc("chunk A"), _make_doc("chunk B")]
        ctx_str, new_idx = _join_context(docs, start_index=0)
        assert "Chunk 1: chunk A" in ctx_str
        assert "Chunk 2: chunk B" in ctx_str
        assert new_idx == 2

    def test_start_index_is_honoured(self):
        from RLAIF_RAG_Agent.FindContextFitmentAgent import _join_context
        docs = [_make_doc("chunk X")]
        ctx_str, new_idx = _join_context(docs, start_index=5)
        assert "Chunk 6: chunk X" in ctx_str
        assert new_idx == 6

    def test_caps_at_three_docs(self):
        from RLAIF_RAG_Agent.FindContextFitmentAgent import _join_context
        docs = [_make_doc(f"doc{i}") for i in range(10)]
        _, new_idx = _join_context(docs, start_index=0)
        assert new_idx == 3   # only first 3 consumed


class TestIsSufficient:
    def test_exact_sufficient_token(self):
        from RLAIF_RAG_Agent.FindContextFitmentAgent import _is_sufficient
        assert _is_sufficient("Sufficient context") is True
        assert _is_sufficient("sufficient context — more text") is True

    def test_insufficient_not_matched(self):
        from RLAIF_RAG_Agent.FindContextFitmentAgent import _is_sufficient
        assert _is_sufficient("Insufficient context") is False
        assert _is_sufficient("") is False

    def test_case_insensitive(self):
        from RLAIF_RAG_Agent.FindContextFitmentAgent import _is_sufficient
        assert _is_sufficient("SUFFICIENT CONTEXT") is True


class TestCapContext:
    def test_short_context_unchanged(self):
        from RLAIF_RAG_Agent.FindContextFitmentAgent import _cap_context
        text = "hello world"
        assert _cap_context(text) == text

    def test_long_context_truncated(self):
        from RLAIF_RAG_Agent.FindContextFitmentAgent import _cap_context
        long_text = "x" * 20_000
        result = _cap_context(long_text, max_chars=12_000)
        assert len(result) == 12_000


class TestRefineContext:
    @patch("RLAIF_RAG_Agent.FindContextFitmentAgent._call_llm")
    def test_returns_sufficient_on_first_iteration(self, mock_llm):
        from RLAIF_RAG_Agent.FindContextFitmentAgent import refine_context, ContextResult
        mock_llm.return_value = "Sufficient context"
        retriever = _make_retriever()

        result = refine_context("What is the capital of France?", retriever, verbose=0)

        assert isinstance(result, ContextResult)
        assert "Paris" in result.context
        assert result.iterations_used == 1

    @patch("RLAIF_RAG_Agent.FindContextFitmentAgent._call_llm")
    def test_refines_question_on_insufficient(self, mock_llm):
        from RLAIF_RAG_Agent.FindContextFitmentAgent import refine_context
        # First eval = insufficient, second question rewrite, third eval = sufficient
        mock_llm.side_effect = [
            "Insufficient context",         # eval iteration 0
            "What is the French capital?",  # question rewrite
            "Sufficient context",           # eval iteration 1
        ]
        retriever = _make_retriever()

        result = refine_context("Capital?", retriever, verbose=0, max_iterations=3)
        assert result.iterations_used == 2

    @patch("RLAIF_RAG_Agent.FindContextFitmentAgent._call_llm")
    def test_no_global_state_between_calls(self, mock_llm):
        """Two concurrent (sequential here) calls must not share chunk indices."""
        from RLAIF_RAG_Agent.FindContextFitmentAgent import refine_context
        mock_llm.return_value = "Sufficient context"
        retriever = _make_retriever()

        r1 = refine_context("Q1", retriever, verbose=0, max_iterations=1)
        r2 = refine_context("Q2", retriever, verbose=0, max_iterations=1)

        # Each result's context starts from Chunk 1, not accumulated from r1
        assert "Chunk 1:" in r1.context
        assert "Chunk 1:" in r2.context


# ---------------------------------------------------------------------------
# Tests: RAGAgent.getAnswer — return type always AgentResponse
# ---------------------------------------------------------------------------

class TestGetAnswer:
    @patch("RLAIF_RAG_Agent.FindContextFitmentAgent._call_llm")
    @patch("RLAIF_RAG_Agent.GenerateAnswerAgent._call_llm")
    def test_returns_agent_response_with_answer(self, mock_gen_llm, mock_ctx_llm):
        from RLAIF_RAG_Agent.RAGAgent import get_answer
        from RLAIF_RAG_Agent.context import AgentConfig, AgentResponse, RequestContext

        mock_ctx_llm.return_value = "Sufficient context"
        mock_gen_llm.return_value = (
            "Some reasoning...\nFinal Answer from LLM:\n- Paris is the capital."
        )

        ctx = RequestContext(retriever=_make_retriever(), config=AgentConfig())
        response = get_answer("What is the capital of France?", ctx)

        assert isinstance(response, AgentResponse)
        assert response.has_answer is True
        assert "Paris" in response.answer

    @patch("RLAIF_RAG_Agent.FindContextFitmentAgent._call_llm")
    def test_no_context_returns_no_info_response(self, mock_llm):
        from RLAIF_RAG_Agent.RAGAgent import get_answer
        from RLAIF_RAG_Agent.context import AgentConfig, AgentResponse, RequestContext

        mock_llm.return_value = "Sufficient context"

        # Retriever returns empty docs -> context stays empty
        empty_retriever = MagicMock()
        empty_retriever.invoke.return_value = []

        ctx = RequestContext(retriever=empty_retriever, config=AgentConfig())
        response = get_answer("Unknown topic", ctx)

        assert isinstance(response, AgentResponse)
        assert response.has_answer is False
        assert response.answer == "No Information available"

    def test_empty_question_raises_value_error(self):
        from RLAIF_RAG_Agent.RAGAgent import get_answer
        from RLAIF_RAG_Agent.context import AgentConfig, RequestContext

        ctx = RequestContext(retriever=_make_retriever(), config=AgentConfig())
        with pytest.raises(ValueError, match="must not be empty"):
            get_answer("   ", ctx)

    def test_question_too_long_raises_value_error(self):
        from RLAIF_RAG_Agent.RAGAgent import get_answer
        from RLAIF_RAG_Agent.context import AgentConfig, RequestContext

        ctx = RequestContext(retriever=_make_retriever(), config=AgentConfig())
        with pytest.raises(ValueError, match="maximum length"):
            get_answer("x" * 2_001, ctx)


# ---------------------------------------------------------------------------
# Tests: ShortTermMemory — eviction and persistence
# ---------------------------------------------------------------------------

class TestShortMemory:
    def test_short_memory_caps_at_limit(self, tmp_path):
        from RLAIF_RAG_Agent.llmMemory.ShortTermMemory import ShortMemory

        mem = ShortMemory(memory_file=str(tmp_path / "hist.json"))
        # Add 5 pairs to a short-memory store (limit = settings.short_memory_max = 3)
        for i in range(5):
            mem.add_to_history([f"q{i}", f"a{i}"], memory_type="short")

        assert len(mem.conversation_history) <= 3

    def test_long_memory_caps_at_limit(self, tmp_path):
        from RLAIF_RAG_Agent.llmMemory.ShortTermMemory import ShortMemory

        mem = ShortMemory(memory_file=str(tmp_path / "hist.json"))
        for i in range(80):
            mem.add_to_history([f"q{i}", f"a{i}"], memory_type="long")

        assert len(mem.conversation_history) <= 70

    def test_save_and_load_roundtrip(self, tmp_path):
        from RLAIF_RAG_Agent.llmMemory.ShortTermMemory import ShortMemory

        mem_file = str(tmp_path / "hist.json")
        mem1 = ShortMemory(memory_file=mem_file)
        mem1.add_to_history(["What is AI?", "AI is artificial intelligence."])

        mem2 = ShortMemory(memory_file=mem_file)
        mem2.load_conversation_from_json()
        assert len(mem2.conversation_history) == 1
        assert mem2.conversation_history[0]["question"] == "What is AI?"

    def test_odd_pairs_raises(self, tmp_path):
        from RLAIF_RAG_Agent.llmMemory.ShortTermMemory import ShortMemory

        mem = ShortMemory(memory_file=str(tmp_path / "hist.json"))
        with pytest.raises(ValueError):
            mem.add_to_history(["only-one-element"])


# ---------------------------------------------------------------------------
# Tests: FastAPI endpoints
# ---------------------------------------------------------------------------

class TestFastAPIEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from RLAIF_RAG_Agent.app import app
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ask_without_ingest_returns_400(self, client):
        resp = client.post("/ask", json={"question": "What is AI?"})
        assert resp.status_code == 400

    def test_ask_empty_question_returns_422(self, client):
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422

    @patch("RLAIF_RAG_Agent.app.retrieval_agent.load_and_process_document")
    def test_ingest_missing_file_returns_404(self, mock_load, client):
        mock_load.side_effect = FileNotFoundError("not found")
        resp = client.post("/ingest", json={"file_path": "/no/such/file.pdf"})
        assert resp.status_code == 404

    @patch("RLAIF_RAG_Agent.app.retrieval_agent.load_and_process_document")
    @patch("RLAIF_RAG_Agent.app.RAGAgent.get_answer")
    def test_ask_success(self, mock_get_answer, mock_load, client):
        from RLAIF_RAG_Agent.context import AgentResponse

        mock_load.return_value = _make_retriever()
        mock_get_answer.return_value = AgentResponse(
            answer="Paris", source_documents=[], context_iterations=1, has_answer=True
        )

        client.post("/ingest", json={"file_path": "/fake/doc.pdf"})
        resp = client.post("/ask", json={"question": "Capital of France?"})

        assert resp.status_code == 200
        assert resp.json()["answer"] == "Paris"
        assert resp.json()["has_answer"] is True

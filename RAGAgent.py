"""
RAGAgent — public orchestration API.

New callers should use get_answer() with a RequestContext for full
reentrancy (required for concurrent FastAPI usage).

Legacy callers can continue using the module-level convenience functions
(setSource, getAnswer, etc.) via the thin backward-compatible shim at the
bottom of this file.
"""

import logging
import re
from typing import Optional

from .context import AgentConfig, AgentResponse, RequestContext
from .config_instance import pedantic_instance
from . import RetrievalAgent as retrieval
from . import FindContextFitmentAgent as find_context
from . import GenerateAnswerAgent as generate_answer
from .llmMemory import ShortTermMemory

logger = logging.getLogger(__name__)

_MAX_QUESTION_LEN = 2_000


# ---------------------------------------------------------------------------
# Core reentrant implementation
# ---------------------------------------------------------------------------

def get_answer(question: str, ctx: RequestContext) -> AgentResponse:
    """
    Answer a question using the provided RequestContext.

    This function is reentrant: all state flows through `ctx`; no module
    globals are mutated.  Use this in async / multi-threaded environments.

    Args:
        question: The user question (max 2 000 chars).
        ctx:      Per-request context containing retriever + config.

    Returns:
        AgentResponse with answer, source documents, and iteration count.
    """
    question = _validate_question(question)
    cfg = ctx.config

    # --- Optional: memory-based question refinement ---
    memory_store: Optional[ShortTermMemory.ShortMemory] = None
    if cfg.short_memory or cfg.long_memory:
        memory_store = ShortTermMemory.ShortMemory()
        question = memory_store.refine_question(question)
        logger.debug("Memory-refined question: %s", question)

    # --- Context retrieval loop ---
    result = find_context.refine_context(
        question=question,
        retriever=ctx.retriever,
        verbose=cfg.verbose,
        config=_config_to_pedantic(cfg),
    )

    if not result.context.strip():
        logger.warning("No context found for question.")
        return AgentResponse(
            answer="No Information available",
            source_documents=[],
            context_iterations=result.iterations_used,
            has_answer=False,
        )

    # --- Final answer generation ---
    raw_answer = generate_answer.generate_final_answer(
        question=question,
        context=result.context,
        doc_type=cfg.doc_type,
        config=_config_to_pedantic(cfg),
    )

    final_answer = _extract_final_answer(raw_answer)

    # --- Persist to memory ---
    if memory_store is not None:
        memory_type = "long" if cfg.long_memory else "short"
        memory_store.add_to_history([question, final_answer], memory_type)

    return AgentResponse(
        answer=final_answer,
        source_documents=result.documents,
        context_iterations=result.iterations_used,
        has_answer=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_question(question: str) -> str:
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if len(question) > _MAX_QUESTION_LEN:
        raise ValueError(
            f"question exceeds maximum length of {_MAX_QUESTION_LEN} characters"
        )
    return question


def _extract_final_answer(raw: str) -> str:
    marker = "Final Answer from LLM:"
    idx = raw.find(marker)
    if idx != -1:
        return raw[idx + len(marker):].strip()
    return raw


def _config_to_pedantic(cfg: AgentConfig):
    """
    Build a temporary object that satisfies the attribute interface
    expected by Tasks / FindContextFitmentAgent / GenerateAnswerAgent
    (DocType, RLAIF, llmTask, OutputFormat).
    """

    class _Compat:
        DocType = cfg.doc_type
        RLAIF = cfg.rlaif
        llmTask = cfg.llm_task
        OutputFormat = cfg.output_format

    return _Compat()


# ---------------------------------------------------------------------------
# Backward-compatible module-level shim
# Uses the pedantic_instance singleton; NOT safe for concurrent requests.
# ---------------------------------------------------------------------------

def init(verbose: int) -> None:
    """Set verbosity on the global singleton. Not needed for RequestContext usage."""
    pedantic_instance.verbose = verbose
    logger.debug("Verbosity set to %d", verbose)


def exclude_RLAIF() -> None:
    pedantic_instance.RLAIF = False


def Include_RLAIF() -> None:
    pedantic_instance.RLAIF = True


def exclude_ShortTermMemory() -> None:
    pedantic_instance.ShortMemory = False


def Include_ShortTermMemory() -> None:
    pedantic_instance.ShortMemory = True


def exclude_LongTermMemory() -> None:
    pedantic_instance.LongMemory = False


def Include_LongTermMemory() -> None:
    pedantic_instance.LongMemory = True


def setSource(file_path: str, force_rebuild: bool = False) -> None:
    """Ingest a PDF document and store the retriever on the global singleton."""
    if not file_path:
        raise ValueError("file_path must not be empty")
    pedantic_instance.retrieval = retrieval.load_and_process_document(
        file_path, force_rebuild=force_rebuild
    )


def setTask(task_description: str) -> None:
    if task_description:
        pedantic_instance.llmTask = task_description


def setDocumentDetails(doc_details: str) -> None:
    if doc_details:
        pedantic_instance.DocType = doc_details


def setOutputFormat(output_format: str) -> None:
    if output_format:
        pedantic_instance.OutputFormat = output_format


def getAnswer(question: str, verbose: int = 1) -> AgentResponse:
    """
    Convenience wrapper that uses the module-level singleton.

    NOT safe for concurrent use (single shared retriever + config).
    Use get_answer(question, RequestContext(...)) for concurrent scenarios.

    Returns:
        AgentResponse (answer, source_documents, context_iterations, has_answer)
    """
    if pedantic_instance.retrieval is None:
        raise RuntimeError("No document loaded. Call setSource(file_path) first.")

    ctx = RequestContext(
        retriever=pedantic_instance.retrieval,
        config=AgentConfig(
            doc_type=pedantic_instance.DocType,
            llm_task=pedantic_instance.llmTask,
            output_format=pedantic_instance.OutputFormat,
            rlaif=pedantic_instance.RLAIF,
            short_memory=pedantic_instance.ShortMemory,
            long_memory=pedantic_instance.LongMemory,
            verbose=verbose,
        ),
    )
    return get_answer(question, ctx)
    
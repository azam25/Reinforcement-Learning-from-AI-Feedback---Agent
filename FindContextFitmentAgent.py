import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config_instance import pedantic_instance
from .settings import settings
from . import LLMConfig as ConfigLLM
from . import Tasks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class ContextResult:
    context: str
    documents: List[Any] = field(default_factory=list)
    iterations_used: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _join_context(docs: list, start_index: int = 0) -> tuple:
    """Build a context string from docs. Returns (context_str, new_chunk_index)."""
    chunks = "\n".join(
        f"Chunk {start_index + i + 1}: {doc.page_content}"
        for i, doc in enumerate(docs[:3])
    )
    new_index = start_index + min(len(docs), 3)
    return "Context:\n" + chunks, new_index


def _cap_context(context: str, max_chars: int = 12_000) -> str:
    """Prevent unbounded context growth; keep the most recent portion."""
    if len(context) > max_chars:
        return context[-max_chars:]
    return context


def _is_sufficient(response: str) -> bool:
    """
    Structured sufficiency check.
    The evaluation prompt instructs the LLM to reply ONLY with one of two
    exact tokens, so we do a case-insensitive prefix match rather than a
    substring search (which would match 'insufficient context' too).
    """
    normalized = response.strip().lower()
    return normalized.startswith("sufficient context")


_RETRYABLE = (openai.APIError, openai.APIConnectionError, openai.RateLimitError)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def _call_llm(messages: list, temperature: float, max_tokens: int) -> str:
    response = ConfigLLM.client.chat.completions.create(
        model=ConfigLLM.MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_question(question: str, doc_type: Optional[str] = None) -> str:
    """Rewrite a question to be simpler/more self-contained."""
    prompt = [
        {"role": "system", "content": Tasks.getGeneratedQuestionTask()},
        {
            "role": "user",
            "content": f"Original Question: {question}\n\nSimplify and rewrite the question.",
        },
    ]
    return _call_llm(prompt, ConfigLLM.TEMPRATURE_GENERATE_QUESTION, ConfigLLM.MAX_TOEKNS_GENERATE_QUESTION)


def refine_context(
    question: str,
    retriever: Any,
    verbose: int = 1,
    max_iterations: Optional[int] = None,
    config=None,
) -> ContextResult:
    """
    Iteratively retrieve and evaluate context until sufficient or
    max_iterations reached.  All state is local — this function is
    thread-safe and reentrant.

    Args:
        question:       The (optionally memory-refined) user question.
        retriever:      FAISS retriever instance.
        verbose:        0 = silent, 1 = normal, 2 = debug.
        max_iterations: Override the global default from settings.
        config:         Optional config object (defaults to pedantic_instance).

    Returns:
        ContextResult with accumulated context, source docs, and iteration count.
    """
    cfg = config or pedantic_instance
    iterations = max_iterations if max_iterations is not None else settings.max_context_iterations

    if verbose == 2:
        logger.debug("Context evaluation loop started (max_iterations=%d)", iterations)

    input_question = question
    accumulated_context = ""
    all_documents: List[Any] = []
    chunk_index = 0          # local — no module global

    for iteration in range(iterations):
        docs = retriever.invoke(input_question)
        chunk_str, chunk_index = _join_context(docs, chunk_index)
        accumulated_context = _cap_context(accumulated_context + "\n" + chunk_str)
        all_documents.extend(docs)

        evaluation_prompt = [
            {
                "role": "system",
                "content": Tasks.getContextEvaluationTask(cfg.DocType),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{accumulated_context}\n\n"
                    f"Question: {question}\n\n"
                    "Does this context sufficiently answer the question? "
                    "Reply ONLY with 'Sufficient context' or 'Insufficient context'."
                ),
            },
        ]

        try:
            eval_response = _call_llm(
                evaluation_prompt,
                ConfigLLM.TEMPRATURE_GENERATE_QUESTION,
                ConfigLLM.MAX_TOEKNS_GENERATE_QUESTION,
            ).strip()
        except Exception:
            logger.exception("LLM evaluation call failed at iteration %d; using context so far.", iteration)
            break

        if _is_sufficient(eval_response):
            if verbose == 2:
                logger.debug("Context sufficient after %d iteration(s).", iteration + 1)
            return ContextResult(accumulated_context, all_documents, iteration + 1)

        if verbose == 2:
            logger.debug("Context insufficient at iteration %d; refining question.", iteration)

        try:
            input_question = generate_question(input_question)
            if verbose == 2:
                logger.debug("Refined question: %s", input_question)
        except Exception:
            logger.exception("Question generation failed at iteration %d; stopping refinement.", iteration)
            break

    logger.debug("Using accumulated context after %d iteration(s).", iterations)
    return ContextResult(accumulated_context, all_documents, iterations)


# ---------------------------------------------------------------------------
# Backward-compat alias (old callers that unpack (context, lstContext))
# ---------------------------------------------------------------------------

def refine_context_compat(question, retriever, verbose, max_iterations=3, config=None):
    """Deprecated shim — prefer refine_context() which returns ContextResult."""
    result = refine_context(question, retriever, verbose, max_iterations, config)
    return result.context, result.documents

import logging
from typing import Optional

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import LLMConfig as ConfigLLM
from . import Tasks as task
from .config_instance import pedantic_instance

logger = logging.getLogger(__name__)

_RETRYABLE = (openai.APIError, openai.APIConnectionError, openai.RateLimitError)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def _call_llm(messages: list) -> str:
    response = ConfigLLM.client.chat.completions.create(
        model=ConfigLLM.MODEL_NAME,
        messages=messages,
        temperature=ConfigLLM.TEMPRATURE_GENERATE_QUESTION,
        max_tokens=ConfigLLM.MAX_TOEKNS_GENERATE_ANSWER,
    )
    return response.choices[0].message.content


def generate_final_answer(
    question: str,
    context: str,
    doc_type: str,
    config=None,
) -> str:
    """
    Generate the final answer, optionally with RLAIF self-evaluation.

    Args:
        question:  The user question.
        context:   Accumulated retrieved context.
        doc_type:  Document type string for the system prompt.
        config:    Optional config object (defaults to pedantic_instance).

    Returns:
        Raw LLM response string (contains 'Final Answer from LLM:' section).
    """
    cfg = config or pedantic_instance

    # Prefer config-provided values for reentrancy; fall back to Tasks functions
    # (which read from the global singleton) only when no config is supplied.
    if config is not None and hasattr(config, "llmTask"):
        llm_task_text = config.llmTask
        output_format_instructions = config.OutputFormat
    elif config is not None and hasattr(config, "llm_task"):
        llm_task_text = config.llm_task
        output_format_instructions = config.output_format
    else:
        llm_task_text = task.getFinalAnswerTask()
        output_format_instructions = task.getOuputFormatTask()

    system_content = (
        f"You are an expert in analyzing {doc_type} documents and a self-evaluating agent. "
        + llm_task_text
    )

    if cfg.RLAIF:
        logger.debug("RLAIF self-evaluation enabled.")
        user_content = (
            f"Context: {context}\n\nQuestion: {question}\n"
            f"{llm_task_text}\n\n"
            "Ensure the final answer is under the heading 'Final Answer from LLM:'.\n\n"
            "Your important tasks are:\n"
            "1. Generate an initial answer to the question based on the Context.\n"
            "2. Evaluate the answer by considering the question, required format, "
            "Context, and the generated answer.\n"
            "3. If the answer is incomplete, inaccurate, or misaligned, refine it with "
            "more detail, accuracy, and completeness.\n"
            "4. Preserve all original values such as numbers, dates, and names without alteration.\n"
            "5. Present the Final Answer in bullet points under 'Final Answer from LLM:' "
            "at the very end of the response.\n\n"
            f"Output format instructions:\n{output_format_instructions}"
        )
    else:
        logger.debug("Standard (non-RLAIF) response generation.")
        user_content = (
            f"Context: {context}\n\nQuestion: {question}\n"
            f"{llm_task_text}\n\n"
            "Ensure the final answer is under the heading 'Final Answer from LLM:'.\n\n"
            f"Output format instructions:\n{output_format_instructions}"
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    return _call_llm(messages)
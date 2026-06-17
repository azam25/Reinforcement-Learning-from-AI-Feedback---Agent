"""
Per-request context and typed response objects.

RequestContext bundles all per-request state so the pipeline is
reentrant: no module globals are mutated during a request.
"""

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class AgentConfig:
    """Runtime configuration for one agent invocation."""
    doc_type: str = "Complex Document"
    llm_task: str = (
        "Generate an accurate answer based on the provided question and document."
    )
    output_format: str = (
        "Use the provided context to generate a complete and detailed answer. "
        "Ensure the final answer includes all key points in bullets."
    )
    rlaif: bool = True
    short_memory: bool = False
    long_memory: bool = False
    verbose: int = 1


@dataclass
class RequestContext:
    """
    All per-request mutable state.  One instance per call to getAnswer().
    Never share this object across requests.
    """
    retriever: Any              # FAISS retriever instance
    config: AgentConfig = field(default_factory=AgentConfig)


@dataclass
class AgentResponse:
    """Consistent return type from getAnswer(), regardless of code path."""
    answer: str
    source_documents: List[Any] = field(default_factory=list)
    context_iterations: int = 0
    has_answer: bool = True

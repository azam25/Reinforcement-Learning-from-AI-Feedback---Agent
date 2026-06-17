import json
import logging
import os
from typing import List

from . import LLM
from . import Prompt
from ..settings import settings

logger = logging.getLogger(__name__)


class ShortMemory:
    """Conversation memory backed by a local JSON file."""

    def __init__(self, memory_file: str | None = None):
        self.memory_file = memory_file or settings.memory_file
        self.conversation_history: List[dict] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_conversation_to_json(self) -> None:
        with open(self.memory_file, "w") as fh:
            json.dump(self.conversation_history, fh, indent=4)

    def load_conversation_from_json(self) -> None:
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, "w") as fh:
                json.dump([], fh, indent=4)
            self.conversation_history = []
            return

        if os.path.getsize(self.memory_file) == 0:
            self.conversation_history = []
            return

        with open(self.memory_file, "r") as fh:
            self.conversation_history = json.load(fh)

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def add_to_history(self, question_answer_pairs: list, memory_type: str = "long") -> None:
        """
        Add Q/A pairs to history, respecting the size cap for the given memory type.

        Args:
            question_answer_pairs: Flat list [q1, a1, q2, a2, ...]
            memory_type:           'short' or 'long'
        """
        if len(question_answer_pairs) % 2 != 0:
            raise ValueError("question_answer_pairs must contain an even number of elements.")

        max_items = (
            settings.long_memory_max if memory_type == "long" else settings.short_memory_max
        )

        for i in range(0, len(question_answer_pairs), 2):
            question = question_answer_pairs[i]
            answer = question_answer_pairs[i + 1]

            # Evict oldest entry BEFORE appending to honour the cap
            while len(self.conversation_history) >= max_items:
                self.conversation_history.pop(0)

            self.conversation_history.append({"question": question, "answer": answer})

        # Save once after all pairs are added (not inside the inner loop)
        self.save_conversation_to_json()
        logger.debug(
            "Memory updated: %d/%d entries (%s)",
            len(self.conversation_history),
            max_items,
            memory_type,
        )

    # ------------------------------------------------------------------
    # Question refinement
    # ------------------------------------------------------------------

    def refine_question(self, current_question: str) -> str:
        """Use conversation history to make a follow-up question self-contained."""
        self.load_conversation_from_json()

        if not self.conversation_history:
            return current_question

        context = "\n".join(
            f"Q: {entry['question']}\nA: {entry['answer']}"
            for entry in self.conversation_history
        )
        prompt = Prompt.getRefineQuestionPrompt(context, current_question)

        try:
            refined = self.call_llm(prompt)
        except Exception:
            logger.exception("LLM call for question refinement failed; returning original.")
            return current_question

        if "Not Applicable" in refined:
            logger.debug("Memory refinement returned 'Not Applicable'; using original question.")
            return current_question

        return refined.strip()

    def call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant for refining questions."},
            {"role": "user", "content": prompt},
        ]
        return LLM.generateFromLLM(messages)


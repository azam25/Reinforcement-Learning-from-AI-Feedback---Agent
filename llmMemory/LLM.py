import logging

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import Config

logger = logging.getLogger(__name__)

_RETRYABLE = (openai.APIError, openai.APIConnectionError, openai.RateLimitError)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def generateFromLLM(str_messages: list) -> str:
    client = openai.OpenAI(
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_API_BASE,
    )
    chat_response = client.chat.completions.create(
        model=Config.MODEL_NAME,
        messages=str_messages,
        temperature=Config.TEMPRATURE,
        max_tokens=Config.MAX_TOKEN,
    )
    return chat_response.choices[0].message.content
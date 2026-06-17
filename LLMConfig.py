import openai
from .settings import settings

# ---------------------------------------------------------------------------
# All values are driven by the Settings singleton (env / .env file).
# Do NOT add literals here.
# ---------------------------------------------------------------------------
MODEL_NAME                   = settings.model_name
TEMPRATURE_GENERATE_QUESTION = settings.temperature_question
MAX_TOEKNS_GENERATE_QUESTION = settings.max_tokens_question
MAX_TOEKNS_GENERATE_ANSWER   = settings.max_tokens_answer

client = openai.OpenAI(
    api_key  = settings.openai_api_key,
    base_url = settings.openai_api_base,
)
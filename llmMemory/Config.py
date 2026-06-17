# All values are driven by the shared Settings singleton (env / .env file).
# Do NOT add literals here.
from ..settings import settings

OPENAI_API_KEY  = settings.openai_api_key
OPENAI_API_BASE = settings.openai_api_base
MODEL_NAME      = settings.model_name
TEMPRATURE      = settings.temperature_memory
MAX_TOKEN       = settings.max_tokens_question

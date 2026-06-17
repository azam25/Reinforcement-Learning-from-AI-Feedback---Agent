# Package init — configure logging and load settings from environment.
# Secrets must be provided via a .env file or real environment variables;
# nothing is hard-coded here.
import logging
import logging.config
import os

# Silence OTEL if not configured (prevents noisy startup warnings).
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from .settings import settings  # noqa: F401  — loads .env on first import

# ------------------------------------------------------------------
# Structured logging: JSON-friendly when running in a service,
# human-readable when running interactively.
# ------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format=_LOG_FORMAT,
    datefmt=_DATE_FORMAT,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())
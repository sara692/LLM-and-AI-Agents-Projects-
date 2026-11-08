"""Central configuration, loaded from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Any column whose name contains one of these fragments is treated as personal data:
# hidden from the LLM, unreferenceable in generated code, and never returned.
DEFAULT_PROTECTED_PATTERNS = (
    "name", "email", "phone", "ssn", "address", "passport", "iban", "password",
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip())


@dataclass(frozen=True)
class Settings:
    # --- OpenRouter -------------------------------------------------------
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Code-specialised instruct model: strong pandas generation at low cost.
    openrouter_model: str = "qwen/qwen-2.5-coder-32b-instruct"
    # Optional comma-separated fallbacks; OpenRouter tries them in order on failure.
    openrouter_fallback_models: tuple[str, ...] = ()
    llm_timeout_s: float = 60.0
    llm_http_retries: int = 2

    # --- Agent / safety ---------------------------------------------------
    max_retries: int = 5                 # self-repair attempts for generated code
    use_llm_mapping: bool = True         # LLM-assisted schema mapping (fail-soft)
    expose_category_values: bool = True  # show low-cardinality labels (not rows) to the LLM
    protected_patterns: tuple[str, ...] = DEFAULT_PROTECTED_PATTERNS

    # --- Sandbox ----------------------------------------------------------
    sandbox_mode: str = "thread"         # "thread" (fast) or "process" (killable, spawn)
    sandbox_timeout_s: float = 10.0
    max_result_rows: int = 50

    # --- Data / API -------------------------------------------------------
    data_dir: Path = BASE_DIR / "data"
    reports_dir: Path = BASE_DIR / "reports"
    max_upload_mb: int = 10
    max_uploaded_datasets: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", cls.openrouter_base_url),
            openrouter_model=os.getenv("OPENROUTER_MODEL", cls.openrouter_model),
            openrouter_fallback_models=tuple(
                m.strip() for m in os.getenv("OPENROUTER_FALLBACK_MODELS", "").split(",") if m.strip()
            ),
            llm_timeout_s=_env_float("LLM_TIMEOUT_S", cls.llm_timeout_s),
            llm_http_retries=_env_int("LLM_HTTP_RETRIES", cls.llm_http_retries),
            max_retries=_env_int("MAX_RETRIES", cls.max_retries),
            use_llm_mapping=_env_bool("USE_LLM_MAPPING", cls.use_llm_mapping),
            expose_category_values=_env_bool("EXPOSE_CATEGORY_VALUES", cls.expose_category_values),
            protected_patterns=_env_list("PROTECTED_COLUMN_PATTERNS", DEFAULT_PROTECTED_PATTERNS),
            sandbox_mode=os.getenv("SANDBOX_MODE", cls.sandbox_mode).strip().lower(),
            sandbox_timeout_s=_env_float("SANDBOX_TIMEOUT_S", cls.sandbox_timeout_s),
            max_result_rows=_env_int("MAX_RESULT_ROWS", cls.max_result_rows),
            data_dir=Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))),
            reports_dir=Path(os.getenv("REPORTS_DIR", str(BASE_DIR / "reports"))),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", cls.max_upload_mb),
            max_uploaded_datasets=_env_int("MAX_UPLOADED_DATASETS", cls.max_uploaded_datasets),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()

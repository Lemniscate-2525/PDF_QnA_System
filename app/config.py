import os
from dotenv import load_dotenv

load_dotenv()


def get_config(key, default=None):
    """
    Read configuration from environment variables or Streamlit Secrets.
    Environment variables take precedence.
    """
    value = os.getenv(key)

    if value is not None:
        return value

    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# PostgreSQL Configuration
DB_HOST = get_config("DB_HOST", "localhost")
DB_PORT = int(get_config("DB_PORT", "5432"))
DB_NAME = get_config("DB_NAME", "pdf_qna")
DB_USER = get_config("DB_USER", "postgres")
DB_PASSWORD = get_config("DB_PASSWORD", "")

# OpenRouter Configuration
OPENROUTER_API_KEY = get_config("OPENROUTER_API_KEY")
OPENROUTER_MODEL = get_config(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3.5-super-vl:free"
)

# OCR Configuration
OCR_ENABLED = str(
    get_config("OCR_ENABLED", "true")
).strip().lower() in {"1", "true", "yes", "on"}

OCR_LANGUAGES = get_config("OCR_LANGUAGES", "eng")

# Supabase Auth
SUPABASE_URL = get_config("SUPABASE_URL")
SUPABASE_ANON_KEY = get_config("SUPABASE_ANON_KEY")
APP_URL = get_config("APP_URL", "http://localhost:8501")

# Redis (optional locally; required for distributed limits/cache in production)
REDIS_URL = get_config("REDIS_URL")

SUPABASE_SERVICE_ROLE_KEY = get_config("SUPABASE_SERVICE_ROLE_KEY")
UPLOAD_BUCKET = get_config("UPLOAD_BUCKET", "pdf-uploads")

APP_ENV = get_config("APP_ENV", "development").strip().lower()

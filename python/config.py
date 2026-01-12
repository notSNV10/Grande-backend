"""
Configuration loader for environment-driven settings.
Use simple defaults for local development; override via environment variables.
"""
import os
from functools import lru_cache
from typing import Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file from the same directory as this config file
    config_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(config_dir, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip .env loading
    # Environment variables must be set manually or via system
    pass


class Settings:
    # Database
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_user: str = os.getenv("DB_USER", "root")
    db_password: str = os.getenv("DB_PASSWORD", "")
    db_name: str = os.getenv("DB_NAME", "adarna_db")
    db_port: int = int(os.getenv("DB_PORT", "3306"))

    # AI keys (placeholders; use env vars in local .env or host secrets)
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    mistral_api_key: Optional[str] = os.getenv("MISTRAL_API_KEY")

    # Operational
    ai_timeout_seconds: int = int(os.getenv("AI_TIMEOUT_SECONDS", "20"))
    ai_log_prompts: bool = os.getenv("AI_LOG_PROMPTS", "true").lower() == "true"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


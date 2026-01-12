"""
Shared prompt helpers to ensure all AI calls are auditable.
"""
import logging
from typing import Any, Dict
from config import get_settings

logger = logging.getLogger(__name__)


def log_prompt(role: str, payload: Dict[str, Any]) -> None:
    """Log prompts for audit; avoid secrets."""
    settings = get_settings()
    if settings.ai_log_prompts:
        logger.info("AI_PROMPT | role=%s | payload=%s", role, payload)


def envelope(system: str, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Standardized prompt envelope."""
    return {
        "system": system,
        "task": task,
        "context": context,
    }


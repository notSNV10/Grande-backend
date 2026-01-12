"""
Chat / User Interaction AI (OpenRouter-compatible LLM).
Real HTTP call with deterministic fallback and full prompt logging.
"""
import logging
from typing import Dict, Any

import requests

from config import get_settings
from ai.prompt_utils import log_prompt, envelope

logger = logging.getLogger(__name__)

# Preferred OpenRouter models for this capstone
OPENROUTER_MODEL_PRIMARY = "meta-llama/llama-3.1-8b-instruct"
OPENROUTER_MODEL_FALLBACK = "mistralai/mistral-7b-instruct"


def _fallback_reply(audit_prompt: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Deterministic, non-LLM fallback so the system remains usable."""
    reply = (
        "The conversational AI is temporarily unavailable. "
        "Based on the latest KPIs, keep promoting your best-performing services, "
        "watch items flagged as low or urgent in inventory, and ensure peak days have enough staff."
    )
    return {
        "reply": reply,
        "reasoning": reason,
        # Store only a minimal, non-recursive view of the prompt for API responses
        "prompt_used": audit_prompt,
        "fallback_used": True,
    }


def chat_llm(query: str, data_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call OpenRouter to generate a business-analytics answer.

    Returns dict with keys: reply, reasoning, prompt_used, fallback_used.
    All prompts are logged via log_prompt() for auditability.
    """
    settings = get_settings()

    full_prompt = envelope(
        system=(
            "You are a virtual business analyst for a multi-branch hair & nail studio in the Philippines. "
            "Use ONLY the data and summaries given in context; do not invent numbers. "
            "IMPORTANT: Always use Philippine Peso (₱) for all currency amounts. Never use dollars ($) or any other currency. "
            "All prices, revenue, costs, and monetary values must be displayed in Philippine Peso format (₱)."
        ),
        task=(
            "Answer the user question in clear business language, referencing KPIs when relevant, "
            "and suggest 2–3 concrete next actions. "
            "Remember: Use Philippine Peso (₱) for all currency amounts, never dollars. "
            f"User question: {query}"
        ),
        context=data_summary,
    )
    # Log the full prompt for audit in server logs
    log_prompt("chat", full_prompt)

    # Minimal prompt info returned in API to avoid circular references
    audit_prompt: Dict[str, Any] = {
        "system": full_prompt.get("system"),
        "task": full_prompt.get("task"),
    }
    ctx = full_prompt.get("context")
    if isinstance(ctx, dict):
        audit_prompt["context_keys"] = list(ctx.keys())

    if not settings.openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY is not set; using local chat fallback.")
        return _fallback_reply(audit_prompt, "missing-openrouter-api-key")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL_PRIMARY,
        "messages": [
            {"role": "system", "content": full_prompt["system"]},
            {
                "role": "user",
                "content": f"Task: {full_prompt['task']}\n\nContext JSON:\n{full_prompt['context']}\n\nCRITICAL: All currency must be in Philippine Peso (₱). Replace any dollar signs ($) with peso signs (₱). Never use $ or USD.",
            },
        ],
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=settings.ai_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            logger.error("OpenRouter response missing choices: %s", data)
            return _fallback_reply(audit_prompt, "openrouter-empty-choices")

        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if not isinstance(content, str) or not content.strip():
            logger.error("OpenRouter response had empty content: %s", data)
            return _fallback_reply(audit_prompt, "openrouter-empty-content")

        # Post-process: Replace any dollar signs with peso signs
        content = content.replace("$", "₱")
        content = content.replace("USD", "PHP")
        content = content.replace("US$", "₱")
        content = content.replace("dollars", "pesos")
        content = content.replace("Dollars", "Pesos")
        content = content.replace("DOLLARS", "PESOS")

        return {
            "reply": content.strip(),
            "reasoning": "openrouter-llm-response",
            "prompt_used": audit_prompt,
            "fallback_used": False,
        }
    except requests.Timeout:
        logger.exception("OpenRouter chat request timed out.")
        return _fallback_reply(audit_prompt, "openrouter-timeout")
    except requests.RequestException as exc:
        logger.exception("OpenRouter chat HTTP error: %s", exc)
        return _fallback_reply(audit_prompt, "openrouter-http-error")
    except Exception as exc:
        logger.exception("Unexpected error during OpenRouter chat call: %s", exc)
        return _fallback_reply(audit_prompt, "openrouter-unexpected-error")
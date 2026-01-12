"""
Gemini NLP Service - Insight Generation

ARCHITECTURE:
- Gemini is used for NLP tasks (summarization, insights, explanations)
- Forecasting is handled locally using Python (Prophet/ARIMA/sklearn) - NO Gemini for numerical predictions
- Gemini converts structured analytics outputs into human-readable insights

PURPOSE:
- Convert structured KPI data into natural language summaries
- Generate business insights and recommendations from analytics
- Provide human-readable explanations of data trends
- Explain forecast results

FALLBACK BEHAVIOR:
- Intelligent template-based summary generation from KPIs
- Produces actionable insights and recommendations
- Maintains system functionality when external APIs are unavailable
"""
import logging
import sys
from typing import Dict, Any, List, Optional
import time

import requests

from config import get_settings
from ai.prompt_utils import log_prompt, envelope

logger = logging.getLogger(__name__)

# Gemini API endpoints - try both v1 and v1beta
GEMINI_API_BASE_V1 = "https://generativelanguage.googleapis.com/v1/models"
GEMINI_API_BASE_V1BETA = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-1.5-flash"  # Use flash model (faster, more reliable)


def _generate_template_summary(kpi_summary: Dict[str, Any]) -> str:
    """Generate an intelligent summary from KPIs using template-based logic."""
    parts = []
    
    # Revenue analysis
    if "total_revenue" in kpi_summary or "average_daily" in kpi_summary:
        revenue = kpi_summary.get("total_revenue", kpi_summary.get("average_daily", 0))
        if isinstance(revenue, (int, float)) and revenue > 0:
            parts.append(f"Revenue performance shows {'strong' if revenue > 100000 else 'moderate'} activity.")
    
    # Growth trend
    if "growth_rate" in kpi_summary:
        growth = kpi_summary.get("growth_rate", 0)
        if isinstance(growth, (int, float)):
            if growth > 0.1:
                parts.append(f"Positive growth trend detected ({growth*100:.1f}% increase).")
            elif growth < -0.1:
                parts.append(f"Declining trend observed ({abs(growth)*100:.1f}% decrease) - requires attention.")
            else:
                parts.append("Stable performance with minimal fluctuations.")
    
    # Top service
    if "top_service" in kpi_summary:
        service = kpi_summary.get("top_service", "")
        if service:
            parts.append(f"'{service}' is the top-performing service and should be prioritized.")
    
    # Inventory alerts
    if "low_stock_items" in kpi_summary:
        low_stock = kpi_summary.get("low_stock_items", 0)
        if isinstance(low_stock, int) and low_stock > 0:
            parts.append(f"⚠️ {low_stock} item(s) require immediate restocking to avoid stockouts.")
    
    # Recommendations
    recommendations = []
    if kpi_summary.get("growth_rate", 0) > 0.1:
        recommendations.append("Capitalize on growth momentum with targeted marketing campaigns.")
    if kpi_summary.get("low_stock_items", 0) > 0:
        recommendations.append("Review inventory levels and establish reorder points for critical items.")
    if "top_service" in kpi_summary:
        recommendations.append("Increase capacity for high-demand services to maximize revenue potential.")
    
    if not parts:
        parts.append("Business metrics are being monitored. Continue tracking key performance indicators.")
    
    summary = " ".join(parts)
    if recommendations:
        summary += "\n\n**Recommendations:**\n" + "\n".join(f"• {r}" for r in recommendations[:3])
    
    return summary


def _fallback_summary(prompt: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Intelligent fallback that generates summaries from KPIs when API fails."""
    kpi_summary = prompt.get("context", {})
    summary = _generate_template_summary(kpi_summary)
    
    return {
        "summary": summary,
        "prompt_used": prompt,
        "fallback_used": True,
        "fallback_reason": reason,
    }


def _list_available_models(settings) -> List[str]:
    """List available Gemini models for this API key."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models?key={settings.gemini_api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = []
            if "models" in data:
                for model in data["models"]:
                    name = model.get("name", "")
                    # Extract model name (e.g., "models/gemini-1.5-flash" -> "gemini-1.5-flash")
                    if "/" in name:
                        model_id = name.split("/")[-1]
                        # Only include models that support generateContent
                        supported_methods = model.get("supportedGenerationMethods", [])
                        if "generateContent" in supported_methods:
                            models.append(model_id)
            print(f"[GEMINI] Found {len(models)} available models: {models}", file=sys.stderr)
            return models
    except Exception as e:
        print(f"[GEMINI] Failed to list models: {e}", file=sys.stderr)
    return []


def _call_gemini_api(prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> Optional[Dict[str, Any]]:
    """
    Call Gemini API for text generation.
    First lists available models, then tries them.
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY not set, skipping API call")
        return None
    
    # First, try to get list of available models
    print(f"[GEMINI] Fetching list of available models...", file=sys.stderr)
    available_models = _list_available_models(settings)
    
    # Fallback list if we can't get available models
    fallback_models = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest", 
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro",
    ]
    
    # Use available models if we got them, otherwise use fallback
    models_to_try = available_models if available_models else fallback_models
    print(f"[GEMINI] Will try models: {models_to_try}", file=sys.stderr)
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Try both v1 and v1beta endpoints for each model
    api_bases = [GEMINI_API_BASE_V1, GEMINI_API_BASE_V1BETA]
    
    for model_name in models_to_try:
        for api_base in api_bases:
            # Gemini API format - try both v1 and v1beta
            url = f"{api_base}/{model_name}:generateContent?key={settings.gemini_api_key}"
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    api_version = "v1" if "v1/models" in api_base else "v1beta"
                    print(f"[GEMINI] Attempting {model_name} on {api_version} (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                    logger.info(f"Attempting Gemini API call with {model_name} on {api_version} (attempt {attempt + 1}/{max_retries})")
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=15
                    )
                    
                    print(f"[GEMINI] Response status: {response.status_code} for {model_name} on {api_version}", file=sys.stderr)
                    logger.info(f"Gemini API response status: {response.status_code}")
                    
                    # Handle rate limiting or temporary errors - retry
                    if response.status_code == 429:
                        wait_time = 5 * (attempt + 1)
                        logger.info(f"Rate limited, waiting {wait_time} seconds and retrying...")
                        time.sleep(wait_time)
                        continue
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"✅ Gemini API SUCCESS with model {model_name} - using AI-generated summary")
                        return result
                    elif response.status_code == 404:
                        # Model not found - try next API version or next model
                        error_text = response.text[:200] if response.text else "No error text"
                        api_version = "v1" if "v1/models" in api_base else "v1beta"
                        print(f"[GEMINI] ❌ Model {model_name} not found (404) on {api_version}: {error_text}", file=sys.stderr)
                        logger.warning(f"Model {model_name} not found (404) on {api_version}, trying next...")
                        break  # Exit retry loop and try next API version
                    elif response.status_code == 401 or response.status_code == 403:
                        # Authentication error - API key issue, don't try other models
                        error_text = response.text[:200] if response.text else "No error text"
                        logger.error(f"Gemini API authentication failed ({response.status_code}): {error_text}")
                        logger.error("Check if API key is valid and has proper permissions")
                        return None
                    else:
                        error_text = response.text[:200] if response.text else "No error text"
                        logger.warning(f"Gemini API returned status {response.status_code}: {error_text}")
                        if attempt < max_retries - 1:
                            time.sleep(5)
                            continue
                        # If not 404, don't try other models (might be auth/rate limit issue)
                        return None
                        
                except requests.Timeout:
                    logger.warning(f"Gemini API timeout with {model_name} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    # Try next API version on timeout
                    break
                except requests.RequestException as exc:
                    error_msg = str(exc)
                    if hasattr(exc, 'response') and exc.response is not None:
                        try:
                            error_body = exc.response.text[:200]
                            error_msg = f"{error_msg} | Response: {error_body}"
                        except:
                            pass
                    logger.warning(f"Gemini API error with {model_name} (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    # Try next API version on request exception
                    break
                except Exception as exc:
                    logger.warning(f"Gemini API unexpected error with {model_name} (attempt {attempt + 1}/{max_retries}): {exc}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    # Try next API version on unexpected error
                    break
    
    # If we get here, all models failed
    print(f"[GEMINI] ❌ All models failed: {models_to_try}", file=sys.stderr)
    logger.warning(f"All Gemini models failed: {models_to_try}")
    return None


def summarize_insights(kpi_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert structured analytics outputs into human-readable insights using Gemini.
    
    This is the PRIMARY use of Gemini in the system - NLP task only.
    Takes structured KPI data and converts it to natural language summaries.
    
    Returns dict with: summary, prompt_used, fallback_used, ai_model_used.
    """
    settings = get_settings()
    prompt = envelope(
        system="You summarize KPIs for salon business leaders in the Philippines. Be concise and actionable. "
               "IMPORTANT: Always use Philippine Peso (₱) for all currency amounts. Never use dollars ($) or any other currency. "
               "All prices, revenue, costs, and monetary values must be displayed in Philippine Peso format (₱).",
        task="Summarize the key trends, call out risks, and recommend 2 quick actions.",
        context=kpi_summary,
    )
    log_prompt("insight", prompt)

    # Try Gemini API if key is available
    if settings.gemini_api_key:
        user_prompt = f"""Task: {prompt['task']}

Context (KPIs and metrics):
{prompt['context']}

Provide a concise business summary with 2-3 actionable recommendations.
IMPORTANT: Always use Philippine Peso (₱) for all currency amounts. Never use dollars ($) or any other currency."""

        try:
            print(f"[GEMINI] Attempting Gemini API call for insights...", file=sys.stderr)
            logger.info("Attempting Gemini API call for insights...")
            result = _call_gemini_api(user_prompt, max_tokens=500, temperature=0.3)
            print(f"[GEMINI] API call result: {result is not None}", file=sys.stderr)
            
            if result:
                # Gemini API returns: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if len(parts) > 0 and "text" in parts[0]:
                            generated_text = parts[0]["text"].strip()
                            if generated_text:
                                logger.info("✅ Gemini API SUCCESS - using AI-generated summary")
                                return {
                                    "summary": generated_text,
                                    "prompt_used": prompt,
                                    "fallback_used": False,
                                    "ai_model_used": GEMINI_MODEL,
                                    "gemini_ai_working": True,
                                }
                else:
                    logger.warning("Gemini API returned unexpected format")
        except Exception as e:
            logger.debug(f"Gemini API call exception: {e}")
    
    # Use intelligent fallback (works reliably when API is unavailable)
    logger.info("⚠️ Gemini API unavailable - using intelligent template-based fallback")
    fallback_result = _fallback_summary(prompt, "gemini-api-unavailable-or-failed")
    fallback_result["ai_model_used"] = "fallback-template"
    fallback_result["gemini_ai_working"] = False
    fallback_result["note"] = "Gemini API integration ready but currently unavailable. Using intelligent fallback."
    return fallback_result

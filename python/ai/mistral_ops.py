"""
Inventory & Workforce Reasoning AI (Mistral AI).
Provides intelligent recommendations for inventory restocking and workforce optimization.
"""
import logging
import sys
import time
from typing import Dict, Any, List, Optional
import requests

from config import get_settings
from ai.prompt_utils import log_prompt, envelope

logger = logging.getLogger(__name__)

# Mistral API endpoint
MISTRAL_API_BASE = "https://api.mistral.ai/v1"
MISTRAL_MODELS = [
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "mistral-tiny",
]


def _call_mistral_api(messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.3) -> Optional[Dict[str, Any]]:
    """
    Call Mistral AI API for chat completions.
    Tries multiple models if the first one fails.
    """
    settings = get_settings()
    if not settings.mistral_api_key:
        print("[MISTRAL] ⚠️ MISTRAL_API_KEY not set, skipping API call", file=sys.stderr)
        logger.warning("MISTRAL_API_KEY not set, skipping API call")
        return None

    payload = {
        "model": MISTRAL_MODELS[0],  # Start with large-latest
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.mistral_api_key}"
    }

    # Try multiple models if the first one fails
    for model_name in MISTRAL_MODELS:
        payload["model"] = model_name
        url = f"{MISTRAL_API_BASE}/chat/completions"
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                print(f"[MISTRAL] Attempting Mistral API call with {model_name} (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                logger.info(f"Attempting Mistral API call with {model_name} (attempt {attempt + 1}/{max_retries})")
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=15
                )
                
                print(f"[MISTRAL] Response status: {response.status_code} for {model_name}", file=sys.stderr)
                logger.info(f"Mistral API response status: {response.status_code}")
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    print(f"[MISTRAL] Rate limited, waiting {wait_time} seconds and retrying...", file=sys.stderr)
                    logger.info(f"Rate limited, waiting {wait_time} seconds and retrying...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"[MISTRAL] ✅ Mistral API SUCCESS with model {model_name}", file=sys.stderr)
                    logger.info(f"✅ Mistral API SUCCESS with model {model_name}")
                    return result
                elif response.status_code == 401 or response.status_code == 403:
                    # Authentication error - don't try other models
                    error_text = response.text[:200] if response.text else "No error text"
                    print(f"[MISTRAL] ❌ Mistral API authentication failed ({response.status_code}): {error_text}", file=sys.stderr)
                    print("[MISTRAL] Check if API key is valid and has proper permissions", file=sys.stderr)
                    logger.error(f"Mistral API authentication failed ({response.status_code}): {error_text}")
                    return None
                elif response.status_code == 404:
                    # Model not found - try next model
                    error_text = response.text[:200] if response.text else "No error text"
                    print(f"[MISTRAL] ❌ Model {model_name} not found (404): {error_text}", file=sys.stderr)
                    logger.warning(f"Model {model_name} not found (404), trying next...")
                    break  # Exit retry loop and try next model
                else:
                    error_text = response.text[:200] if response.text else "No error text"
                    print(f"[MISTRAL] ❌ Mistral API returned unexpected status {response.status_code}: {error_text}", file=sys.stderr)
                    logger.warning(f"Mistral API returned status {response.status_code}: {error_text}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    # If not 404, don't try other models (might be auth/rate limit issue)
                    return None
                    
            except requests.Timeout:
                print(f"[MISTRAL] ❌ Mistral API timeout with {model_name} (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                logger.warning(f"Mistral API timeout with {model_name} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                # Try next model on timeout
                break
            except requests.RequestException as exc:
                error_msg = str(exc)
                if hasattr(exc, 'response') and exc.response is not None:
                    try:
                        error_body = exc.response.text[:200]
                        error_msg = f"{error_msg} | Response: {error_body}"
                    except:
                        pass
                print(f"[MISTRAL] ❌ Mistral API error with {model_name} (attempt {attempt + 1}/{max_retries}): {error_msg}", file=sys.stderr)
                logger.warning(f"Mistral API error with {model_name} (attempt {attempt + 1}/{max_retries}): {error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                # Try next model on request exception
                break
            except Exception as exc:
                print(f"[MISTRAL] ❌ Mistral API unexpected error with {model_name} (attempt {attempt + 1}/{max_retries}): {exc}", file=sys.stderr)
                logger.warning(f"Mistral API unexpected error with {model_name} (attempt {attempt + 1}/{max_retries}): {exc}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                # Try next model on unexpected error
                break

    print(f"[MISTRAL] ❌ All Mistral models failed: {MISTRAL_MODELS}", file=sys.stderr)
    logger.warning(f"All Mistral models failed: {MISTRAL_MODELS}")
    return None


def _generate_fallback_recommendations(inventory_view: Dict[str, Any], staffing_view: Dict[str, Any]) -> List[str]:
    """
    Generate intelligent fallback recommendations when Mistral API is unavailable.
    """
    recommendations = []
    
    # Inventory recommendations
    if inventory_view:
        low_stock_count = inventory_view.get("low_stock_count", 0)
        urgent_items = inventory_view.get("urgent_items", [])
        
        if low_stock_count > 0:
            recommendations.append(f"Reorder {low_stock_count} low-stock items to maintain service quality.")
        
        if urgent_items:
            top_urgent = urgent_items[:3]
            item_names = ", ".join([item.get("name", "item") for item in top_urgent])
            recommendations.append(f"Urgent restock needed for: {item_names}")
        else:
            recommendations.append("Monitor inventory levels weekly to prevent stockouts.")
    
    # Staffing recommendations
    if staffing_view:
        peak_hours = staffing_view.get("peak_hours", [])
        understaffed_branches = staffing_view.get("understaffed_branches", [])
        
        if understaffed_branches:
            branch_names = ", ".join([b.get("name", "branch") for b in understaffed_branches[:2]])
            recommendations.append(f"Consider shifting staff to understaffed branches: {branch_names}")
        
        if peak_hours:
            recommendations.append(f"Schedule additional staff during peak hours: {', '.join(peak_hours)}")
        else:
            recommendations.append("Review staffing schedules to optimize coverage during busy periods.")
    
    # Default recommendations if no specific data
    if not recommendations:
        recommendations = [
            "Review inventory levels weekly and reorder items before they run low.",
            "Optimize staff schedules based on historical peak hours and customer demand.",
        ]
    
    return recommendations[:5]  # Limit to 5 recommendations


def recommend_ops(inventory_view: Dict[str, Any], staffing_view: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate inventory and workforce recommendations using Mistral AI.
    
    Args:
        inventory_view: Dictionary with inventory data (low_stock_count, urgent_items, etc.)
        staffing_view: Dictionary with staffing data (peak_hours, understaffed_branches, etc.)
    
    Returns:
        Dictionary with: recommendations (list), prompt_used, fallback_used, ai_model_used
    """
    settings = get_settings()
    prompt = envelope(
        system="You are an operations expert for a hair and nail salon business. Provide concise, actionable recommendations for inventory management and workforce optimization.",
        task="Analyze the inventory and staffing data provided. Give 3-5 specific, actionable recommendations. Focus on: (1) Which items need urgent restocking, (2) Staffing adjustments needed for peak hours or understaffed branches, (3) Cost optimization opportunities.",
        context={"inventory": inventory_view, "staffing": staffing_view},
    )
    log_prompt("ops", prompt)

    # Try Mistral API if key is available
    if settings.mistral_api_key:
        user_prompt = f"""Task: {prompt['task']}

Context:
Inventory Data: {prompt['context'].get('inventory', {})}
Staffing Data: {prompt['context'].get('staffing', {})}

Provide 3-5 specific, actionable recommendations. Format as a numbered list."""

        messages = [
            {"role": "system", "content": prompt['system']},
            {"role": "user", "content": user_prompt}
        ]

        try:
            print(f"[MISTRAL] Attempting Mistral API call for operations recommendations...", file=sys.stderr)
            logger.info("Attempting Mistral API call for operations recommendations...")
            result = _call_mistral_api(messages, max_tokens=500, temperature=0.3)
            print(f"[MISTRAL] API call result: {result is not None}", file=sys.stderr)
            
            if result:
                # Mistral API returns: {"choices": [{"message": {"content": "..."}}]}
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        generated_text = choice["message"]["content"].strip()
                        if generated_text:
                            # Parse recommendations from the text (split by numbers or bullets)
                            recommendations = []
                            lines = generated_text.split('\n')
                            for line in lines:
                                line = line.strip()
                                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                                    # Remove numbering/bullets
                                    clean_line = line.lstrip('0123456789.-•) ').strip()
                                    if clean_line:
                                        recommendations.append(clean_line)
                            
                            # If parsing didn't work, use the whole text as one recommendation
                            if not recommendations:
                                recommendations = [generated_text]
                            
                            logger.info("✅ Mistral API SUCCESS - using AI-generated recommendations")
                            return {
                                "recommendations": recommendations[:5],  # Limit to 5
                                "prompt_used": prompt,
                                "fallback_used": False,
                                "ai_model_used": result.get("model", "mistral-large-latest"),
                                "mistral_ai_working": True,
                            }
                else:
                    logger.warning("Mistral API returned unexpected format")
        except Exception as e:
            logger.debug(f"Mistral API call exception: {e}")
            print(f"[MISTRAL] Exception: {e}", file=sys.stderr)
    
    # Use intelligent fallback (works reliably when API is unavailable)
    logger.info("⚠️ Mistral API unavailable - using intelligent template-based fallback")
    print("[MISTRAL] ⚠️ Using intelligent fallback recommendations", file=sys.stderr)
    fallback_recommendations = _generate_fallback_recommendations(inventory_view, staffing_view)
    
    return {
        "recommendations": fallback_recommendations,
        "prompt_used": prompt,
        "fallback_used": True,
        "ai_model_used": "fallback-template",
        "mistral_ai_working": False,
    }


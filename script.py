import os, time, json, re, logging
from typing import Tuple, Dict, Any
import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-pro"  # change if needed
GEMINI_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
GEMINI_RETRIES = 3
GEMINI_BACKOFF_BASE = 1.5
logger = logging.getLogger("news-pipeline")

def call_gemini(prompt: str, max_output_tokens: int = 200) -> Tuple[bool, str]:
    """Call Gemini generateContent REST endpoint with retries. Returns (ok, text)."""
    if not GEMINI_API_KEY:
        return False, "no_api_key"
    endpoint = GEMINI_ENDPOINT_TEMPLATE.format(model=GEMINI_MODEL, key=GEMINI_API_KEY)
    headers = {"Content-Type": "application/json"}
    body = {
        "content": [{"parts": [{"text": prompt}]}],
        "temperature": 0.0,
        "maxOutputTokens": max_output_tokens
    }
    for attempt in range(1, GEMINI_RETRIES + 1):
        try:
            r = requests.post(endpoint, headers=headers, json=body, timeout=60)
            if r.status_code == 200:
                try:
                    j = r.json()
                except Exception:
                    return False, r.text[:2000]
                # Typical response shape: candidates -> content -> parts -> text
                try:
                    cands = j.get("candidates") or j.get("candidates", [])
                    if isinstance(cands, list) and cands:
                        first = cands[0]
                        content = first.get("content") or {}
                        parts = content.get("parts") or []
                        if isinstance(parts, list) and parts:
                            text = parts[0].get("text") or ""
                            return True, text
                    # fallback: try output.candidates
                    out = j.get("output") or {}
                    ocands = out.get("candidates") or []
                    if ocands and isinstance(ocands[0], dict):
                        parts = ocands[0].get("content", {}).get("parts", [])
                        if parts:
                            return True, parts[0].get("text", "")
                    # fallback to stringified JSON
                    return True, json.dumps(j)
                except Exception:
                    return False, json.dumps(j)[:2000]
            # transient errors: backoff
            if r.status_code in (429, 500, 502, 503, 504):
                wait = GEMINI_BACKOFF_BASE ** attempt
                logger.warning(f"Gemini returned {r.status_code}. Backing off {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
                continue
            return False, f"status:{r.status_code} text:{r.text[:500]}"
        except requests.RequestException as e:
            wait = GEMINI_BACKOFF_BASE ** attempt
            logger.warning(f"Gemini request exception {e}. Backing off {wait:.1f}s (attempt {attempt})")
            time.sleep(wait)
    return False, "max_retries"

def _extract_json_or_heuristics(text: str) -> Dict[str, Any]:
    """Return dict parsed from JSON in text or heuristics for summary/label/score."""
    # 1) direct JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # 2) JSON substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    # 3) heuristics via regex
    result: Dict[str, Any] = {}
    m_summary = re.search(r'(?i)"?summary"?\s*[:=]\s*"(.*?)"', text, re.DOTALL)
    if m_summary:
        result["summary"] = m_summary.group(1).strip()
    else:
        m = re.search(r'(?i)summary[:=]\s*(.+?)(?:\n|$)', text)
        if m:
            result["summary"] = m.group(1).strip().strip('"')
    m_label = re.search(r'(?i)"?(sentiment_label|sentiment|label)"?\s*[:=]\s*"?([A-Za-z]+)"?', text)
    if m_label:
        result["sentiment_label"] = m_label.group(2).strip().lower()
    m_score = re.search(r'(?i)"?(sentiment_score|score|sentiment_score_0_100)"?\s*[:=]\s*([+-]?\d+(\.\d+)?)', text)
    if m_score:
        try:
            result["sentiment_score"] = float(m_score.group(2))
            return result
        except Exception:
            pass
    m_any = re.search(r'(?i)(score|sentiment)[^\d\-]{0,10}([+-]?\d+(\.\d+)?)', text)
    if m_any:
        try:
            result["sentiment_score"] = float(m_any.group(2))
        except Exception:
            pass
    return result

def ask_gemini_for_summary_and_sentiment(text: str) -> Tuple[str, str, float]:
    """Prompt Gemini to return strict JSON. Parse and normalize score to 0-100."""
    prompt = (
        "You are a concise financial assistant. Respond with a single valid JSON object and nothing else. "
        "The JSON must contain exactly these keys: \"summary\", \"sentiment_label\", \"sentiment_score\". "
        "\"summary\" must be 1-2 sentences. \"sentiment_label\" must be one of: positive, neutral, negative. "
        "\"sentiment_score\" must be a number from 0 to 100 (0 = most negative, 100 = most positive). "
        "Do not include any extra commentary or explanation. Input:\n\n" + text
    )
    ok, out = call_gemini(prompt, max_output_tokens=200)
    if not ok:
        logger.debug(f"Gemini call failed: {out}")
        return (text[:200], "neutral", 50.0)

    parsed = _extract_json_or_heuristics(out)
    summary = parsed.get("summary") or ""
    label = (parsed.get("sentiment_label") or parsed.get("sentiment") or "").lower()
    score = parsed.get("sentiment_score", None)

    # Normalize numeric score to 0-100
    score_val = None
    try:
        if score is None:
            score_val = None
        else:
            score_val = float(score)
            if 0.0 <= score_val <= 1.0:
                score_val = round(score_val * 100.0, 2)
            else:
                if -1.0 <= score_val <= 1.0:
                    score_val = round((score_val + 1.0) * 50.0, 2)
                else:
                    score_val = round(score_val, 2)
    except Exception:
        score_val = None

    if score_val is None:
        if label == "positive":
            score_val = 75.0
        elif label == "negative":
            score_val = 25.0
        else:
            score_val = 50.0

    if label not in ("positive", "neutral", "negative"):
        if score_val >= 66:
            label = "positive"
        elif score_val <= 34:
            label = "negative"
        else:
            label = "neutral"

    if not summary:
        summary = text[:200]

    return (summary, label, score_val)

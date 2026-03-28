#!/usr/bin/env python3
"""
script.py

- Reads tickers from tickers.txt (one ticker per line)
- Fetches Seeking Alpha RSS for each ticker
- Calls MiniMax-M2.5 via Hugging Face Inference API for summary + sentiment
- Writes one CSV per ticker: news_{TICKER}.csv
- Sentiment Score is scaled 0-100
- Skips duplicate URLs already present in each ticker CSV
- Includes retries, backoff, deterministic generation parameters, and robust parsing
"""

import os
import time
import json
import re
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any

import requests
import feedparser
import pandas as pd

# -------------------------
# Configuration
# -------------------------
CSV_TEMPLATE = "news_{ticker}.csv"
TICKERS_FILE = "tickers.txt"
SEEKINGALPHA_TEMPLATE = "https://seekingalpha.com/api/sa/combined/{ticker}.xml"

# Limits and timeouts
REQUEST_TIMEOUT = 20
PER_TICKER_SLEEP = 1.0
MAX_ARTICLES_PER_TICKER = 10

# Hugging Face / MiniMax
HF_API_KEY = os.environ.get("HF_API_KEY")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
# MiniMax text generation endpoint (model repo id)
HF_MINIMAX = "https://api-inference.huggingface.co/models/MiniMaxAI/MiniMax-M2.5"
HF_RETRIES = 3
HF_BACKOFF_BASE = 1.5  # seconds

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("news-pipeline")

# CSV columns
CSV_COLUMNS = ["Ticker", "Title", "URL", "Published", "Summary", "Sentiment", "Score", "FetchedAt"]

# -------------------------
# Utilities
# -------------------------
def load_tickers() -> List[str]:
    if not os.path.exists(TICKERS_FILE):
        logger.warning(f"{TICKERS_FILE} not found")
        return []
    with open(TICKERS_FILE, "r", encoding="utf-8") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    logger.info(f"Loaded {len(tickers)} tickers")
    return tickers

def fetch_feed_for_ticker(ticker: str) -> List[Any]:
    url = SEEKINGALPHA_TEMPLATE.format(ticker=ticker)
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "news-bot/1.0"})
        if r.status_code != 200:
            logger.warning(f"{ticker}: feed returned status {r.status_code}")
            return []
        feed = feedparser.parse(r.content)
        entries = feed.entries[:MAX_ARTICLES_PER_TICKER]
        logger.info(f"{ticker}: found {len(entries)} entries")
        return entries
    except Exception as e:
        logger.exception(f"{ticker}: error fetching feed: {e}")
        return []

# -------------------------
# MiniMax call + parsing (deterministic + robust)
# -------------------------
def call_minimax(prompt: str, max_new_tokens: int = 256) -> Tuple[bool, str]:
    """Call MiniMax model endpoint with deterministic params and retries. Returns (ok, text)."""
    if not HF_API_KEY:
        return False, "no_api_key"
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.0
        },
        "options": {"wait_for_model": True}
    }
    for attempt in range(1, HF_RETRIES + 1):
        try:
            r = requests.post(HF_MINIMAX, headers=HF_HEADERS, json=payload, timeout=60)
            if r.status_code == 200:
                try:
                    j = r.json()
                    # Common shapes: dict with generated_text, or list of dicts
                    if isinstance(j, dict) and "generated_text" in j:
                        return True, j["generated_text"]
                    if isinstance(j, list) and j and isinstance(j[0], dict) and "generated_text" in j[0]:
                        return True, j[0]["generated_text"]
                    # If the API returned a string or other JSON, convert to string
                    if isinstance(j, str):
                        return True, j
                    return True, json.dumps(j)
                except Exception:
                    return False, r.text[:2000]
            if r.status_code in (429, 500, 502, 503, 504):
                wait = HF_BACKOFF_BASE ** attempt
                logger.warning(f"MiniMax returned {r.status_code}. Backing off {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
                continue
            return False, f"status:{r.status_code} text:{r.text[:500]}"
        except requests.RequestException as e:
            wait = HF_BACKOFF_BASE ** attempt
            logger.warning(f"MiniMax request exception {e}. Backing off {wait:.1f}s (attempt {attempt})")
            time.sleep(wait)
    return False, "max_retries"

def parse_minimax_json(text: str) -> Dict[str, Any]:
    """Try to extract JSON from model output. Also extract numeric fields via regex if JSON missing."""
    # 1) direct JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2) find JSON substring
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

    # 3) regex heuristics for key:value pairs like sentiment_score: 0.87 or "score": 0.87
    result: Dict[str, Any] = {}

    # summary: capture between quotes after summary key or after "Summary:" lines
    m_summary = re.search(r'(?i)"?summary"?\s*[:=]\s*"(.*?)"', text, re.DOTALL)
    if m_summary:
        result["summary"] = m_summary.group(1).strip()
    else:
        m = re.search(r'(?i)summary[:=]\s*(.+?)(?:\n|$)', text)
        if m:
            result["summary"] = m.group(1).strip().strip('"')

    # sentiment label
    m_label = re.search(r'(?i)"?(sentiment_label|sentiment|label)"?\s*[:=]\s*"?([A-Za-z]+)"?', text)
    if m_label:
        result["sentiment_label"] = m_label.group(2).strip().lower()

    # numeric score patterns
    m_score = re.search(r'(?i)"?(sentiment_score|score|sentiment_score_0_100)"?\s*[:=]\s*([+-]?\d+(\.\d+)?)', text)
    if m_score:
        try:
            result["sentiment_score"] = float(m_score.group(2))
            return result
        except Exception:
            pass

    # try to find any standalone number 0-1 or 0-100 near the word score
    m_any = re.search(r'(?i)(score|sentiment)[^\d\-]{0,10}([+-]?\d+(\.\d+)?)', text)
    if m_any:
        try:
            result["sentiment_score"] = float(m_any.group(2))
        except Exception:
            pass

    return result

def ask_minimax_for_summary_and_sentiment(text: str) -> Tuple[str, str, float]:
    """Prompt MiniMax to return strict JSON. Parse and normalize score to 0-100."""
    prompt = (
        "You are a concise financial assistant. Respond with a single valid JSON object and nothing else. "
        "The JSON must contain exactly these keys: \"summary\", \"sentiment_label\", \"sentiment_score\". "
        "\"summary\" must be 1-2 sentences. \"sentiment_label\" must be one of: positive, neutral, negative. "
        "\"sentiment_score\" must be a number from 0 to 100 (0 = most negative, 100 = most positive). "
        "Do not include any extra commentary or explanation. Input:\n\n" + text
    )
    ok, out = call_minimax(prompt, max_new_tokens=200)
    if not ok:
        logger.debug(f"MiniMax call failed: {out}")
        return (text[:200], "neutral", 50.0)

    parsed = parse_minimax_json(out)
    summary = parsed.get("summary") or ""
    label = (parsed.get("sentiment_label") or parsed.get("sentiment") or "").lower()
    score = parsed.get("sentiment_score", None)

    # Normalize score
    score_val = None
    try:
        if score is None:
            score_val = None
        else:
            score_val = float(score)
            # If model returned 0-1, scale to 0-100
            if 0.0 <= score_val <= 1.0:
                score_val = round(score_val * 100.0, 2)
            else:
                # If model returned -1..1, map to 0..100
                if -1.0 <= score_val <= 1.0:
                    score_val = round((score_val + 1.0) * 50.0, 2)
                else:
                    score_val = round(score_val, 2)
    except Exception:
        score_val = None

    # If no numeric score, try heuristics from label
    if score_val is None:
        if label == "positive":
            score_val = 75.0
        elif label == "negative":
            score_val = 25.0
        else:
            score_val = 50.0

    # Normalize label if missing
    if label not in ("positive", "neutral", "negative"):
        if score_val >= 66:
            label = "positive"
        elif score_val <= 34:
            label = "negative"
        else:
            label = "neutral"

    # Final summary fallback
    if not summary:
        summary = text[:200]

    return (summary, label, score_val)

# -------------------------
# Article extraction + CSV handling
# -------------------------
def extract_article_row(ticker: str, entry: Any) -> Dict[str, Any]:
    title = getattr(entry, "title", "") or ""
    link = getattr(entry, "link", "") or ""
    published = getattr(entry, "published", "") or getattr(entry, "updated", "") or ""
    raw_text = getattr(entry, "summary", "") or title
    # Use MiniMax if API key present; otherwise fallback
    if HF_API_KEY:
        summary, label, score = ask_minimax_for_summary_and_sentiment(raw_text)
    else:
        summary = raw_text[:200] if raw_text else ""
        label, score = "", 0.0
    return {
        "Ticker": ticker,
        "Title": title,
        "URL": link,
        "Published": published,
        "Summary": summary,
        "Sentiment": label,
        "Score": score,
        "FetchedAt": datetime.utcnow().isoformat() + "Z"
    }

def load_existing_for_ticker(ticker: str) -> pd.DataFrame:
    fname = CSV_TEMPLATE.format(ticker=ticker)
    if os.path.exists(fname):
        try:
            df = pd.read_csv(fname)
            for c in CSV_COLUMNS:
                if c not in df.columns:
                    df[c] = ""
            return df[CSV_COLUMNS]
        except Exception:
            logger.warning(f"{fname} unreadable, starting fresh for {ticker}")
            return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.DataFrame(columns=CSV_COLUMNS)

# -------------------------
# Main
# -------------------------
def main():
    logger.info("Script start")
    tickers = load_tickers()
    if not tickers:
        logger.error("No tickers to process. Exiting.")
        return

    for ticker in tickers:
        logger.info(f"Processing {ticker}")
        existing = load_existing_for_ticker(ticker)
        existing_urls = set(existing["URL"].astype(str).tolist()) if not existing.empty else set()

        rows = []
        entries = fetch_feed_for_ticker(ticker)
        for e in entries:
            url = getattr(e, "link", "") or ""
            if not url:
                continue
            if url in existing_urls:
                logger.debug(f"{ticker}: skipping existing URL {url}")
                continue
            try:
                row = extract_article_row(ticker, e)
                rows.append(row)
            except Exception:
                logger.exception(f"{ticker}: error extracting article; skipping")
        time.sleep(PER_TICKER_SLEEP)

        if rows:
            df_new = pd.DataFrame(rows)
            df = pd.concat([df_new, existing], ignore_index=True).drop_duplicates(subset=["URL"])
        else:
            df = existing

        # Ensure columns and write CSV for this ticker
        for c in CSV_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        df = df[CSV_COLUMNS]
        fname = CSV_TEMPLATE.format(ticker=ticker)
        df.to_csv(fname, index=False)
        logger.info(f"Wrote {len(df)} rows to {fname}")

    logger.info("Script end")

if __name__ == "__main__":
    main()

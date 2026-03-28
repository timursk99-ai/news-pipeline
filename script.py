#!/usr/bin/env python3
"""
script.py

- Reads tickers from tickers.txt (one ticker per line)
- Fetches Seeking Alpha RSS for each ticker
- Calls MiniMax-M2.5 via Hugging Face Inference API for summary + sentiment
- Writes one CSV per ticker: news_{TICKER}.csv
- Sentiment Score is scaled 0-100
- Skips duplicate URLs already present in each ticker CSV
- Includes retries, backoff, and logging
"""

import os
import time
import json
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

def call_minimax(prompt: str) -> Tuple[bool, str]:
    """Call MiniMax model endpoint with retries. Returns (ok, text)."""
    if not HF_API_KEY:
        return False, "no_api_key"
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    for attempt in range(1, HF_RETRIES + 1):
        try:
            r = requests.post(HF_MINIMAX, headers=HF_HEADERS, json=payload, timeout=60)
            if r.status_code == 200:
                try:
                    # Many generation endpoints return JSON with 'generated_text' or a list
                    j = r.json()
                    # Try common shapes
                    if isinstance(j, dict) and "generated_text" in j:
                        return True, j["generated_text"]
                    if isinstance(j, list) and j and isinstance(j[0], dict) and "generated_text" in j[0]:
                        return True, j[0]["generated_text"]
                    # Otherwise return raw text if present
                    if isinstance(j, str):
                        return True, j
                    return True, json.dumps(j)[:2000]
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
    """Try to extract JSON from model output. Accepts raw JSON or text containing JSON."""
    # First try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find a JSON substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    # fallback empty
    return {}

def ask_minimax_for_summary_and_sentiment(text: str) -> Tuple[str, str, float]:
    """Prompt MiniMax to return JSON with summary, sentiment_label, sentiment_score (0-100)."""
    # Keep prompt short and deterministic
    prompt = (
        "You are a concise financial assistant. Given the input text, return a JSON object only, "
        "with keys: \"summary\", \"sentiment_label\", and \"sentiment_score\". "
        "\"summary\" should be a short 1-2 sentence summary. "
        "\"sentiment_label\" should be one of: positive, neutral, negative. "
        "\"sentiment_score\" should be a number from 0 to 100 representing sentiment strength (100 = most positive). "
        "Input:\n\n" + text
    )
    ok, out = call_minimax(prompt)
    if not ok:
        logger.debug(f"MiniMax call failed: {out}")
        # fallback: short summary and neutral score
        return (text[:200], "neutral", 50.0)
    parsed = parse_minimax_json(out)
    summary = parsed.get("summary") or parsed.get("Summary") or ""
    label = parsed.get("sentiment_label") or parsed.get("sentiment") or parsed.get("label") or ""
    score = parsed.get("sentiment_score") or parsed.get("score") or parsed.get("sentiment_score_0_100") or None
    # Normalize score
    try:
        if score is None:
            # try to infer from label if possible
            if label and label.lower() == "positive":
                score_val = 75.0
            elif label and label.lower() == "negative":
                score_val = 25.0
            else:
                score_val = 50.0
        else:
            score_val = float(score)
            # If model returned 0-1, scale to 0-100
            if 0.0 <= score_val <= 1.0:
                score_val = round(score_val * 100.0, 2)
            else:
                score_val = round(score_val, 2)
    except Exception:
        score_val = 50.0
    # Ensure label normalized
    label = (label or "").lower()
    if label not in ("positive", "neutral", "negative"):
        # try heuristics
        if score_val >= 66:
            label = "positive"
        elif score_val <= 34:
            label = "negative"
        else:
            label = "neutral"
    return (summary or text[:200], label, score_val)

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

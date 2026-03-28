#!/usr/bin/env python3
"""
script.py

- Reads tickers from tickers.txt (one ticker per line)
- Fetches Seeking Alpha RSS for each ticker
- Calls Hugging Face inference for summary and sentiment when HF_API_KEY is set
- Writes news_data.csv with columns: Ticker, Title, URL, Published, Summary, Sentiment, Score, FetchedAt
- Skips duplicate URLs already present in news_data.csv
- Includes retries, backoff, and logging
"""

import os
import time
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any

import requests
import feedparser
import pandas as pd

# -------------------------
# Configuration
# -------------------------
CSV_FILE = "news_data.csv"
TICKERS_FILE = "tickers.txt"
SEEKINGALPHA_TEMPLATE = "https://seekingalpha.com/api/sa/combined/{ticker}.xml"

# Limits and timeouts
REQUEST_TIMEOUT = 20
PER_TICKER_SLEEP = 1.0
MAX_ARTICLES_PER_TICKER = 10

# Hugging Face configuration
HF_API_KEY = os.environ.get("HF_API_KEY")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
HF_SUMMARIZER = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_SENTIMENT = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
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

def call_hf(endpoint: str, payload: dict) -> Tuple[bool, Any]:
    """Call Hugging Face inference endpoint with retries and backoff."""
    if not HF_API_KEY:
        return False, {"error": "no_api_key"}
    for attempt in range(1, HF_RETRIES + 1):
        try:
            r = requests.post(endpoint, headers=HF_HEADERS, json=payload, timeout=30)
            if r.status_code == 200:
                try:
                    return True, r.json()
                except Exception:
                    return False, {"error": "invalid_json", "text": r.text[:500]}
            if r.status_code in (429, 500, 502, 503, 504):
                wait = HF_BACKOFF_BASE ** attempt
                logger.warning(f"HF {endpoint} returned {r.status_code}. Backing off {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
                continue
            return False, {"error": "status", "status_code": r.status_code, "text": r.text[:500]}
        except requests.RequestException as e:
            wait = HF_BACKOFF_BASE ** attempt
            logger.warning(f"HF request exception {e}. Backing off {wait:.1f}s (attempt {attempt})")
            time.sleep(wait)
    return False, {"error": "max_retries"}

def summarize_text(text: str) -> str:
    if not text:
        return ""
    ok, out = call_hf(HF_SUMMARIZER, {"inputs": text[:2000]})
    if not ok:
        logger.debug(f"Summarizer failed: {out}")
        return text[:200]  # fallback
    if isinstance(out, list) and out and isinstance(out[0], dict):
        return out[0].get("summary_text", "") or text[:200]
    if isinstance(out, dict) and "summary_text" in out:
        return out.get("summary_text", "") or text[:200]
    return str(out)[:200]

def sentiment_text(text: str) -> Tuple[str, float]:
    if not text:
        return "", 0.0
    ok, out = call_hf(HF_SENTIMENT, {"inputs": text[:2000]})
    if not ok:
        logger.debug(f"Sentiment failed: {out}")
        return "", 0.0
    try:
        if isinstance(out, list) and out and isinstance(out[0], dict):
            label = out[0].get("label", "")
            score = float(out[0].get("score", 0.0))
            return label, score
        if isinstance(out, dict) and "label" in out:
            return out.get("label", ""), float(out.get("score", 0.0))
    except Exception:
        logger.exception("Error parsing HF sentiment response")
    return "", 0.0

def extract_article_row(ticker: str, entry: Any) -> Dict[str, Any]:
    title = getattr(entry, "title", "") or ""
    link = getattr(entry, "link", "") or ""
    published = getattr(entry, "published", "") or getattr(entry, "updated", "") or ""
    raw_text = getattr(entry, "summary", "") or title
    # Use HF only if API key present; otherwise fallback to raw text
    summary = summarize_text(raw_text) if HF_API_KEY else (raw_text[:200] if raw_text else "")
    label, score = sentiment_text(raw_text) if HF_API_KEY else ("", 0.0)
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

def load_existing() -> pd.DataFrame:
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            # Ensure expected columns exist
            for c in CSV_COLUMNS:
                if c not in df.columns:
                    df[c] = ""
            return df[CSV_COLUMNS]
        except Exception:
            logger.warning("Existing CSV unreadable, starting fresh")
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

    existing = load_existing()
    existing_urls = set(existing["URL"].astype(str).tolist()) if not existing.empty else set()

    rows = []
    for ticker in tickers:
        entries = fetch_feed_for_ticker(ticker)
        for e in entries:
            url = getattr(e, "link", "") or ""
            if not url:
                continue
            if url in existing_urls:
                logger.debug(f"Skipping existing URL {url}")
                continue
            try:
                row = extract_article_row(ticker, e)
                rows.append(row)
            except Exception:
                logger.exception(f"Error extracting article for {ticker}; skipping")
        time.sleep(PER_TICKER_SLEEP)

    if rows:
        df_new = pd.DataFrame(rows)
        df = pd.concat([df_new, existing], ignore_index=True).drop_duplicates(subset=["URL"])
    else:
        df = existing

    # Ensure columns and write CSV
    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[CSV_COLUMNS]
    df.to_csv(CSV_FILE, index=False)
    logger.info(f"Wrote {len(df)} rows to {CSV_FILE}")
    logger.info("Script end")

if __name__ == "__main__":
    main()

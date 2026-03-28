#!/usr/bin/env python3
"""
script.py

- Reads tickers from tickers.txt (one ticker per line)
- Fetches Seeking Alpha RSS for each ticker
- Calls Hugging Face inference for summary and sentiment when HF_API_KEY is set
- Writes one CSV per ticker: news_{TICKER}.csv
- Sentiment Score is scaled 0-100
- Skips duplicate URLs already present in each ticker CSV
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
CSV_TEMPLATE = "news_{ticker}.csv"
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
HF_SENTIMENT = "https://api-inference.huggingface.co/models/peejm/finbert-financial-sentiment"
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
    """Return (label, score_0_100). Score scaled to 0-100."""
    if not text:
        return "", 0.0
    ok, out = call_hf(HF_SENTIMENT, {"inputs": text[:2000]})
    if not ok:
        logger.debug(f"Sentiment failed: {out}")
        return "", 0.0
    # peejm/finbert-financial-sentiment returns a list of dicts like [{"label":"positive","score":0.9}, ...]
    try:
        if isinstance(out, list) and out and isinstance(out[0], dict):
            label = out[0].get("label", "")
            score01 = float(out[0].get("score", 0.0))
            score100 = round(score01 * 100, 2)
            return label, score100
        if isinstance(out, dict) and "label" in out:
            label = out.get("label", "")
            score01 = float(out.get("score", 0.0))
            return label, round(score01 * 100, 2)
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

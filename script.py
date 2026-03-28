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

REQUEST_TIMEOUT = 20
PER_TICKER_SLEEP = 1.0
MAX_ARTICLES_PER_TICKER = 10

HF_API_KEY = os.environ.get("HF_API_KEY")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
HF_MINIMAX = "https://api-inference.huggingface.co/models/MiniMaxAI/MiniMax-M2.5"
HF_RETRIES = 3
HF_BACKOFF_BASE = 1.5  # seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("news-pipeline")

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
# MiniMax call + parsing
# -------------------------
def call_minimax(prompt: str, max_new_tokens: int = 256) -> Tuple[bool, str]:
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
                    if isinstance(j, dict) and "generated_text" in j:
                        return True, j["generated_text"]
                    if isinstance(j, list) and j and isinstance(j[0], dict) and "generated_text" in j[0]:
                        return True, j[0]["generated_text"]
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
    """Robust JSON parsing from MiniMax output"""
    # 1) direct JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2) regex to find JSON-like blocks containing summary
    matches = re.findall(r"\{.*?summary.*?\}", text, flags=re.DOTALL)
    for m in matches:
        try:
            parsed = json.loads(m)
            if all(k in parsed for k in ["summary","sentiment_label","sentiment_score"]):
                return parsed
        except Exception:
            continue

    # 3) fallback heuristics
    result: Dict[str, Any] = {}
    m_summary = re.search(r'(?i)"?summary"?\s*[:=]\s*"(.*?)"', text, re.DOTALL)
    if m_summary: result["summary"] = m_summary.group(1).strip()
    m_label = re.search(r'(?i)"?(sentiment_label|sentiment|label)"?\s*[:=]\s*"?([A-Za-z]+)"?', text)
    if m_label: result["sentiment_label"] = m_label.group(2).strip().lower()
    m_score = re.search(r'(?i)"?(sentiment_score|score|sentiment_score_0_100)"?\s*[:=]\s*([+-]?\d+(\.\d+)?)', text)
    if m_score:
        try: result["sentiment_score"] = float(m_score.group(2))
        except: pass
    return result

def ask_minimax_for_summary_and_sentiment(text: str) -> Tuple[str, str, float]:
    prompt = (
        "You are a concise financial assistant. Respond with a single valid JSON object and nothing else. "
        "The JSON must contain exactly these keys: \"summary\", \"sentiment_label\", \"sentiment_score\". "
        "\"summary\" must be 1-2 sentences. \"sentiment_label\" must be one of: positive, neutral, negative. "
        "\"sentiment_score\" must be a number from 0 to 100 (0 = most negative, 100 = most positive). "
        "Do not include any extra commentary or explanation. Input:\n\n" + text
    )
    ok, out = call_minimax(prompt, max_new_tokens=200)
    if not ok:
        logger.warning(f"MiniMax call failed: {out}")
        return (text[:200], "neutral", 50.0)

    logger.debug(f"MiniMax raw output: {out}")

    parsed = parse_minimax_json(out)
    summary = parsed.get("summary") or text[:200]
    label = (parsed.get("sentiment_label") or "").lower()
    score = parsed.get("sentiment_score", None)

    # Normalize score to 0-100
    score_val = None
    try:
        if score is None:
            score_val = None
        else:
            score_val = float(score)
            if 0.0 <= score_val <= 1.0:
                score_val = round(score_val * 100.0, 2)
            elif -1.0 <= score_val <= 1.0:
                score_val = round((score_val + 1.0) * 50.0, 2)
            else:
                score_val = round(score_val, 2)
    except:
        score_val = None

    # Fallback heuristics if no score
    if score_val is None:
        if label == "positive": score_val = 75.0
        elif label == "negative": score_val = 25.0
        else: score_val = 50.0

    # Normalize label
    if label not in ("positive","neutral","negative"):
        if score_val >= 66: label = "positive"
        elif score_val <= 34: label = "negative"
        else: label = "neutral"

    return summary, label, score_val

# -------------------------
# Article extraction + CSV handling
# -------------------------
def extract_article_row(ticker: str, entry: Any) -> Dict[str, Any]:
    title = getattr(entry, "title", "") or ""
    link = getattr(entry, "link", "") or ""
    published = getattr(entry, "published", "") or getattr(entry, "updated", "") or ""
    raw_text = getattr(entry, "summary", "") or title
    if HF_API_KEY:
        summary, label, score = ask_minimax_for_summary_and_sentiment(raw_text)
    else:
        summary = raw_text[:200]
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
        except:
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
            if not url or url in existing_urls:
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

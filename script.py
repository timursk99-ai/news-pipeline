#!/usr/bin/env python3
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

# 🔥 MODEL FALLBACK SYSTEM
GEMINI_MODELS = [
    "gemini-2.5-flash",   # will fail if no quota
    "gemini-2.0-flash",   # main working model
    "gemini-1.5-flash-8b" # guaranteed fallback
]

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

GEMINI_RETRIES = 3
GEMINI_BACKOFF_BASE = 1.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("news-pipeline")

CSV_COLUMNS = ["Ticker", "Title", "URL", "Published", "Summary", "Sentiment", "Score", "FetchedAt"]

# -------------------------
# Utilities
# -------------------------
def load_tickers() -> List[str]:
    if not os.path.exists(TICKERS_FILE):
        return []
    with open(TICKERS_FILE, "r") as f:
        return [x.strip().upper() for x in f if x.strip()]

def fetch_feed_for_ticker(ticker: str):
    url = SEEKINGALPHA_TEMPLATE.format(ticker=ticker)
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        feed = feedparser.parse(r.content)
        return feed.entries[:MAX_ARTICLES_PER_TICKER]
    except:
        return []

# -------------------------
# GEMINI CORE (FIXED)
# -------------------------
def call_gemini(prompt: str) -> Tuple[bool, str]:
    if not GEMINI_API_KEY:
        return False, "no_api_key"

    headers = {"Content-Type": "application/json"}

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 200
        }
    }

    for model in GEMINI_MODELS:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"

        for attempt in range(1, GEMINI_RETRIES + 1):
            try:
                r = requests.post(endpoint, headers=headers, json=body, timeout=60)

                if r.status_code == 200:
                    j = r.json()
                    try:
                        return True, j["candidates"][0]["content"]["parts"][0]["text"]
                    except:
                        return True, json.dumps(j)

                # 🔥 QUOTA HANDLING
                if r.status_code == 429 or "quota" in r.text.lower():
                    logger.warning(f"{model} quota unavailable → switching")
                    break

                if r.status_code in (500, 502, 503, 504):
                    time.sleep(GEMINI_BACKOFF_BASE ** attempt)
                    continue

                return False, f"{model} failed: {r.status_code}"

            except requests.RequestException:
                time.sleep(GEMINI_BACKOFF_BASE ** attempt)

    return False, "all_models_failed"

# -------------------------
# PARSING
# -------------------------
def parse_output(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass

    return {}

# -------------------------
# MAIN AI FUNCTION
# -------------------------
def analyze(text: str) -> Tuple[str, str, float]:
    prompt = (
        "Return ONLY JSON:\n"
        "{\"summary\": \"...\", \"sentiment_label\": \"positive|neutral|negative\", \"sentiment_score\": number}\n\n"
        f"Text:\n{text}"
    )

    ok, out = call_gemini(prompt)

    if not ok:
        return text[:200], "neutral", 50.0

    parsed = parse_output(out)

    summary = parsed.get("summary", text[:200])
    label = parsed.get("sentiment_label", "neutral").lower()
    score = parsed.get("sentiment_score", 50)

    try:
        score = float(score)
        if 0 <= score <= 1:
            score *= 100
        elif -1 <= score <= 1:
            score = (score + 1) * 50
    except:
        score = 50

    if label not in ["positive", "neutral", "negative"]:
        label = "neutral"

    return summary, label, round(score, 2)

# -------------------------
# PIPELINE
# -------------------------
def extract_row(ticker: str, entry):
    text = getattr(entry, "summary", "") or entry.title
    summary, label, score = analyze(text)

    return {
        "Ticker": ticker,
        "Title": entry.title,
        "URL": entry.link,
        "Published": getattr(entry, "published", ""),
        "Summary": summary,
        "Sentiment": label,
        "Score": score,
        "FetchedAt": datetime.utcnow().isoformat()
    }

# -------------------------
# MAIN
# -------------------------
def main():
    tickers = load_tickers()

    for ticker in tickers:
        existing = pd.read_csv(CSV_TEMPLATE.format(ticker=ticker)) if os.path.exists(CSV_TEMPLATE.format(ticker=ticker)) else pd.DataFrame(columns=CSV_COLUMNS)
        seen = set(existing["URL"]) if not existing.empty else set()

        rows = []
        for e in fetch_feed_for_ticker(ticker):
            if e.link in seen:
                continue
            rows.append(extract_row(ticker, e))

        df = pd.concat([pd.DataFrame(rows), existing]).drop_duplicates("URL")

        df.to_csv(CSV_TEMPLATE.format(ticker=ticker), index=False)
        print(f"{ticker}: {len(df)} rows")

        time.sleep(PER_TICKER_SLEEP)

if __name__ == "__main__":
    main()

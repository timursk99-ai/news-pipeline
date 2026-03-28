import requests
import feedparser
import pandas as pd
import os
import time
import logging
from datetime import datetime

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Config
CSV_FILE = "news_data.csv"
TICKERS_FILE = "tickers.txt"
SEEKINGALPHA_TEMPLATE = "https://seekingalpha.com/api/sa/combined/{ticker}.xml"
REQUEST_TIMEOUT = 20
PER_TICKER_SLEEP = 1.0
MAX_ARTICLES_PER_TICKER = 10

def load_tickers():
    if not os.path.exists(TICKERS_FILE):
        logging.warning(f"{TICKERS_FILE} not found")
        return []
    with open(TICKERS_FILE, "r", encoding="utf-8") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    logging.info(f"Loaded {len(tickers)} tickers")
    return tickers

def fetch_feed_for_ticker(ticker):
    url = SEEKINGALPHA_TEMPLATE.format(ticker=ticker)
    try:
        # Use requests to check reachability then feedparser to parse
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "news-bot/1.0"})
        if r.status_code != 200:
            logging.warning(f"Ticker {ticker} feed returned status {r.status_code}")
            return []
        feed = feedparser.parse(r.content)
        entries = feed.entries[:MAX_ARTICLES_PER_TICKER]
        logging.info(f"{ticker}: found {len(entries)} entries")
        return entries
    except Exception as e:
        logging.exception(f"Error fetching feed for {ticker}: {e}")
        return []

def summarize_stub(text):
    # Lightweight fallback summary for cloud runs without external NLP
    return text if text else ""

def extract_article_row(ticker, entry):
    title = getattr(entry, "title", "")
    link = getattr(entry, "link", "")
    published = getattr(entry, "published", "") or getattr(entry, "updated", "")
    summary = getattr(entry, "summary", "") or summarize_stub(title)
    return {
        "Ticker": ticker,
        "Title": title,
        "URL": link,
        "Published": published,
        "Summary": summary,
        "FetchedAt": datetime.utcnow().isoformat() + "Z"
    }

def load_existing():
    if os.path.exists(CSV_FILE):
        try:
            return pd.read_csv(CSV_FILE)
        except Exception:
            logging.warning("Existing CSV unreadable, starting fresh")
            return pd.DataFrame(columns=["Ticker","Title","URL","Published","Summary","FetchedAt"])
    return pd.DataFrame(columns=["Ticker","Title","URL","Published","Summary","FetchedAt"])

def main():
    logging.info("Script start")
    tickers = load_tickers()
    if not tickers:
        logging.error("No tickers to process. Exiting.")
        return

    existing = load_existing()
    existing_urls = set(existing["URL"].astype(str).tolist()) if not existing.empty else set()

    rows = []
    for ticker in tickers:
        entries = fetch_feed_for_ticker(ticker)
        for e in entries:
            url = getattr(e, "link", "")
            if not url:
                continue
            if url in existing_urls:
                logging.debug(f"Skipping existing URL {url}")
                continue
            row = extract_article_row(ticker, e)
            rows.append(row)
        time.sleep(PER_TICKER_SLEEP)

    if rows:
        df_new = pd.DataFrame(rows)
        df = pd.concat([df_new, existing], ignore_index=True).drop_duplicates(subset=["URL"])
    else:
        df = existing

    # Always write CSV so file exists for commit step
    df.to_csv(CSV_FILE, index=False)
    logging.info(f"Wrote {len(df)} rows to {CSV_FILE}")
    logging.info("Script end")

if __name__ == "__main__":
    main()

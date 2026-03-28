import feedparser
import requests
import pandas as pd
import os
import time

HF_API_KEY = os.environ.get("HF_API_KEY")
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

SUMMARIZER = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
SENTIMENT = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

CSV_FILE = "news_data.csv"

def get_articles():
    feed = feedparser.parse("https://feeds.reuters.com/reuters/businessNews")
    return feed.entries[:5]

def summarize(text):
    if not HEADERS:
        return ""
    try:
        res = requests.post(SUMMARIZER, headers=HEADERS, json={"inputs": text[:2000]}, timeout=30)
        return res.json()[0].get("summary_text", "")
    except Exception:
        return ""

def sentiment(text):
    if not HEADERS:
        return "", 0.0
    try:
        res = requests.post(SENTIMENT, headers=HEADERS, json={"inputs": text[:2000]}, timeout=30)
        out = res.json()[0]
        return out.get("label", ""), float(out.get("score", 0.0))
    except Exception:
        return "", 0.0

def load_existing():
    if os.path.exists(CSV_FILE):
        try:
            return pd.read_csv(CSV_FILE)
        except Exception:
            return pd.DataFrame(columns=["Title","URL","Summary","Sentiment","Score"])
    return pd.DataFrame(columns=["Title","URL","Summary","Sentiment","Score"])

def main():
    articles = get_articles()
    existing = load_existing()
    existing_urls = set(existing["URL"].tolist()) if not existing.empty else set()
    new_rows = []

    for a in articles:
        url = getattr(a, "link", "")
        title = getattr(a, "title", "")
        if not url or url in existing_urls:
            continue

        summary = summarize(title)
        label, score = sentiment(title)

        new_rows.append({
            "Title": title,
            "URL": url,
            "Summary": summary,
            "Sentiment": label,
            "Score": score
        })

        time.sleep(1)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df = pd.concat([df_new, existing], ignore_index=True).drop_duplicates(subset=["URL"])
    else:
        df = existing

    # Always write CSV so file exists in repo
    df.to_csv(CSV_FILE, index=False)

if __name__ == "__main__":
    main()

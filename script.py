import feedparser
import requests
import pandas as pd
import os
import time

HF_API_KEY = os.environ.get("HF_API_KEY")

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Hugging Face endpoints
SUMMARIZER = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
SENTIMENT = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

CSV_FILE = "news_data.csv"

def get_articles():
    feed = feedparser.parse("https://feeds.reuters.com/reuters/businessNews")
    return feed.entries[:5]

def summarize(text):
    try:
        res = requests.post(SUMMARIZER, headers=HEADERS, json={"inputs": text[:2000]})
        return res.json()[0]["summary_text"]
    except:
        return "ERROR"

def sentiment(text):
    try:
        res = requests.post(SENTIMENT, headers=HEADERS, json={"inputs": text[:2000]})
        out = res.json()[0]
        return out["label"], out["score"]
    except:
        return "ERROR", 0

def load_existing():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=["Title","URL","Summary","Sentiment","Score"])

def main():
    articles = get_articles()
    existing = load_existing()

    existing_urls = set(existing["URL"].tolist())

    new_rows = []

    for a in articles:
        if a.link in existing_urls:
            continue

        text = a.title

        summary = summarize(text)
        label, score = sentiment(text)

        new_rows.append({
            "Title": a.title,
            "URL": a.link,
            "Summary": summary,
            "Sentiment": label,
            "Score": score
        })

        time.sleep(2)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df = pd.concat([df_new, existing]).drop_duplicates(subset=["URL"])
        df.to_csv(CSV_FILE, index=False)

if __name__ == "__main__":
    main()

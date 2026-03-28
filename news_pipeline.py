import feedparser
import requests
import gspread
import os
import json
import time
from oauth2client.service_account import ServiceAccountCredentials
from newspaper import Article

# ==============================
# CONFIG
# ==============================

RSS_URL = "https://feeds.reuters.com/reuters/businessNews"

HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise ValueError("Missing HF_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

# ==============================
# GOOGLE SHEETS CONNECTION
# ==============================

def connect_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]

    if os.getenv("GOOGLE_CREDENTIALS"):
        creds_dict = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
        with open("temp_creds.json", "w") as f:
            json.dump(creds_dict, f)
        creds = ServiceAccountCredentials.from_json_keyfile_name("temp_creds.json", scope)
    else:
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)

    client = gspread.authorize(creds)
    sheet = client.open("News Analysis").sheet1
    return sheet

# ==============================
# FETCH NEWS
# ==============================

def fetch_news():
    feed = feedparser.parse(RSS_URL)
    articles = []

    for entry in feed.entries[:5]:
        articles.append({
            "title": entry.title,
            "link": entry.link
        })

    return articles

# ==============================
# EXTRACT ARTICLE TEXT
# ==============================

def extract_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text[:2000]  # limit size
    except:
        return None

# ==============================
# HUGGING FACE NLP
# ==============================

def summarize(text):
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

    response = requests.post(url, headers=HEADERS, json={"inputs": text})
    try:
        return response.json()[0]["summary_text"]
    except:
        return "Summary failed"

def get_sentiment(text):
    url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

    response = requests.post(url, headers=HEADERS, json={"inputs": text})
    try:
        return response.json()[0]["label"], response.json()[0]["score"]
    except:
        return "ERROR", 0

# ==============================
# MAIN ANALYSIS
# ==============================

def analyze_article(url):
    text = extract_text(url)

    if not text:
        return "ERROR", "ERROR", 0

    summary = summarize(text)
    sentiment_label, sentiment_score = get_sentiment(text)

    return summary, sentiment_label, sentiment_score

# ==============================
# MAIN PIPELINE
# ==============================

def main():
    sheet = connect_sheet()
    articles = fetch_news()

    existing_links = sheet.col_values(2)[-100:]

    for article in articles:
        if article["link"] in existing_links:
            continue

        print(f"Processing: {article['title']}")

        summary, sentiment_label, sentiment_score = analyze_article(article["link"])

        sheet.append_row([
            article["title"],
            article["link"],
            summary,
            sentiment_label,
            sentiment_score
        ])

        time.sleep(3)  # rate limiting

# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    main()
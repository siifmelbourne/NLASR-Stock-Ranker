import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline
from datetime import datetime
import feedparser

RSS_URLS = ["https://news.google.com/rss/search?q=site%3Areuters.com&hl=en-US&gl=US&ceid=US%3Aen"]
tickers = input("Enter tickers separated by spaces(e.g. TLX.AX BHP.AX): ").split()
HEADERS = {"User-Agent": "Mozilla/5.0"}

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

def sentiment_score(text):
    """Calculate sentiment score for given text"""
    if not text:
        return 0

    result = sentiment_pipeline(text[:512])[0]
    label = result["label"]
    score = result["score"]

    if label == "positive":
        return score
    elif label == "negative":
        return -score
    return 0

def get_article_text(url):
    """Fetch article text from given URL"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"Failed to fetch {url}: Status code {r.status_code}")
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        return text
    except:
        return ""

data = []

for rss in RSS_URLS:
    feed = feedparser.parse(rss)

    for entry in feed.entries:
        title = entry.get("title", "")
        link = entry.get("link", "")
        published = entry.get("published", "")
        # convert date
        try:
            date = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z").strftime("%Y-%m")
        except:
            date = "unknown"
        get_article_text(link)
        content = get_article_text(link)

        for ticker in tickers:
            if ticker.split(".")[0] in (title + content):
                data.append((date, ticker, title, content))

df = pd.DataFrame(data, columns=["Date (YYYY-MM)", "Ticker", "Title", "Content"])

df["Sentiment"] = df["Content"].apply(sentiment_score)

sentiment_summary = df.groupby(["Date (YYYY-MM)", "Ticker"])["Sentiment"].mean().reset_index()

df.to_csv("reuters_news_sentiment.csv", index=False)
sentiment_summary.to_csv("reuters_sentiment_summaries.csv", index=False)
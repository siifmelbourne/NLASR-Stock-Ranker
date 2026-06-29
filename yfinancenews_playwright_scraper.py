#import requests # This got blocked, so used playwright as fallback
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline
from playwright.sync_api import sync_playwright
# otherwise can be hard-coded to a list
# To update every month add and update data to SQLite databasehis    
tickers = input("Enter stock tickers separated by spaces (add .AX suffix): ").split()
data = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)

    for ticker in tickers:
        page = browser.new_page()
        page.goto(f"https://finance.yahoo.com/quote/{ticker}/news")
        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        articles = soup.find_all("li")

        for article in articles:
            h3 = article.find("h3")
            a = article.find("a")
            if not h3 or not a:
                continue
            title = h3.text.strip()
            link = a["href"]

            #make sure link is formatted correctly
            if link.startswith("/"):
                link = "https://finance.yahoo.com" + link

            # article pages:
            try:
                article_page = browser.new_page()
                article_page.goto(link, wait_until="domcontentloaded")
                article_page.wait_for_timeout(2000)
                article_html = article_page.content()
                article_page.close()
                article_soup = BeautifulSoup(article_html, "html.parser")
                time_tag = article_soup.find("time")
                date = time_tag["datetime"][:7] if time_tag and "datetime" in time_tag.attrs else "unknown"
                paragraphs = article_soup.find_all("p")
                content = " ".join(p.text.strip() for p in paragraphs)

                data.append((date, ticker, title, content))
            except Exception:
                continue

        page.close()

    browser.close()

df = pd.DataFrame(data, columns=["Date (YYYY-MM)", "Ticker", "Title", "Content"])

# finbert sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def sentiment_score(text):
    """Calculate sentiment score for given text"""
    if not text:
        return 0
    result = sentiment_pipeline(text[:512])[0]  # avoid token overflow
    label = result["label"]
    score = result["score"]
    #convert to score based on confidence
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    return 0

df["Sentiment"] = df["Content"].apply(sentiment_score)

sentiment_df_summary = df.groupby(["Date (YYYY-MM)", "Ticker"])["Sentiment"].mean().reset_index()
sentiment_df_summary.to_csv("yahoo_news_sentiments.csv", index=False)

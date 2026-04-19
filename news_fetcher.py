import requests

# ── CONFIG ───────────────────────────────────────────────
NEWS_API_KEY = "pub_3847b7c7931f4f589c2f958b054f38c8"
NEWS_API_URL = "https://newsdata.io/api/1/news"


def fetch_live_news(query: str = "latest", page_size: int = 10) -> list[dict]:
    """
    Search news articles using NewsData.io API
    Returns list of {title, source, url, text, published_at}
    """

    params = {
        "apikey": NEWS_API_KEY,
        "q": query,
        "language": "en",
        "size": min(page_size, 50)
    }

    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=10)
        data = resp.json()

        # Debug (optional)
        print("API Response:", data)

        if data.get("status") != "success":
            print("API Error:", data)
            return []

        return _parse_articles(data.get("results", []))

    except Exception as e:
        print("API Connection Error:", e)
        return []


def fetch_top_headlines(category: str = "top", page_size: int = 10) -> list[dict]:
    """
    Fetch top headlines using NewsData API
    """

    params = {
        "apikey": NEWS_API_KEY,
        "category": category,
        "language": "en",
        "size": min(page_size, 50)
    }

    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=10)
        data = resp.json()

        if data.get("status") != "success":
            print("API Error:", data)
            return []

        return _parse_articles(data.get("results", []))

    except Exception as e:
        print("API Connection Error:", e)
        return []


def _parse_articles(articles: list) -> list[dict]:
    """
    Convert API response to clean format.
    Models were trained on TITLE text.
    """

    results = []

    for a in articles:
        title = (a.get("title") or "").strip()
        description = (a.get("description") or "").strip()

        if not title:
            continue

        full_text = f"{title} {description}".strip()

        results.append({
            "title": title,
            "text": full_text,
            "title_only": title,
            "source": a.get("source_id", "Unknown"),
            "url": a.get("link", "#"),
            "published_at": (a.get("pubDate") or "")[:10]
        })

    return results
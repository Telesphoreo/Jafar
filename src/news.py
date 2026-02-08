"""
News fetching module for Jafar.

Fetches economic news headlines via DuckDuckGo's news search
to provide a daily news roundup alongside Twitter signal detection.
"""

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger("jafar.news")


@dataclass
class NewsArticle:
    """A single news article from DuckDuckGo news search."""
    title: str
    body: str       # snippet
    url: str
    source: str
    date: str


async def fetch_economic_news(
    queries: list[str],
    max_results_per_query: int = 5,
) -> list[NewsArticle]:
    """
    Fetch economic news headlines via DuckDuckGo news search.

    Runs DDGS in executor (synchronous library) and deduplicates by URL.

    Args:
        queries: List of search queries to run.
        max_results_per_query: Max results per query.

    Returns:
        Deduplicated list of NewsArticle objects.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs not installed. News fetching disabled.")
        return []

    articles: list[NewsArticle] = []
    seen_urls: set[str] = set()

    loop = asyncio.get_running_loop()

    for query in queries:
        try:
            ddgs = DDGS()

            def run_news_search():
                return ddgs.news(query, max_results=max_results_per_query)

            results = await loop.run_in_executor(None, run_news_search)

            if not results:
                continue

            for r in results:
                url = r.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                articles.append(NewsArticle(
                    title=r.get("title", "No Title"),
                    body=r.get("body", ""),
                    url=url,
                    source=r.get("source", "Unknown"),
                    date=r.get("date", ""),
                ))

        except Exception as e:
            logger.warning(f"News fetch failed for query '{query}': {e}")
            continue

    logger.info(f"Fetched {len(articles)} news articles from {len(queries)} queries")
    return articles


def format_news_for_llm(articles: list[NewsArticle]) -> str:
    """
    Format news articles into a structured string for LLM consumption.

    Args:
        articles: List of NewsArticle objects.

    Returns:
        Formatted string with headlines and snippets.
    """
    if not articles:
        return ""

    parts = [f"## Today's Economic News Headlines ({len(articles)} articles)\n"]

    for i, article in enumerate(articles, 1):
        parts.append(f"{i}. **{article.title}**")
        if article.body:
            parts.append(f"   {article.body}")
        parts.append(f"   Source: {article.source} | {article.date}")
        parts.append("")

    return "\n".join(parts)

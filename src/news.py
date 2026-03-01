"""
News fetching module for Jafar.

Fetches economic news headlines via DuckDuckGo's news search
to provide a daily news roundup alongside Twitter signal detection.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from dateutil import parser as dateutil_parser

logger = logging.getLogger("jafar.news")


@dataclass
class NewsArticle:
    """A single news article from DuckDuckGo news search."""
    title: str
    body: str       # snippet
    url: str
    source: str
    date: str


def parse_article_age_hours(date_str: str) -> float | None:
    """Parse a DuckDuckGo date string and return age in hours.

    DuckDuckGo returns dates in various formats (ISO, relative, etc.).
    Returns None if unparseable.
    """
    if not date_str:
        return None
    try:
        parsed = dateutil_parser.parse(date_str, fuzzy=True)
        if parsed.tzinfo:
            delta = datetime.now(parsed.tzinfo) - parsed
        else:
            delta = datetime.now() - parsed
        return max(0, delta.total_seconds() / 3600)
    except (ValueError, OverflowError):
        return None


async def fetch_economic_news(
    queries: list[str],
    max_results_per_query: int = 5,
    max_age_hours: float = 48,
    exclude_urls: set[str] | None = None,
) -> list[NewsArticle]:
    """
    Fetch economic news headlines via DuckDuckGo news search.

    Runs DDGS in executor (synchronous library) and deduplicates by URL.
    Filters out stale articles and previously reported URLs.

    Args:
        queries: List of search queries to run.
        max_results_per_query: Max results per query.
        max_age_hours: Maximum article age in hours (older articles filtered out).
        exclude_urls: URLs to exclude (previously reported in past runs).

    Returns:
        Deduplicated, freshness-filtered list of NewsArticle objects.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs not installed. News fetching disabled.")
        return []

    articles: list[NewsArticle] = []
    seen_urls: set[str] = set()
    stale_count = 0

    if exclude_urls:
        seen_urls.update(exclude_urls)

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

                # Check freshness
                age_hours = parse_article_age_hours(r.get("date", ""))
                if age_hours is not None and age_hours > max_age_hours:
                    logger.debug(
                        f"Skipping stale article ({age_hours:.0f}h old): "
                        f"{r.get('title', '')[:60]}"
                    )
                    stale_count += 1
                    continue

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

    excluded_count = len(exclude_urls) if exclude_urls else 0
    logger.info(
        f"Fetched {len(articles)} news articles from {len(queries)} queries "
        f"(filtered: {stale_count} stale, {excluded_count} previously reported)"
    )
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

    parts = [f"## Today's Economic News Headlines ({len(articles)} fresh articles)\n"]
    parts.append("NOTE: These articles have been filtered for freshness and deduplication against previous digests.\n")

    for i, article in enumerate(articles, 1):
        parts.append(f"{i}. **{article.title}**")
        if article.body:
            parts.append(f"   {article.body}")
        parts.append(f"   Source: {article.source} | {article.date}")
        parts.append("")

    return "\n".join(parts)

"""
Test fixtures and sample data for Jafar tests.
"""

from datetime import datetime

from src.scraper import ScrapedTweet


def make_sample_tweet(
    id: int = 1234567890,
    text: str = "This is a sample tweet about $NVDA and silver prices. #stocks",
    username: str = "testuser",
    display_name: str = "Test User",
    created_at: datetime = None,
    likes: int = 100,
    retweets: int = 50,
    replies: int = 10,
    views: int = 1000,
    language: str = "en",
    is_retweet: bool = False,
    hashtags: list[str] = None,
) -> ScrapedTweet:
    """Create a sample ScrapedTweet for testing."""
    return ScrapedTweet(
        id=id,
        text=text,
        username=username,
        display_name=display_name,
        created_at=created_at or datetime.now(),
        likes=likes,
        retweets=retweets,
        replies=replies,
        views=views,
        language=language,
        is_retweet=is_retweet,
        hashtags=hashtags or [],
    )


def make_sample_tweets(count: int = 10, base_engagement: int = 100) -> list[ScrapedTweet]:
    """Create a list of sample tweets with varying engagement."""
    tweets = []
    for i in range(count):
        tweets.append(
            make_sample_tweet(
                id=1234567890 + i,
                text=f"Sample tweet #{i} about markets $AAPL $TSLA #trading",
                username=f"user{i}",
                display_name=f"User {i}",
                likes=base_engagement * (i + 1),
                retweets=base_engagement // 2 * (i + 1),
                replies=base_engagement // 10 * (i + 1),
                hashtags=["trading"],
            )
        )
    return tweets


def make_financial_tweets(count: int = 5) -> list[ScrapedTweet]:
    """Create tweets with strong financial context."""
    texts = [
        "$NVDA just broke resistance at $140. Bullish momentum building. Options flow heavy.",
        "Silver spot price hitting $30. Physical demand overwhelming supply. Shortages reported.",
        "Oil futures up 3% on OPEC news. Energy sector rotation in play. $XLE breaking out.",
        "Fed rate decision incoming. Treasury yields volatile. Bond market pricing in cuts.",
        "Copper supply deficit growing. Infrastructure demand rising. $FCX looking strong.",
    ]
    tweets = []
    for i, text in enumerate(texts[:count]):
        tweets.append(
            make_sample_tweet(
                id=9999990 + i,
                text=text,
                username=f"fintwitter{i}",
                likes=500 * (i + 1),
                retweets=200 * (i + 1),
                hashtags=["stocks", "trading"],
            )
        )
    return tweets

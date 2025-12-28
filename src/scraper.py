"""
Twitter Scraper Module using twscrape.

Handles asynchronous scraping of Twitter/X for broad topic discovery
and specific entity sentiment gathering.

IMPORTANT: Before running this script, you must add Twitter accounts to twscrape.

1. Create a file called `accounts.txt` with your Twitter credentials:
   username:password:email:email_password

2. Add accounts from the file:
   twscrape add_accounts accounts.txt username:password:email:email_password

3. Login all accounts:
   twscrape login_accounts

   If your email provider doesn't support IMAP (e.g., ProtonMail), use:
   twscrape login_accounts --manual

4. Check account status:
   twscrape accounts

This populates the accounts.db SQLite database that twscrape uses for authentication.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator

from twscrape import API, gather
from twscrape.models import Tweet

logger = logging.getLogger("twitter_sentiment.scraper")


@dataclass
class ScrapedTweet:
    """Normalized tweet data structure."""
    id: int
    text: str
    username: str
    display_name: str
    created_at: datetime
    likes: int
    retweets: int
    replies: int
    views: int | None
    language: str | None
    is_retweet: bool
    hashtags: list[str] = field(default_factory=list)

    @classmethod
    def from_twscrape(cls, tweet: Tweet) -> "ScrapedTweet":
        """Create ScrapedTweet from twscrape Tweet object."""
        # Extract hashtags from tweet entities
        hashtags = []
        if tweet.hashtags:
            hashtags = [tag for tag in tweet.hashtags]

        return cls(
            id=tweet.id,
            text=tweet.rawContent,
            username=tweet.user.username if tweet.user else "unknown",
            display_name=tweet.user.displayname if tweet.user else "Unknown",
            created_at=tweet.date,
            likes=tweet.likeCount or 0,
            retweets=tweet.retweetCount or 0,
            replies=tweet.replyCount or 0,
            views=tweet.viewCount,
            language=tweet.lang,
            is_retweet=tweet.rawContent.startswith("RT @") if tweet.rawContent else False,
            hashtags=hashtags,
        )


class TwitterScraper:
    """
    Asynchronous Twitter scraper using twscrape.

    SETUP REQUIRED:
    Before using this class, you must add Twitter accounts via CLI:

    1. Create accounts.txt with format: username:password:email:email_password
    2. Run: twscrape add_accounts accounts.txt username:password:email:email_password
    3. Run: twscrape login_accounts (or with --manual flag for non-IMAP emails)
    4. Verify: twscrape accounts

    The scraper handles account pools automatically for rate limit management.
    """

    def __init__(self, db_path: str = "accounts.db"):
        """
        Initialize the Twitter scraper.

        Args:
            db_path: Path to the twscrape SQLite database containing accounts.
        """
        self.db_path = db_path
        self._api: API | None = None
        logger.info(f"TwitterScraper initialized with database: {db_path}")

    async def _get_api(self) -> API:
        """Get or create the twscrape API instance."""
        if self._api is None:
            self._api = API(self.db_path)
        return self._api

    async def add_account(
        self,
        username: str,
        password: str,
        email: str,
        email_password: str,
    ) -> bool:
        """
        Programmatically add a Twitter account to the pool.

        Note: It's generally recommended to add accounts via CLI instead:
            1. Create accounts.txt: username:password:email:email_password
            2. twscrape add_accounts accounts.txt username:password:email:email_password

        Args:
            username: Twitter username.
            password: Twitter password.
            email: Email associated with the account.
            email_password: Email password for verification.

        Returns:
            True if account was added successfully.
        """
        try:
            api = await self._get_api()
            await api.pool.add_account(username, password, email, email_password)
            logger.info(f"Added account: {username}")
            return True
        except Exception as e:
            logger.error(f"Failed to add account {username}: {e}")
            return False

    async def login_all(self) -> None:
        """
        Login all accounts in the pool.

        This is equivalent to running `twscrape login_accounts` from CLI.
        """
        try:
            api = await self._get_api()
            await api.pool.login_all()
            logger.info("All accounts logged in")
        except Exception as e:
            logger.error(f"Failed to login accounts: {e}")
            raise

    async def get_account_stats(self) -> dict:
        """Get statistics about the account pool."""
        api = await self._get_api()
        stats = await api.pool.stats()
        logger.debug(f"Account pool stats: {stats}")
        return stats

    async def search_tweets(
        self,
        query: str,
        limit: int = 50,
        lang: str = "en",
    ) -> list[ScrapedTweet]:
        """
        Search for tweets matching a query.

        Args:
            query: Search query (hashtag, keyword, or phrase).
            limit: Maximum number of tweets to retrieve.
            lang: Language filter (default: English).

        Returns:
            List of ScrapedTweet objects.
        """
        api = await self._get_api()
        tweets: list[ScrapedTweet] = []

        # Add language filter to query
        search_query = f"{query} lang:{lang}"
        logger.info(f"Searching for: '{search_query}' (limit: {limit})")

        try:
            # Use gather for async collection of tweets
            raw_tweets = await gather(api.search(search_query, limit=limit))

            for tweet in raw_tweets:
                try:
                    scraped = ScrapedTweet.from_twscrape(tweet)
                    tweets.append(scraped)
                except Exception as e:
                    logger.warning(f"Failed to parse tweet {tweet.id}: {e}")
                    continue

            logger.info(f"Retrieved {len(tweets)} tweets for query: {query}")
            return tweets

        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            # Return empty list instead of crashing
            return []

    async def get_broad_tweets(
        self,
        topics: list[str],
        limit_per_topic: int = 50,
    ) -> list[ScrapedTweet]:
        """
        Gather tweets from multiple broad topics for trend discovery.

        This is the "Scout" phase - casting a wide net to discover what's trending.

        Args:
            topics: List of broad topics to search (e.g., ["economy", "markets"]).
            limit_per_topic: Number of tweets per topic.

        Returns:
            Combined list of tweets from all topics.
        """
        logger.info(f"Starting broad search across {len(topics)} topics")
        all_tweets: list[ScrapedTweet] = []

        # Gather tweets from all topics concurrently
        tasks = [
            self.search_tweets(topic, limit=limit_per_topic)
            for topic in topics
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for topic, result in zip(topics, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to scrape topic '{topic}': {result}")
                continue
            all_tweets.extend(result)
            logger.info(f"Topic '{topic}': {len(result)} tweets")

        logger.info(f"Broad search complete: {len(all_tweets)} total tweets")
        return all_tweets

    async def get_specific_sentiment(
        self,
        trends: list[str],
        limit_per_trend: int = 20,
    ) -> dict[str, list[ScrapedTweet]]:
        """
        Deep dive into specific trending entities for detailed sentiment.

        This is the "Deep Dive" phase - targeted scraping for context.

        Args:
            trends: List of trending entity names to investigate.
            limit_per_trend: Number of tweets per trend.

        Returns:
            Dictionary mapping trend names to their tweets.
        """
        logger.info(f"Starting deep dive into {len(trends)} trends")
        trend_tweets: dict[str, list[ScrapedTweet]] = {}

        # Search for each trend concurrently
        tasks = [
            self.search_tweets(trend, limit=limit_per_trend)
            for trend in trends
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for trend, result in zip(trends, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to scrape trend '{trend}': {result}")
                trend_tweets[trend] = []
                continue
            trend_tweets[trend] = result
            logger.info(f"Trend '{trend}': {len(result)} tweets")

        total_tweets = sum(len(t) for t in trend_tweets.values())
        logger.info(f"Deep dive complete: {total_tweets} total tweets")

        return trend_tweets

    async def close(self) -> None:
        """Clean up resources."""
        # twscrape doesn't require explicit cleanup, but keeping for interface
        logger.debug("Scraper resources cleaned up")

"""
Twitter Scraper Module using twscrape.

Handles asynchronous scraping of Twitter/X for broad topic discovery
and specific entity sentiment gathering.

twscrape handles rate limiting, account rotation, request jitter, and
ban detection internally. This module focuses on search orchestration,
DB persistence, and pipeline integration.

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

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from twscrape import API
from twscrape.models import Tweet

logger = logging.getLogger("jafar.scraper")


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
            hashtags = list(tweet.hashtags)

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


async def fetch_trending_topics(api: API) -> list[str]:
    """Fetch current trending topics from Twitter.

    Returns list of trending topic strings (hashtags, terms, etc.)
    """
    try:
        trend_names = []
        async for trend in api.trends("trending"):
            if hasattr(trend, 'name'):
                trend_names.append(trend.name)

        logger.info(f"Fetched {len(trend_names)} trending topics from Twitter")
        return trend_names
    except Exception as e:
        logger.warning(f"Failed to fetch Twitter trends: {e}")
        return []


async def store_tweets(tweets: list[ScrapedTweet], source_query: str, pipeline_run: str) -> int:
    """Store scraped tweets to Postgres for ML training data. Returns count stored."""
    from src.database import get_session
    from src.models import Tweet as TweetModel

    if not tweets:
        return 0

    stored = 0
    session = await get_session()
    async with session:
        async with session.begin():
            for tweet in tweets:
                model = TweetModel(
                    id=tweet.id,
                    username=tweet.username,
                    display_name=tweet.display_name,
                    content=tweet.text,
                    created_at=tweet.created_at.replace(tzinfo=None) if tweet.created_at and tweet.created_at.tzinfo else tweet.created_at,
                    likes=tweet.likes,
                    retweets=tweet.retweets,
                    replies=tweet.replies,
                    views=tweet.views,
                    language=tweet.language,
                    hashtags=tweet.hashtags,
                    source_query=source_query,
                    pipeline_run=pipeline_run,
                )
                await session.merge(model)
                stored += 1
    return stored


def _tweet_model_to_scraped(row) -> ScrapedTweet:
    """Convert a Tweet ORM model back to a ScrapedTweet."""
    return ScrapedTweet(
        id=row.id,
        text=row.content,
        username=row.username,
        display_name=row.display_name or "Unknown",
        created_at=row.created_at,
        likes=row.likes,
        retweets=row.retweets,
        replies=row.replies,
        views=row.views,
        language=row.language,
        hashtags=row.hashtags or [],
        is_retweet=row.content.startswith("RT @") if row.content else False,
    )


async def load_broad_tweets_from_db(
    pipeline_run: str,
    completed_topics: list[str],
) -> list[ScrapedTweet]:
    """Load broad-scrape tweets for a pipeline run from the DB.

    Only returns tweets whose source_query matches a completed topic,
    avoiding contamination from deep-dive tweets on partial resumes.
    """
    from src.database import get_session
    from src.models import Tweet as TweetModel
    from sqlalchemy import select

    if not completed_topics:
        return []

    session = await get_session()
    async with session:
        result = await session.execute(
            select(TweetModel).where(
                TweetModel.pipeline_run == pipeline_run,
                TweetModel.source_query.in_(completed_topics),
            )
        )
        rows = result.scalars().all()

    return [_tweet_model_to_scraped(row) for row in rows]


async def load_trend_tweets_from_db(
    pipeline_run: str,
    trends: list[str],
) -> dict[str, list[ScrapedTweet]]:
    """Load trend-specific tweets for a pipeline run from the DB."""
    from src.database import get_session
    from src.models import Tweet as TweetModel
    from sqlalchemy import select

    if not trends:
        return {}

    session = await get_session()
    async with session:
        result = await session.execute(
            select(TweetModel).where(
                TweetModel.pipeline_run == pipeline_run,
                TweetModel.source_query.in_(trends),
            )
        )
        rows = result.scalars().all()

    out: dict[str, list[ScrapedTweet]] = {t: [] for t in trends}
    for row in rows:
        if row.source_query in out:
            out[row.source_query].append(_tweet_model_to_scraped(row))
    return out


class TwitterScraper:
    """
    Asynchronous Twitter scraper using twscrape.

    twscrape handles rate limiting, account rotation, and request pacing
    internally via its account pool. This class provides search orchestration
    and DB persistence.

    SETUP REQUIRED:
    Before using this class, you must add Twitter accounts via CLI:

    1. Create accounts.txt with format: username:password:email:email_password
    2. Run: twscrape add_accounts accounts.txt username:password:email:email_password
    3. Run: twscrape login_accounts (or with --manual flag for non-IMAP emails)
    4. Verify: twscrape accounts
    """

    def __init__(self, db_path: str = "accounts.db"):
        """
        Initialize the Twitter scraper.

        Args:
            db_path: Path to the twscrape SQLite database containing accounts.
                     Proxies are configured per-account in the database.
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

    async def fix_locks(self) -> None:
        """
        Reset account locks in the database.
        Useful when the scraper was interrupted and accounts remain locked.
        """
        try:
            api = await self._get_api()
            await api.pool.reset_locks()
            logger.info("Account locks reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset account locks: {e}")

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

        twscrape handles rate limiting, account rotation, and pacing internally.

        Args:
            query: Search query (hashtag, keyword, or phrase).
            limit: Maximum number of tweets to retrieve.
            lang: Language filter (default: English).

        Returns:
            List of ScrapedTweet objects.
        """
        api = await self._get_api()
        tweets: list[ScrapedTweet] = []

        search_query = f"{query} lang:{lang}"
        logger.info(f"Searching for: '{search_query}' (limit: {limit})")

        try:
            async for tweet in api.search(search_query, limit=limit):
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
            return tweets

    async def get_broad_tweets(
        self,
        topics: list[str],
        limit_per_topic: int = 50,
        skip_topics: list[str] | None = None,
        pipeline_run: str = "",
        on_topic_complete: Callable[[str, list[ScrapedTweet]], None] | None = None,
    ) -> int:
        """
        Search broad topics and store tweets to DB as each completes.

        Args:
            topics: List of broad topics to search.
            limit_per_topic: Number of tweets per topic.
            skip_topics: Topics to skip (already completed, from checkpoint).
            pipeline_run: Pipeline run ID for DB storage.
            on_topic_complete: Optional callback(topic, tweets) after each topic.

        Returns:
            Total number of tweets stored.
        """
        skip = set(skip_topics or [])
        remaining = [t for t in topics if t not in skip]

        if not remaining:
            logger.info("No topics remaining to scrape.")
            return 0

        logger.info(f"Broad search: {len(remaining)} topics remaining (skipping {len(skip)})")

        total_stored = 0
        for topic in remaining:
            tweets = await self.search_tweets(topic, limit=limit_per_topic)

            stored = await store_tweets(tweets, source_query=topic, pipeline_run=pipeline_run)
            total_stored += stored
            logger.info(f"Topic '{topic}': {len(tweets)} tweets, {stored} stored")

            if on_topic_complete:
                on_topic_complete(topic, tweets)

        logger.info(f"Broad search complete: {total_stored} tweets stored")
        return total_stored

    async def get_specific_sentiment(
        self,
        trends: list[str],
        limit_per_trend: int = 20,
        skip_trends: list[str] | None = None,
        pipeline_run: str = "",
        on_trend_complete: Callable[[str, list[ScrapedTweet]], None] | None = None,
    ) -> int:
        """
        Deep dive into specific trends and store tweets to DB as each completes.

        Args:
            trends: List of trending entity names to investigate.
            limit_per_trend: Number of tweets per trend.
            skip_trends: Trends to skip (already completed, from checkpoint).
            pipeline_run: Pipeline run ID for DB storage.
            on_trend_complete: Optional callback(trend, tweets) after each trend.

        Returns:
            Total number of tweets stored.
        """
        skip = set(skip_trends or [])
        remaining = [t for t in trends if t not in skip]

        if not remaining:
            logger.info("No trends remaining to scrape.")
            return 0

        logger.info(f"Deep dive: {len(remaining)} trends remaining (skipping {len(skip)})")

        total_stored = 0
        for trend in remaining:
            tweets = await self.search_tweets(trend, limit=limit_per_trend)

            stored = await store_tweets(tweets, source_query=trend, pipeline_run=pipeline_run)
            total_stored += stored
            logger.info(f"Trend '{trend}': {len(tweets)} tweets, {stored} stored")

            if on_trend_complete:
                on_trend_complete(trend, tweets)

        logger.info(f"Deep dive complete: {total_stored} tweets stored")
        return total_stored

    async def close(self) -> None:
        """Clean up resources."""
        # twscrape doesn't require explicit cleanup, but keeping for interface
        logger.debug("Scraper resources cleaned up")

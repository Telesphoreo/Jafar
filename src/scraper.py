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
import random
import time
from dataclasses import dataclass, field
from datetime import datetime

from twscrape import API
from twscrape.models import Tweet

from .config import worker_context

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
                     Proxies are configured per-account in the database.
        """
        self.db_path = db_path
        self._api: API | None = None
        # Global rate limiter: only 1 concurrent API call at a time
        self._api_semaphore = asyncio.Semaphore(1)
        # Minimum delay between API calls (seconds)
        self._min_api_delay = 5.0
        self._last_api_call: float = 0.0
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
        timeout: int = 300,
        worker_id: int | str | None = None,
    ) -> list[ScrapedTweet]:
        """
        Search for tweets matching a query.

        Args:
            query: Search query (hashtag, keyword, or phrase).
            limit: Maximum number of tweets to retrieve.
            lang: Language filter (default: English).
            timeout: Maximum time in seconds to wait for results (default: 300s/5min).
            worker_id: Optional ID of the worker initiating the search for logging.

        Returns:
            List of ScrapedTweet objects.
        """
        # Set worker context for logging
        if worker_id is not None:
            worker_context.set(worker_id)

        api = await self._get_api()
        tweets: list[ScrapedTweet] = []
        worker_prefix = f"[Worker {worker_id}] " if worker_id is not None else ""

        # Add language filter to query
        search_query = f"{query} lang:{lang}"
        
        # Check account availability and adjust timeout if rate limited
        wait_time_needed = 0
        try:
            stats = await api.pool.stats()
            active = stats.get("active", 0)
            total = stats.get("total", 0)
            
            if total > 0 and active == 0:
                # All accounts rate limited. Twscrape will wait automatically.
                # We need to ensure our timeout is longer than the wait time.
                # Since we can't easily get the exact reset time from stats here,
                # we'll assume a standard 15-minute window + buffer if we detect this state.
                wait_time_needed = 900  # 15 minutes
                logger.warning(f"{worker_prefix}All {total} accounts are rate-limited. Increasing timeout to allow waiting...")
                timeout = max(timeout, wait_time_needed + 60)
        except Exception as e:
            logger.debug(f"{worker_prefix}Could not check account availability: {e}")

        logger.info(f"{worker_prefix}Searching for: '{search_query}' (limit: {limit}, timeout: {timeout}s)")

        if limit > 100:
            logger.warning(f"{worker_prefix}High tweet limit ({limit}) detected. This may trigger rate limits quickly.")
            logger.warning(f"{worker_prefix}Consider reducing 'broad_tweet_limit' in config.yaml to < 100 for safer scraping.")

        # Use a long safety timeout (20 min) to allow twscrape to wait for rate limits (15 min window)
        # We rely on twscrape's internal logic to handle 429s and waits.
        safety_timeout = 1200

        try:
            raw_tweets = []

            # Acquire global semaphore to ensure only one search runs at a time
            # This prevents multiple workers from hammering the API simultaneously
            logger.debug(f"{worker_prefix}Waiting for API access...")
            async with self._api_semaphore:
                # Enforce minimum delay since last API call
                now = time.time()
                time_since_last = now - self._last_api_call
                if time_since_last < self._min_api_delay:
                    wait_time = self._min_api_delay - time_since_last
                    logger.debug(f"{worker_prefix}Rate limit delay: waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)

                # Add random jitter on top of the minimum delay
                jitter = random.uniform(2, 5)
                logger.debug(f"{worker_prefix}Jitter: {jitter:.1f}s")
                await asyncio.sleep(jitter)

                logger.info(f"{worker_prefix}Starting search...")
                self._last_api_call = time.time()

                # Manual consumption of the generator to allow inter-page delays
                # This makes the scraping "slow and steady" and much harder to detect
                async def fetch_with_delays():
                    count = 0
                    async for tweet in api.search(search_query, limit=limit):
                        raw_tweets.append(tweet)
                        count += 1

                        # Every ~20 tweets (approx one page request), take a human-like breath
                        if count % 20 == 0:
                            delay = random.uniform(10, 15)
                            logger.debug(f"{worker_prefix}Search '{query}': {count} tweets retrieved. Humanizing delay {delay:.1f}s...")
                            await asyncio.sleep(delay)
                            # Update last API call time for inter-page requests
                            self._last_api_call = time.time()
                    return raw_tweets

                # Still use wait_for to prevent total hangs, but with the manual loop inside
                await asyncio.wait_for(fetch_with_delays(), timeout=safety_timeout)

                # Update last API call time after completion
                self._last_api_call = time.time()

            for tweet in raw_tweets:
                try:
                    scraped = ScrapedTweet.from_twscrape(tweet)
                    tweets.append(scraped)
                except Exception as e:
                    logger.warning(f"{worker_prefix}Failed to parse tweet {tweet.id}: {e}")
                    continue

            logger.info(f"{worker_prefix}Retrieved {len(tweets)} tweets for query: {query}")
            return tweets

        except asyncio.TimeoutError:
            logger.error(f"{worker_prefix}Safety timeout reached for '{query}' after {safety_timeout}s")
            logger.error(f"{worker_prefix}This suggests a genuine network hang or extremely long rate limit.")
            return tweets
        except Exception as e:
            logger.error(f"{worker_prefix}Error searching for '{query}': {e}")
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
            self.search_tweets(topic, limit=limit_per_topic, worker_id=i)
            for i, topic in enumerate(topics)
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

    async def get_broad_tweets_incremental(
        self,
        topics: list[str],
        limit_per_topic: int = 50,
        on_topic_complete: callable = None,
        skip_topics: list[str] = None,
        timeout: int = 300,
    ) -> list[ScrapedTweet]:
        """
        Gather tweets incrementally using concurrent workers for load balancing.

        Args:
            topics: List of topics to search.
            limit_per_topic: Number of tweets per topic.
            on_topic_complete: Callback(topic, tweets) called after each topic.
            skip_topics: Topics to skip (already completed).
            timeout: Maximum time in seconds to wait per topic (default: 300s/5min).

        Returns:
            Combined list of tweets from all topics.
        """
        skip_topics = skip_topics or []
        all_tweets: list[ScrapedTweet] = []
        remaining = [t for t in topics if t not in skip_topics]

        if not remaining:
            logger.info("No topics remaining to scrape.")
            return []

        # Determine concurrency based on active accounts
        stats = await self.get_account_stats()
        active_count = stats.get("active", 1)
        concurrency = min(active_count, 5)
        logger.info(f"Incremental scrape: {len(remaining)} topics remaining. Using {concurrency} concurrent workers.")

        queue = asyncio.Queue()
        for topic in remaining:
            queue.put_nowait(topic)

        async def worker(worker_id: int, startup_delay: float):
            # Stagger worker startup to prevent thundering herd
            if startup_delay > 0:
                logger.debug(f"[Worker {worker_id}] Startup delay: {startup_delay:.1f}s")
                await asyncio.sleep(startup_delay)

            while not queue.empty():
                try:
                    topic = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                logger.info(f"[Worker {worker_id}] Scraping topic: {topic}")

                try:
                    tweets = await self.search_tweets(topic, limit=limit_per_topic, timeout=timeout, worker_id=worker_id)
                    all_tweets.extend(tweets)

                    if on_topic_complete:
                        on_topic_complete(topic, tweets)

                    logger.info(f"[Worker {worker_id}] Topic '{topic}': {len(tweets)} tweets")

                except Exception as e:
                    logger.error(f"[Worker {worker_id}] Failed to scrape topic '{topic}': {e}")
                    if on_topic_complete:
                        on_topic_complete(topic, [])
                finally:
                    queue.task_done()

        # Stagger worker startups by 2 seconds each to reduce initial contention
        workers = [asyncio.create_task(worker(i, i * 2.0)) for i in range(concurrency)]
        await asyncio.gather(*workers)

        logger.info(f"Incremental scrape complete: {len(all_tweets)} tweets from {len(remaining)} topics")
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
            self.search_tweets(trend, limit=limit_per_trend, worker_id=i)
            for i, trend in enumerate(trends)
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

    async def get_specific_sentiment_incremental(
        self,
        trends: list[str],
        limit_per_trend: int = 20,
        on_trend_complete: callable = None,
        skip_trends: list[str] = None,
        timeout: int = 300,
    ) -> dict[str, list[ScrapedTweet]]:
        """
        Deep dive incrementally using concurrent workers for load balancing.

        Args:
            trends: List of trends to search.
            limit_per_trend: Number of tweets per trend.
            on_trend_complete: Callback(trend, tweets) called after each trend.
            skip_trends: Trends to skip (already completed).
            timeout: Maximum time in seconds to wait per trend (default: 300s/5min).

        Returns:
            Dictionary mapping trend names to their tweets.
        """
        skip_trends = skip_trends or []
        trend_tweets: dict[str, list[ScrapedTweet]] = {}
        remaining = [t for t in trends if t not in skip_trends]

        if not remaining:
            logger.info("No trends remaining to scrape.")
            return {}

        # Determine concurrency based on active accounts
        await self.get_account_stats()
        # FORCE SERIAL EXECUTION to prevent 429s until stability is proven
        concurrency = 1
        # concurrency = min(active_count, 5)
        concurrency = max(1, concurrency)

        logger.info(f"Incremental deep dive: {len(remaining)} trends remaining. Using {concurrency} concurrent workers.")

        queue = asyncio.Queue()
        for trend in remaining:
            queue.put_nowait(trend)

        async def worker(worker_id: int, startup_delay: float):
            # Stagger worker startup to prevent thundering herd
            if startup_delay > 0:
                logger.debug(f"[Worker {worker_id}] Startup delay: {startup_delay:.1f}s")
                await asyncio.sleep(startup_delay)

            while not queue.empty():
                try:
                    trend = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                logger.info(f"[Worker {worker_id}] Scraping trend: {trend}")

                try:
                    tweets = await self.search_tweets(trend, limit=limit_per_trend, timeout=timeout, worker_id=worker_id)
                    trend_tweets[trend] = tweets

                    if on_trend_complete:
                        on_trend_complete(trend, tweets)

                except Exception as e:
                    logger.error(f"[Worker {worker_id}] Failed to scrape trend '{trend}': {e}")
                    trend_tweets[trend] = []
                    if on_trend_complete:
                        on_trend_complete(trend, [])
                finally:
                    queue.task_done()

        # Stagger worker startups by 2 seconds each to reduce initial contention
        workers = [asyncio.create_task(worker(i, i * 2.0)) for i in range(concurrency)]
        await asyncio.gather(*workers)

        total = sum(len(t) for t in trend_tweets.values())
        logger.info(f"Incremental deep dive complete: {total} tweets from {len(remaining)} trends")
        return trend_tweets

    async def close(self) -> None:
        """Clean up resources."""
        # twscrape doesn't require explicit cleanup, but keeping for interface
        logger.debug("Scraper resources cleaned up")

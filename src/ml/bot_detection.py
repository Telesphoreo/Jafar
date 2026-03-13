"""Bot detection scoring using Isolation Forest on tweet behavioral signals.

Analyzes stored tweets per account to surface bot-like posting patterns.
Uses unsupervised anomaly detection (no labeled data required).
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select, func

from src.database import get_session
from src.models import Tweet

logger = logging.getLogger("jafar.ml.bot_detection")


@dataclass
class TweetData:
    """Lightweight container for tweet fields needed by the feature extractor."""

    content: str
    created_at: Optional[datetime]
    likes: int
    retweets: int
    replies: int
    views: Optional[int]
    hashtags: Optional[list[str]]
    source_query: Optional[str]


# --------------------------------------------------------------------------- #
# Feature Extraction
# --------------------------------------------------------------------------- #

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")


class BotFeatureExtractor:
    """Extracts per-account numeric features from a list of tweets."""

    # Canonical ordering — must stay stable between train and score.
    FEATURE_NAMES: list[str] = [
        # Timing
        "avg_tweet_interval_seconds",
        "std_tweet_interval_seconds",
        "coefficient_of_variation",
        "night_ratio",
        # Content
        "avg_content_length",
        "duplicate_content_ratio",
        "url_ratio",
        "hashtag_ratio",
        "mention_ratio",
        "avg_hashtags_per_tweet",
        # Engagement
        "avg_likes",
        "avg_retweets",
        "engagement_ratio",
        "zero_engagement_ratio",
        # Account behavior
        "unique_sources_ratio",
        "tweet_count",
    ]

    def extract_features(self, tweets: list[TweetData]) -> dict[str, float]:
        """Return a dict of feature_name -> float for one account's tweets."""
        n = len(tweets)
        if n == 0:
            return {name: 0.0 for name in self.FEATURE_NAMES}

        features: dict[str, float] = {}

        # --- Timing -------------------------------------------------------- #
        timestamps = sorted(
            [t.created_at for t in tweets if t.created_at is not None]
        )
        if len(timestamps) >= 2:
            intervals = [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]
            avg_interval = float(np.mean(intervals))
            std_interval = float(np.std(intervals))
            features["avg_tweet_interval_seconds"] = avg_interval
            features["std_tweet_interval_seconds"] = std_interval
            features["coefficient_of_variation"] = (
                std_interval / avg_interval if avg_interval > 0 else 0.0
            )
        else:
            features["avg_tweet_interval_seconds"] = 0.0
            features["std_tweet_interval_seconds"] = 0.0
            features["coefficient_of_variation"] = 0.0

        night_count = sum(
            1
            for t in tweets
            if t.created_at is not None and 1 <= t.created_at.hour < 5
        )
        features["night_ratio"] = night_count / n

        # --- Content ------------------------------------------------------- #
        contents = [t.content for t in tweets]
        features["avg_content_length"] = float(np.mean([len(c) for c in contents]))

        # Near-duplicate detection: normalise whitespace then compare
        normalised = [re.sub(r"\s+", " ", c.strip().lower()) for c in contents]
        counts = Counter(normalised)
        duplicate_count = sum(v - 1 for v in counts.values() if v > 1)
        features["duplicate_content_ratio"] = duplicate_count / n

        features["url_ratio"] = sum(1 for c in contents if _URL_RE.search(c)) / n
        features["hashtag_ratio"] = sum(
            1 for c in contents if _HASHTAG_RE.search(c)
        ) / n
        features["mention_ratio"] = sum(
            1 for c in contents if _MENTION_RE.search(c)
        ) / n

        total_hashtags = 0
        for t in tweets:
            if t.hashtags:
                total_hashtags += len(t.hashtags)
            else:
                # Fall back to regex count on the text
                total_hashtags += len(_HASHTAG_RE.findall(t.content))
        features["avg_hashtags_per_tweet"] = total_hashtags / n

        # --- Engagement ---------------------------------------------------- #
        likes = [t.likes for t in tweets]
        retweets = [t.retweets for t in tweets]
        replies = [t.replies for t in tweets]
        views = [t.views for t in tweets if t.views is not None]

        features["avg_likes"] = float(np.mean(likes))
        features["avg_retweets"] = float(np.mean(retweets))

        total_engagement = sum(likes) + sum(retweets) + sum(replies)
        total_views = sum(views) if views else 0
        features["engagement_ratio"] = (
            total_engagement / total_views if total_views > 0 else 0.0
        )

        zero_eng = sum(
            1 for t in tweets if t.likes == 0 and t.retweets == 0
        )
        features["zero_engagement_ratio"] = zero_eng / n

        # --- Account behaviour --------------------------------------------- #
        unique_sources = set(
            t.source_query for t in tweets if t.source_query is not None
        )
        features["unique_sources_ratio"] = len(unique_sources) / n
        features["tweet_count"] = float(n)

        return features


# --------------------------------------------------------------------------- #
# Scorer
# --------------------------------------------------------------------------- #


class BotScorer:
    """Scores accounts for bot-like behaviour using an Isolation Forest model."""

    def __init__(self) -> None:
        self.model: Optional[IsolationForest] = None
        self.scaler = StandardScaler()
        self.feature_extractor = BotFeatureExtractor()
        self.feature_names: list[str] = list(
            BotFeatureExtractor.FEATURE_NAMES
        )

    # ----- helpers --------------------------------------------------------- #

    async def _fetch_tweets_by_username(self) -> dict[str, list[TweetData]]:
        """Load all tweets from the database, grouped by username."""
        session = await get_session()
        async with session:
            result = await session.execute(select(Tweet))
            rows = result.scalars().all()

        accounts: dict[str, list[TweetData]] = {}
        for row in rows:
            td = TweetData(
                content=row.content,
                created_at=row.created_at,
                likes=row.likes,
                retweets=row.retweets,
                replies=row.replies,
                views=row.views,
                hashtags=row.hashtags,
                source_query=row.source_query,
            )
            accounts.setdefault(row.username, []).append(td)
        return accounts

    async def _fetch_account_tweets(self, username: str) -> list[TweetData]:
        """Load tweets for a single username."""
        session = await get_session()
        async with session:
            result = await session.execute(
                select(Tweet).where(Tweet.username == username)
            )
            rows = result.scalars().all()

        return [
            TweetData(
                content=row.content,
                created_at=row.created_at,
                likes=row.likes,
                retweets=row.retweets,
                replies=row.replies,
                views=row.views,
                hashtags=row.hashtags,
                source_query=row.source_query,
            )
            for row in rows
        ]

    def _raw_score_to_bot_score(self, decision_score: float) -> float:
        """Convert IsolationForest decision_function output to a 0-1 bot score.

        IsolationForest.decision_function returns negative values for anomalies
        and positive values for inliers.  We invert and clip to [0, 1] so that
        higher values mean *more* bot-like.
        """
        # decision_function: large positive = very normal, large negative = anomaly
        # We negate so anomalies become positive, then sigmoid-ish map to [0, 1].
        raw = -decision_score
        # Empirically, decision scores are roughly in [-0.5, 0.5].
        # A simple linear rescale with clipping works well enough.
        score = (raw + 0.5) / 1.0  # maps -0.5->0, 0.5->1
        return float(np.clip(score, 0.0, 1.0))

    # ----- public API ------------------------------------------------------ #

    async def train(self, min_tweets_per_account: int = 5) -> dict:
        """Train the Isolation Forest on all stored tweets.

        Returns a dict of training statistics.
        """
        accounts = await self._fetch_tweets_by_username()

        # Filter to accounts with enough tweets
        eligible = {
            user: tweets
            for user, tweets in accounts.items()
            if len(tweets) >= min_tweets_per_account
        }

        if len(eligible) < 2:
            msg = (
                f"Need at least 2 accounts with >= {min_tweets_per_account} "
                f"tweets to train. Found {len(eligible)}."
            )
            logger.warning(msg)
            return {"trained": False, "reason": msg, "accounts": len(eligible)}

        # Build feature matrix
        usernames = list(eligible.keys())
        feature_matrix = np.array(
            [
                [
                    self.feature_extractor.extract_features(eligible[u])[f]
                    for f in self.feature_names
                ]
                for u in usernames
            ]
        )

        # Scale
        X_scaled = self.scaler.fit_transform(feature_matrix)

        # Train
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=200,
        )
        self.model.fit(X_scaled)

        # Summary stats
        scores = self.model.decision_function(X_scaled)
        anomaly_labels = self.model.predict(X_scaled)
        n_anomalies = int((anomaly_labels == -1).sum())

        logger.info(
            "Bot detection model trained on %d accounts (%d flagged as anomalies)",
            len(usernames),
            n_anomalies,
        )

        return {
            "trained": True,
            "accounts_total": len(accounts),
            "accounts_eligible": len(eligible),
            "accounts_anomalous": n_anomalies,
            "min_tweets_per_account": min_tweets_per_account,
        }

    async def score_account(self, username: str) -> Optional[dict]:
        """Score a single account.

        Returns None if the model is not trained or there are too few tweets.
        """
        if self.model is None:
            logger.warning("Model not trained — call train() first.")
            return None

        tweets = await self._fetch_account_tweets(username)
        if not tweets:
            return None

        features = self.feature_extractor.extract_features(tweets)
        feature_vector = np.array(
            [[features[f] for f in self.feature_names]]
        )
        X_scaled = self.scaler.transform(feature_vector)

        decision = float(self.model.decision_function(X_scaled)[0])
        prediction = int(self.model.predict(X_scaled)[0])

        return {
            "username": username,
            "bot_score": self._raw_score_to_bot_score(decision),
            "anomaly": prediction == -1,
            "features": features,
        }

    async def score_all_accounts(
        self, min_tweets: int = 5
    ) -> list[dict]:
        """Score every account with enough tweets.

        Returns a list sorted by bot_score descending.
        """
        if self.model is None:
            logger.warning("Model not trained — call train() first.")
            return []

        accounts = await self._fetch_tweets_by_username()
        results: list[dict] = []

        for username, tweets in accounts.items():
            if len(tweets) < min_tweets:
                continue

            features = self.feature_extractor.extract_features(tweets)
            feature_vector = np.array(
                [[features[f] for f in self.feature_names]]
            )
            X_scaled = self.scaler.transform(feature_vector)

            decision = float(self.model.decision_function(X_scaled)[0])
            prediction = int(self.model.predict(X_scaled)[0])

            results.append(
                {
                    "username": username,
                    "bot_score": self._raw_score_to_bot_score(decision),
                    "anomaly": prediction == -1,
                    "features": features,
                }
            )

        results.sort(key=lambda r: r["bot_score"], reverse=True)
        return results

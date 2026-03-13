"""
ML account signal scoring module.

Identifies high-signal accounts from stored tweet data using feature engineering
and unsupervised learning. High-signal accounts provide original, insightful
content about economic trends — not news outlets regurgitating headlines.

A retail manager tweeting about shelf shortages is more valuable than Reuters.
"""

import logging
import re
from collections import defaultdict
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import func, select

from src.database import get_session
from src.models import Tweet

logger = logging.getLogger("jafar.ml.account_scoring")

# Cluster labels assigned after KMeans fitting, ordered by median signal score
CLUSTER_LABELS = ["low_activity", "casual", "news_aggregator", "high_signal"]

# Weights for composite signal score
SCORE_WEIGHTS = {
    "engagement_per_view": 0.25,
    "unique_content_ratio": 0.20,
    "trend_coverage_ratio": 0.15,
    "recurrence_ratio": 0.15,
    "avg_replies": 0.10,
    "engagement_consistency": 0.10,
    "url_ratio": -0.15,  # penalty for link aggregators
}

# Regex for URLs in tweet text
_URL_RE = re.compile(r"https?://\S+")
# Regex for question marks (simple question detection)
_QUESTION_RE = re.compile(r"\?")


class SignalFeatureExtractor:
    """Extracts signal features from an account's tweet history.

    Features span four categories:
    - Content originality (length, URL ratio, uniqueness, questions)
    - Engagement quality (likes, retweets, replies, consistency)
    - Discovery signal (trend coverage, pipeline run recurrence)
    - Timing signal (active hours spread, tweet frequency)
    """

    def extract_features(
        self,
        tweets: list,
        total_trends: int = 1,
        total_runs: int = 1,
    ) -> dict[str, float]:
        """Extract signal features from a list of Tweet model instances.

        Args:
            tweets: List of Tweet ORM objects for a single account.
            total_trends: Total number of distinct source_queries in the dataset.
            total_runs: Total number of distinct pipeline_runs in the dataset.

        Returns:
            Dictionary of feature name -> float value.
        """
        if not tweets:
            return self._empty_features()

        n = len(tweets)

        # --- Content originality ---
        content_lengths = [len(t.content) for t in tweets]
        avg_content_length = sum(content_lengths) / n

        url_count = sum(1 for t in tweets if _URL_RE.search(t.content))
        url_ratio = url_count / n

        # Unique content: normalize whitespace, lowercase, then count unique
        normalized = [re.sub(r"\s+", " ", t.content.strip().lower()) for t in tweets]
        unique_content_ratio = len(set(normalized)) / n

        question_count = sum(1 for t in tweets if _QUESTION_RE.search(t.content))
        question_ratio = question_count / n

        # --- Engagement quality ---
        likes = [t.likes for t in tweets]
        retweets = [t.retweets for t in tweets]
        replies = [t.replies for t in tweets]
        views = [t.views for t in tweets if t.views is not None]

        avg_likes = sum(likes) / n
        avg_retweets = sum(retweets) / n
        avg_replies = sum(replies) / n

        total_engagement = sum(likes) + sum(retweets) + sum(replies)
        total_views = sum(views) if views else 0
        engagement_per_view = (
            total_engagement / total_views if total_views > 0 else 0.0
        )

        reply_to_like_ratio = (
            sum(replies) / sum(likes) if sum(likes) > 0 else 0.0
        )

        # Engagement consistency: lower coefficient of variation = more consistent
        per_tweet_engagement = [t.likes + t.retweets + t.replies for t in tweets]
        mean_eng = np.mean(per_tweet_engagement) if per_tweet_engagement else 0.0
        std_eng = np.std(per_tweet_engagement) if per_tweet_engagement else 0.0
        # Invert so higher = more consistent (1 - CV, clamped to [0, 1])
        engagement_consistency = (
            max(0.0, 1.0 - (std_eng / mean_eng)) if mean_eng > 0 else 0.0
        )

        # --- Discovery signal ---
        source_queries = {t.source_query for t in tweets if t.source_query}
        trend_coverage = len(source_queries)
        trend_coverage_ratio = trend_coverage / max(total_trends, 1)

        pipeline_runs = {t.pipeline_run for t in tweets if t.pipeline_run}
        pipeline_run_appearances = len(pipeline_runs)
        recurrence_ratio = pipeline_run_appearances / max(total_runs, 1)

        # --- Timing signal ---
        hours = [t.created_at.hour for t in tweets if t.created_at]
        active_hours_spread = len(set(hours)) if hours else 0

        # Tweet frequency: tweets per day across the observed window
        timestamps = [t.created_at for t in tweets if t.created_at]
        if len(timestamps) >= 2:
            sorted_ts = sorted(timestamps)
            span_days = (sorted_ts[-1] - sorted_ts[0]).total_seconds() / 86400
            tweet_frequency = n / max(span_days, 1.0)
        else:
            tweet_frequency = float(n)

        return {
            # Content originality
            "avg_content_length": avg_content_length,
            "url_ratio": url_ratio,
            "unique_content_ratio": unique_content_ratio,
            "question_ratio": question_ratio,
            # Engagement quality
            "avg_likes": avg_likes,
            "avg_retweets": avg_retweets,
            "avg_replies": avg_replies,
            "reply_to_like_ratio": reply_to_like_ratio,
            "engagement_per_view": engagement_per_view,
            "engagement_consistency": engagement_consistency,
            # Discovery signal
            "trend_coverage": float(trend_coverage),
            "trend_coverage_ratio": trend_coverage_ratio,
            "pipeline_run_appearances": float(pipeline_run_appearances),
            "recurrence_ratio": recurrence_ratio,
            # Timing signal
            "active_hours_spread": float(active_hours_spread),
            "tweet_frequency": tweet_frequency,
        }

    def _empty_features(self) -> dict[str, float]:
        """Return a zeroed-out feature dict."""
        return {
            "avg_content_length": 0.0,
            "url_ratio": 0.0,
            "unique_content_ratio": 0.0,
            "question_ratio": 0.0,
            "avg_likes": 0.0,
            "avg_retweets": 0.0,
            "avg_replies": 0.0,
            "reply_to_like_ratio": 0.0,
            "engagement_per_view": 0.0,
            "engagement_consistency": 0.0,
            "trend_coverage": 0.0,
            "trend_coverage_ratio": 0.0,
            "pipeline_run_appearances": 0.0,
            "recurrence_ratio": 0.0,
            "active_hours_spread": 0.0,
            "tweet_frequency": 0.0,
        }


class AccountScorer:
    """Scores accounts by signal quality using unsupervised learning.

    Uses a weighted composite score to rank accounts, with optional KMeans
    clustering to label account types (high_signal, news_aggregator, casual,
    low_activity).
    """

    def __init__(self):
        self.model: Optional[KMeans] = None
        self.scaler = StandardScaler()
        self.feature_extractor = SignalFeatureExtractor()

    async def analyze(self, min_tweets_per_account: int = 3) -> list[dict]:
        """Analyze all stored accounts and rank by signal quality.

        Args:
            min_tweets_per_account: Skip accounts with fewer tweets than this.

        Returns:
            List of dicts sorted by signal_score descending:
            [{"username": str, "signal_score": float (0-1), "cluster": int,
              "cluster_label": str, "features": dict, "tweet_count": int}, ...]
        """
        session = await get_session()
        async with session:
            # Get global counts for normalization
            total_trends_result = await session.execute(
                select(func.count(func.distinct(Tweet.source_query)))
            )
            total_trends = total_trends_result.scalar() or 1

            total_runs_result = await session.execute(
                select(func.count(func.distinct(Tweet.pipeline_run)))
            )
            total_runs = total_runs_result.scalar() or 1

            # Get all tweets grouped by username
            result = await session.execute(
                select(Tweet).order_by(Tweet.username)
            )
            all_tweets = result.scalars().all()

        # Group tweets by username
        by_user: dict[str, list] = defaultdict(list)
        for tweet in all_tweets:
            by_user[tweet.username].append(tweet)

        # Filter accounts below minimum tweet threshold
        by_user = {
            u: tweets
            for u, tweets in by_user.items()
            if len(tweets) >= min_tweets_per_account
        }

        if not by_user:
            logger.info("No accounts meet the minimum tweet threshold.")
            return []

        # Extract features for each account
        usernames = list(by_user.keys())
        all_features = []
        for username in usernames:
            features = self.feature_extractor.extract_features(
                by_user[username],
                total_trends=total_trends,
                total_runs=total_runs,
            )
            all_features.append(features)

        # Compute weighted composite scores
        raw_scores = []
        for features in all_features:
            score = 0.0
            for feature_name, weight in SCORE_WEIGHTS.items():
                score += weight * features.get(feature_name, 0.0)
            raw_scores.append(score)

        # Normalize to 0-1 range using min-max scaling
        raw_scores = np.array(raw_scores)
        score_min = raw_scores.min()
        score_max = raw_scores.max()
        if score_max > score_min:
            normalized_scores = (raw_scores - score_min) / (score_max - score_min)
        else:
            normalized_scores = np.zeros_like(raw_scores)

        # Optional KMeans clustering (k=4) for account type labeling
        feature_names = list(all_features[0].keys())
        feature_matrix = np.array(
            [[f[name] for name in feature_names] for f in all_features]
        )

        cluster_labels_arr = np.zeros(len(usernames), dtype=int)
        cluster_label_names = ["unclustered"] * len(usernames)

        if len(usernames) >= 4:
            try:
                scaled_features = self.scaler.fit_transform(feature_matrix)
                self.model = KMeans(n_clusters=4, random_state=42, n_init=10)
                clusters = self.model.fit_predict(scaled_features)

                # Order clusters by median signal score and assign labels
                cluster_median_scores = {}
                for c in range(4):
                    mask = clusters == c
                    if mask.any():
                        cluster_median_scores[c] = float(
                            np.median(normalized_scores[mask])
                        )
                    else:
                        cluster_median_scores[c] = 0.0

                sorted_clusters = sorted(
                    cluster_median_scores, key=cluster_median_scores.get
                )
                label_map = {
                    c: CLUSTER_LABELS[i] for i, c in enumerate(sorted_clusters)
                }

                cluster_labels_arr = clusters
                cluster_label_names = [label_map[c] for c in clusters]
            except Exception as e:
                logger.warning(f"KMeans clustering failed: {e}")

        # Build results
        results = []
        for i, username in enumerate(usernames):
            results.append(
                {
                    "username": username,
                    "signal_score": float(normalized_scores[i]),
                    "cluster": int(cluster_labels_arr[i]),
                    "cluster_label": cluster_label_names[i],
                    "features": all_features[i],
                    "tweet_count": len(by_user[username]),
                }
            )

        # Sort by signal_score descending
        results.sort(key=lambda r: r["signal_score"], reverse=True)

        logger.info(
            f"Analyzed {len(results)} accounts. "
            f"Top signal: {results[0]['username']} ({results[0]['signal_score']:.3f})"
            if results
            else "No accounts analyzed."
        )

        return results

    async def get_top_accounts(
        self, n: int = 20, min_tweets: int = 3
    ) -> list[dict]:
        """Get the top N highest-signal accounts.

        Args:
            n: Number of top accounts to return.
            min_tweets: Minimum tweets required per account.

        Returns:
            Top N accounts sorted by signal_score descending.
        """
        results = await self.analyze(min_tweets_per_account=min_tweets)
        return results[:n]

    async def get_account_profile(self, username: str) -> Optional[dict]:
        """Get detailed signal profile for a single account.

        Args:
            username: Twitter username to profile.

        Returns:
            Dict with signal score, features, and cluster info, or None if
            the account has no stored tweets.
        """
        session = await get_session()
        async with session:
            # Get global counts
            total_trends_result = await session.execute(
                select(func.count(func.distinct(Tweet.source_query)))
            )
            total_trends = total_trends_result.scalar() or 1

            total_runs_result = await session.execute(
                select(func.count(func.distinct(Tweet.pipeline_run)))
            )
            total_runs = total_runs_result.scalar() or 1

            # Get tweets for this account
            result = await session.execute(
                select(Tweet).where(Tweet.username == username)
            )
            tweets = result.scalars().all()

        if not tweets:
            return None

        features = self.feature_extractor.extract_features(
            tweets,
            total_trends=total_trends,
            total_runs=total_runs,
        )

        # Compute raw score using weights
        raw_score = 0.0
        for feature_name, weight in SCORE_WEIGHTS.items():
            raw_score += weight * features.get(feature_name, 0.0)

        return {
            "username": username,
            "signal_score_raw": raw_score,
            "features": features,
            "tweet_count": len(tweets),
        }

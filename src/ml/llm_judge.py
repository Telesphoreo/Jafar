"""LLM-as-a-Judge module for bot detection evaluation.

Uses Gemini to evaluate whether Twitter accounts are bots, creating
pseudo-labeled ground truth for evaluating and improving the ML bot
detection model.
"""

import json
import logging
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger("jafar.ml.llm_judge")

BOT_JUDGE_SYSTEM_PROMPT = """\
You are a signal quality analyst for an economic intelligence system. \
Your job is NOT to detect bots vs humans — it's to determine whether \
an account produces valuable signal or worthless garbage.

THE QUESTION: "If this account's tweets showed up in our economic trend analysis, would they add signal or noise?"

GARBAGE — discard these accounts entirely:
- Crypto pump/spam accounts pushing coins with zero substance
- Follower farm accounts that repost templates with no original content
- Accounts with hundreds of tweets and zero engagement (nobody reads them)
- Automated spam that adds no informational value whatsoever
- Scam accounts, fake giveaways, engagement bait factories

SIGNAL — keep these accounts, even if automated:
- News aggregators (@unusual_whales, @DeItaone) — they surface real market data
- Government/institutional accounts (@WhiteHouse, @FedReserve) — primary sources
- Journalists, analysts, commentators — even controversial ones
- Grifters with actual takes — hot takes are signal, even bad ones
- Industry insiders posting about their sector (retail workers, truck drivers, etc.)
- Prediction markets, data feeds — they reflect real sentiment

KEY PRINCIPLE: We WANT grifters, provocateurs, and opinionated accounts. We want to see the crazy. What we DON'T want is soulless spam that nobody reads. An account being automated does NOT make it garbage — @DeItaone is automated and it's one of the most valuable feeds on Twitter.

You must respond with valid JSON only, no other text."""

RESPONSE_FORMAT = """Respond with this exact JSON structure:
{
    "garbage_probability": <float 0-1, where 1.0 = pure spam/garbage>,
    "confidence": <float 0-1>,
    "classification": "<garbage|likely_garbage|uncertain|likely_signal|signal>",
    "reasoning": "<2-3 sentence explanation of WHY this account is or isn't valuable>",
    "signals": ["<signal 1>", "<signal 2>", ...]
}"""

# Threshold for classifying as "garbage" in binary comparisons
GARBAGE_THRESHOLD = 0.6


class BotJudge:
    """Uses Gemini LLM to evaluate accounts for bot-like behavior.

    Creates pseudo-labels that serve as ground truth for ML model evaluation.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """Initialize the Gemini client.

        Args:
            api_key: Google API key for Gemini.
            model: Gemini model to use.
        """
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def _format_account_for_judgment(
        self,
        username: str,
        tweets: list[dict[str, Any]],
        ml_features: dict[str, float] | None = None,
    ) -> str:
        """Format account data into a clear prompt for LLM judgment.

        Args:
            username: The Twitter username.
            tweets: List of tweet dicts with content, timestamps, engagement.
            ml_features: Optional ML-extracted features for context.

        Returns:
            Formatted string for the LLM prompt.
        """
        lines = [
            f"Account: @{username}",
            f"Total tweets analyzed: {len(tweets)}",
            "",
            "Recent tweets (showing up to 20):",
        ]

        for tweet in tweets[:20]:
            timestamp = tweet.get("created_at", "unknown")
            content = tweet.get("content", "")
            likes = tweet.get("likes", 0)
            retweets = tweet.get("retweets", 0)
            replies = tweet.get("replies", 0)
            views = tweet.get("views", "N/A")

            lines.append("---")
            lines.append(f"[{timestamp}] {content}")
            lines.append(
                f"Likes: {likes} | Retweets: {retweets} | "
                f"Replies: {replies} | Views: {views}"
            )

        lines.append("---")

        if ml_features:
            lines.append("")
            lines.append("ML Feature Analysis:")
            feature_labels = {
                "avg_tweet_interval_seconds": "Posting interval avg",
                "std_tweet_interval_seconds": "Posting interval std",
                "coefficient_of_variation": "Interval coefficient of variation",
                "night_ratio": "Night posting ratio (1-5am)",
                "avg_content_length": "Avg content length",
                "duplicate_content_ratio": "Duplicate content ratio",
                "url_ratio": "URL ratio",
                "hashtag_ratio": "Hashtag ratio",
                "mention_ratio": "Mention ratio",
                "avg_hashtags_per_tweet": "Avg hashtags per tweet",
                "avg_likes": "Avg likes",
                "avg_retweets": "Avg retweets",
                "engagement_ratio": "Engagement ratio",
                "zero_engagement_ratio": "Zero engagement ratio",
                "unique_sources_ratio": "Unique sources ratio",
                "tweet_count": "Tweet count",
            }
            for key, label in feature_labels.items():
                if key in ml_features:
                    value = ml_features[key]
                    if isinstance(value, float):
                        lines.append(f"- {label}: {value:.4f}")
                    else:
                        lines.append(f"- {label}: {value}")

        lines.append("")
        lines.append(RESPONSE_FORMAT)

        return "\n".join(lines)

    async def judge_account(
        self,
        username: str,
        tweets: list[dict[str, Any]],
        ml_features: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Judge a single account using LLM analysis.

        Args:
            username: The Twitter username.
            tweets: List of tweet data (content, timestamps, engagement metrics).
            ml_features: Optional ML-extracted features for context.

        Returns:
            Dict with garbage_probability, confidence, reasoning, signals,
            and classification.
        """
        user_message = self._format_account_for_judgment(
            username, tweets, ml_features
        )

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=BOT_JUDGE_SYSTEM_PROMPT,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        try:
            result = json.loads(response.text)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(
                "Failed to parse LLM response for @%s: %s (response: %s)",
                username,
                e,
                response.text,
            )
            result = {
                "garbage_probability": 0.5,
                "confidence": 0.0,
                "classification": "uncertain",
                "reasoning": f"Failed to parse LLM response: {e}",
                "signals": [],
            }

        # Validate and normalize the result
        result.setdefault("garbage_probability", 0.5)
        result.setdefault("confidence", 0.0)
        result.setdefault("classification", "uncertain")
        result.setdefault("reasoning", "")
        result.setdefault("signals", [])

        # Clamp probabilities to [0, 1]
        result["garbage_probability"] = max(0.0, min(1.0, float(result["garbage_probability"])))
        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

        # Validate classification
        valid_classifications = {"garbage", "likely_garbage", "uncertain", "likely_signal", "signal"}
        if result["classification"] not in valid_classifications:
            result["classification"] = "uncertain"

        return {
            "username": username,
            "garbage_probability": result["garbage_probability"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "signals": result["signals"],
            "classification": result["classification"],
        }



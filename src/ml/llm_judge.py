"""LLM-as-a-Judge module for bot detection evaluation.

Uses Gemini to evaluate whether Twitter accounts are bots, creating
pseudo-labeled ground truth for evaluating and improving the ML bot
detection model.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger("jafar.ml.llm_judge")

BOT_JUDGE_SYSTEM_PROMPT = """You are an expert Twitter bot detection analyst. Your job is to analyze a Twitter account's behavior and determine if it is likely a bot or automated account.

Analyze the following signals:

STRONG BOT INDICATORS:
- Extremely regular posting intervals (humans are irregular)
- High volume of tweets with near-identical structure or templates
- Tweets that are primarily links with minimal original commentary
- Engagement patterns that don't match content quality (e.g., zero engagement consistently)
- Posting at all hours with no sleep pattern
- Content that reads like automated summaries or RSS feeds

STRONG HUMAN INDICATORS:
- Irregular posting patterns with natural gaps
- Original analysis, opinions, or personal experiences
- Conversational replies and engagement with others
- Varied content style and topics
- Natural language patterns including typos, slang, emotional expression
- Engagement that correlates with content quality

NUANCE:
- News aggregator accounts may look bot-like but serve a purpose — still flag them as automated
- Some humans post very regularly — look at content originality as the tiebreaker
- High-frequency posters aren't necessarily bots if content is original
- Consider the COMBINATION of signals, not any single indicator

You must respond with valid JSON only, no other text."""

RESPONSE_FORMAT = """Respond with this exact JSON structure:
{
    "bot_probability": <float 0-1>,
    "confidence": <float 0-1>,
    "classification": "<bot|likely_bot|uncertain|likely_human|human>",
    "reasoning": "<2-3 sentence explanation>",
    "signals": ["<signal 1>", "<signal 2>", ...]
}"""

# Threshold for classifying as "bot" in binary comparisons
BOT_THRESHOLD = 0.6


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
            Dict with bot_probability, confidence, reasoning, signals,
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
                "bot_probability": 0.5,
                "confidence": 0.0,
                "classification": "uncertain",
                "reasoning": f"Failed to parse LLM response: {e}",
                "signals": [],
            }

        # Validate and normalize the result
        result.setdefault("bot_probability", 0.5)
        result.setdefault("confidence", 0.0)
        result.setdefault("classification", "uncertain")
        result.setdefault("reasoning", "")
        result.setdefault("signals", [])

        # Clamp probabilities to [0, 1]
        result["bot_probability"] = max(0.0, min(1.0, float(result["bot_probability"])))
        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

        # Validate classification
        valid_classifications = {"bot", "likely_bot", "uncertain", "likely_human", "human"}
        if result["classification"] not in valid_classifications:
            result["classification"] = "uncertain"

        return {
            "username": username,
            "bot_probability": result["bot_probability"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "signals": result["signals"],
            "classification": result["classification"],
        }

    async def judge_batch(
        self,
        accounts: list[dict[str, Any]],
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """Judge multiple accounts with concurrency control.

        Args:
            accounts: List of dicts, each with 'username' and 'tweets' keys.
                Optionally includes 'ml_features'.
            max_concurrent: Maximum concurrent LLM calls.

        Returns:
            List of judgment dicts.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _judge_with_semaphore(account: dict) -> dict[str, Any]:
            async with semaphore:
                return await self.judge_account(
                    username=account["username"],
                    tweets=account["tweets"],
                    ml_features=account.get("ml_features"),
                )

        tasks = [_judge_with_semaphore(account) for account in accounts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        judgments = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                username = accounts[i]["username"]
                logger.error("Failed to judge @%s: %s", username, result)
                judgments.append(
                    {
                        "username": username,
                        "bot_probability": 0.5,
                        "confidence": 0.0,
                        "reasoning": f"Error during judgment: {result}",
                        "signals": [],
                        "classification": "uncertain",
                    }
                )
            else:
                judgments.append(result)

        return judgments

    async def evaluate_ml_model(
        self,
        ml_scores: list[dict[str, Any]],
        sample_size: int = 50,
    ) -> dict[str, Any]:
        """Compare ML bot scores against LLM judgments to compute evaluation metrics.

        Takes ML model outputs, samples accounts, judges them with the LLM,
        and computes agreement metrics.

        Args:
            ml_scores: List of dicts from BotScorer.score_all_accounts(),
                each with 'username', 'bot_score', and 'features' keys.
                Must also include a 'tweets' key with tweet data for LLM judgment.
            sample_size: Number of accounts to sample for evaluation.

        Returns:
            Dict with agreement_rate, precision, recall, f1_score,
            confusion_matrix, and disagreements.
        """
        # Sample accounts (take all if fewer than sample_size)
        import random

        sample = random.sample(ml_scores, min(sample_size, len(ml_scores)))

        # Build batch for LLM judgment
        accounts_for_judgment = [
            {
                "username": s["username"],
                "tweets": s["tweets"],
                "ml_features": s.get("features"),
            }
            for s in sample
        ]

        llm_judgments = await self.judge_batch(accounts_for_judgment)

        # Build lookup for LLM results
        llm_by_user = {j["username"]: j for j in llm_judgments}

        # Compute confusion matrix
        tp = fp = tn = fn = 0
        disagreements = []

        for s in sample:
            username = s["username"]
            ml_is_bot = s["bot_score"] > BOT_THRESHOLD
            llm_result = llm_by_user.get(username, {})
            llm_is_bot = llm_result.get("bot_probability", 0.5) > BOT_THRESHOLD

            if ml_is_bot and llm_is_bot:
                tp += 1
            elif ml_is_bot and not llm_is_bot:
                fp += 1
                disagreements.append(
                    {
                        "username": username,
                        "ml_bot_score": s["bot_score"],
                        "llm_bot_probability": llm_result.get("bot_probability"),
                        "llm_classification": llm_result.get("classification"),
                        "llm_reasoning": llm_result.get("reasoning"),
                        "type": "false_positive",
                    }
                )
            elif not ml_is_bot and llm_is_bot:
                fn += 1
                disagreements.append(
                    {
                        "username": username,
                        "ml_bot_score": s["bot_score"],
                        "llm_bot_probability": llm_result.get("bot_probability"),
                        "llm_classification": llm_result.get("classification"),
                        "llm_reasoning": llm_result.get("reasoning"),
                        "type": "false_negative",
                    }
                )
            else:
                tn += 1

        total = tp + fp + tn + fn
        agreement_rate = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "total_evaluated": total,
            "agreement_rate": agreement_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "disagreements": disagreements,
        }

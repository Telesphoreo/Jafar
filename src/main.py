"""
Main Orchestration Script for Twitter Sentiment Analysis.

This script coordinates the full pipeline:
1. Scout (Scraper): Broad search across economic topics
2. Investigator (NER): Extract trending entities using spaCy
3. Deep Dive: Targeted scraping of trending entities
4. Analyst (LLM): Generate cohesive sentiment summary
5. Reporter: Email the digest

SETUP REQUIRED:
1. Copy .env.example to .env and fill in credentials
2. Add Twitter accounts to twscrape:
   - Create accounts.txt: username:password:email:email_password
   - twscrape add_accounts accounts.txt username:password:email:email_password
   - twscrape login_accounts (use --manual for non-IMAP emails)
   - twscrape accounts (to verify)
3. Download spaCy model:
   python -m spacy download en_core_web_sm
4. Install dependencies:
   uv sync
"""

import asyncio
import logging
import sys
from datetime import datetime

from .config import config
from .scraper import TwitterScraper, ScrapedTweet
from .analyzer import TrendAnalyzer
from .llm import create_llm_provider, LLMProvider
from .reporter import create_reporter_from_config

logger = logging.getLogger("twitter_sentiment.main")

# System prompt for the LLM analyst
ANALYST_SYSTEM_PROMPT = """You are an expert financial analyst tasked with summarizing economic sentiment from Twitter/X data.

Your role is to:
1. Identify the overall market sentiment (bullish, bearish, or neutral)
2. Highlight key themes and narratives driving discussion
3. Note any significant events, announcements, or concerns
4. Distinguish between informed analysis and noise/spam

IMPORTANT GUIDELINES:
- IGNORE obvious bots, spam, and promotional content
- IGNORE repetitive copy-paste tweets or coordinated campaigns
- FOCUS on tweets that appear to be from informed individuals, analysts, or news sources
- LOOK FOR consensus views and notable contrarian opinions
- Be CONCISE but comprehensive
- Use professional, objective language
- If sentiment is mixed, explain the different perspectives
- Highlight any emerging risks or opportunities mentioned

Your summary should be actionable and insightful for someone tracking economic trends."""


def format_tweets_for_llm(trend_tweets: dict[str, list[ScrapedTweet]]) -> str:
    """
    Format collected tweets into a structured prompt for the LLM.

    Args:
        trend_tweets: Dictionary mapping trends to their tweets.

    Returns:
        Formatted string for LLM analysis.
    """
    parts = ["# Twitter/X Economic Sentiment Data\n"]
    parts.append(f"Collected on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    for trend, tweets in trend_tweets.items():
        if not tweets:
            continue

        parts.append(f"\n## Topic: {trend}")
        parts.append(f"({len(tweets)} tweets)\n")

        for i, tweet in enumerate(tweets, 1):
            # Skip retweets for cleaner analysis
            if tweet.is_retweet:
                continue

            engagement = f"[Likes: {tweet.likes}, RTs: {tweet.retweets}]"
            parts.append(f"\n{i}. @{tweet.username} {engagement}")
            parts.append(f"   {tweet.text[:500]}")  # Truncate long tweets

    return "\n".join(parts)


async def analyze_with_llm(
    llm: LLMProvider,
    trend_tweets: dict[str, list[ScrapedTweet]],
) -> str:
    """
    Use the LLM to generate an economic sentiment analysis.

    Args:
        llm: The LLM provider to use.
        trend_tweets: Dictionary mapping trends to their tweets.

    Returns:
        Generated analysis text.
    """
    logger.info(f"Generating analysis with {llm.provider_name} ({llm.model_name})")

    # Format the data for the LLM
    data_prompt = format_tweets_for_llm(trend_tweets)

    user_prompt = f"""Based on the following Twitter/X data about economic topics, provide a comprehensive sentiment analysis.

{data_prompt}

Please provide:
1. **Overall Sentiment**: Is the general mood bullish, bearish, or mixed?
2. **Key Themes**: What are the main topics driving discussion?
3. **Notable Insights**: Any interesting observations or contrarian views?
4. **Potential Concerns**: Any risks or warnings being discussed?
5. **Summary**: A brief 2-3 sentence takeaway for someone who needs the quick version.

Remember to filter out obvious spam, bots, and promotional content in your analysis."""

    try:
        response = await llm.generate(
            prompt=user_prompt,
            system_prompt=ANALYST_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=2000,
        )
        logger.info(f"Analysis generated ({response.token_count} tokens used)")
        return response.content

    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        raise


async def run_pipeline() -> bool:
    """
    Run the full sentiment analysis pipeline.

    Returns:
        True if the pipeline completed successfully.
    """
    # Set up logging
    logger = config.setup_logging()
    logger.info("=" * 60)
    logger.info("Twitter Economic Sentiment Analysis Pipeline")
    logger.info("=" * 60)

    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        return False

    # Initialize components
    scraper = TwitterScraper(db_path=config.twitter.db_path)
    analyzer = TrendAnalyzer()

    try:
        llm = create_llm_provider(
            provider=config.app.llm_provider,
            openai_api_key=config.openai.api_key,
            openai_model=config.openai.model,
            google_api_key=config.google.api_key,
            google_model=config.google.model,
        )
    except ValueError as e:
        logger.error(f"Failed to create LLM provider: {e}")
        return False

    reporter = create_reporter_from_config(
        host=config.smtp.host,
        port=config.smtp.port,
        username=config.smtp.username,
        password=config.smtp.password,
        use_tls=config.smtp.use_tls,
        email_from=config.smtp.email_from,
        email_to=config.smtp.email_to,
    )

    try:
        # ============================================================
        # STEP 1: THE SCOUT - Broad Twitter Search
        # ============================================================
        logger.info("\n[STEP 1] THE SCOUT: Gathering broad economic tweets...")

        broad_tweets = await scraper.get_broad_tweets(
            topics=config.app.broad_topics,
            limit_per_topic=config.app.broad_tweet_limit,
        )

        if not broad_tweets:
            logger.error("No tweets retrieved from broad search. Check twscrape setup.")
            logger.error("Make sure you've run: twscrape add_accounts <file> <format> && twscrape login_accounts")
            return False

        logger.info(f"Collected {len(broad_tweets)} tweets from broad search")

        # ============================================================
        # STEP 2: THE INVESTIGATOR - NER Analysis
        # ============================================================
        logger.info("\n[STEP 2] THE INVESTIGATOR: Extracting trending entities...")

        trends = analyzer.extract_trends(
            tweets=broad_tweets,
            top_n=config.app.top_trends_count,
        )

        if not trends:
            logger.warning("No trends extracted. Using fallback topics.")
            trends = ["Federal Reserve", "Stock Market", "Inflation"]

        logger.info(f"Top trends identified: {trends}")

        # ============================================================
        # STEP 3: THE DEEP DIVE - Targeted Scraping
        # ============================================================
        logger.info("\n[STEP 3] THE DEEP DIVE: Gathering sentiment for each trend...")

        trend_tweets = await scraper.get_specific_sentiment(
            trends=trends,
            limit_per_trend=config.app.specific_tweet_limit,
        )

        total_tweets = sum(len(t) for t in trend_tweets.values())
        logger.info(f"Collected {total_tweets} tweets for sentiment analysis")

        # ============================================================
        # STEP 4: THE ANALYST - LLM Summary
        # ============================================================
        logger.info("\n[STEP 4] THE ANALYST: Generating AI-powered summary...")

        analysis = await analyze_with_llm(llm, trend_tweets)

        if not analysis:
            logger.error("LLM analysis returned empty result")
            return False

        logger.info("Analysis generated successfully")
        logger.debug(f"Analysis preview: {analysis[:200]}...")

        # ============================================================
        # STEP 5: THE REPORTER - Email Digest
        # ============================================================
        logger.info("\n[STEP 5] THE REPORTER: Sending email digest...")

        provider_info = f"{llm.provider_name} {llm.model_name}"
        success = reporter.send_email(
            report_content=analysis,
            trends=trends,
            tweet_count=total_tweets,
            provider_info=provider_info,
        )

        if success:
            logger.info("Email digest sent successfully!")
        else:
            logger.error("Failed to send email digest")
            # Don't return False here - the analysis was still generated
            # User might want to see it even if email failed

        # ============================================================
        # COMPLETE
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)

        # Print the analysis to console as well
        print("\n" + "=" * 60)
        print("ECONOMIC SENTIMENT DIGEST")
        print("=" * 60)
        print(f"\nTrending Topics: {', '.join(trends)}")
        print(f"Tweets Analyzed: {total_tweets}")
        print("\n" + "-" * 60)
        print(analysis)
        print("-" * 60 + "\n")

        return True

    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        return False

    finally:
        await scraper.close()


def main():
    """Entry point for the application."""
    print("""
    ==============================================================
    |     Twitter/X Economic Sentiment Analysis                  |
    |     Dynamic Discovery & AI-Powered Digest                  |
    ==============================================================
    """)

    try:
        success = asyncio.run(run_pipeline())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

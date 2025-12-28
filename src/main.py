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
from .analyzer import TrendAnalyzer, DiscoveredTrend
from .llm import create_llm_provider, LLMProvider
from .reporter import create_reporter_from_config
from .history import DigestHistory, calculate_signal_strength
from .memory import create_memory_manager, MemoryManager
from .checkpoint import CheckpointManager
from .fact_checker import MarketFactChecker

logger = logging.getLogger("jafar.main")

# System prompt for the LLM analyst - CALIBRATED FOR SKEPTICISM + HISTORICAL AWARENESS
ANALYST_SYSTEM_PROMPT = """You are a skeptical, experienced financial analyst with a long memory. Your job is to separate SIGNAL from NOISE, and to recognize when history rhymes.

CRITICAL MINDSET:
- Most days are BORING. Normal market chatter is not news.
- Your default assumption should be "nothing unusual today" unless data proves otherwise.
- A seasoned trader would roll their eyes at hype. Channel that energy.
- Just because people are discussing something doesn't mean it matters.
- Engagement metrics can be gamed. Be skeptical of viral content.

WHAT ACTUALLY MATTERS (rare):
- Genuine supply/demand shocks (not just people talking about them)
- Unusual volume/engagement that's 5-10x normal levels
- Multiple independent sources converging on the same narrative
- Information that ISN'T already priced in by mainstream news

WHAT DOESN'T MATTER (common):
- People complaining about the Fed, inflation, or politicians (eternal noise)
- Generic bullish/bearish sentiment (this is ALWAYS present)
- Crypto pumps and meme stock chatter (unless specifically asked)
- Recycled narratives from last week/month
- Promotional content or coordinated campaigns

FACT-CHECKING PROTOCOL:
When verified market data is provided, you MUST use it to validate claims:
1. Compare tweet claims against the actual price/volume data
2. Flag claims that contradict the verified numbers (e.g., "silver crashing" when data shows +3%)
3. Note when sentiment ALIGNS with real price action - this strengthens the signal
4. "Massive volume" claims should show >2x average in the data; otherwise it's exaggeration
5. "All-time high" or "52-week high" claims should match the Notes column
6. In your assessment, classify claims as:
   - VERIFIED: Claims that match the market data
   - EXAGGERATED: Directionally correct but overstated
   - FALSE: Claims that directly contradict the data
   - UNVERIFIABLE: Claims about assets not in the provided data

This is CRITICAL: Do NOT let unverified hype drive your signal strength rating.
If tweets scream "SILVER MOONING!!!" but the data shows +0.5%, that's LOW signal, not HIGH.

HISTORICAL PARALLELS - USE WITH EXTREME CARE:
"History doesn't repeat itself, but it often rhymes." - Mark Twain

When historical parallels are provided:
- ONLY mention them if the similarity is SUBSTANTIVE, not superficial
- Ask: Is this just keyword overlap, or are the underlying dynamics similar?
- Consider: What happened AFTER those historical periods? Is that instructive?
- Be honest: Sometimes there IS no meaningful parallel. Say so.
- Never force a connection just because data is available

GOOD parallel usage:
- "This silver rally shows similar engagement patterns to March 2024, when physical demand surged before a 15% price move"
- "Unlike the semiconductor chatter in Q2, today's discussion includes specific supply chain concerns"

BAD parallel usage (AVOID):
- "This reminds me of last Tuesday" (too recent, not meaningful)
- "Similar to every time gold is mentioned" (too generic)
- "History shows..." without specific context (lazy analysis)

YOUR OUTPUT CALIBRATION:
1. **Signal Strength**: Rate today as HIGH / MEDIUM / LOW / NONE
   - HIGH: Genuinely unusual activity, potential market-moving (rare - maybe 1-2x per month)
   - MEDIUM: Interesting developments worth monitoring (weekly occurrence)
   - LOW: Normal market chatter, nothing actionable (most days)
   - NONE: Below-average activity, truly nothing to report

2. **If signal is LOW or NONE**: Say so clearly. "Today's Twitter activity shows normal market discussion with no unusual signals." is a VALID and GOOD response.

3. **Actionability**: Even if something IS trending, explicitly state whether action is warranted:
   - "Interesting to monitor but NOT actionable yet"
   - "Worth researching further before any decisions"
   - "Pure speculation at this point"
   - Only rarely: "This warrants immediate attention"

4. **Historical Comparison**: If parallels exist, analyze them critically. If not, say "No meaningful historical parallels identified."

NEVER:
- Manufacture urgency where none exists
- Suggest action on every digest
- Force historical parallels where none exist
- Hype normal market discussion as "breaking"
- Use exclamation points or urgent language unless truly warranted
- Assume the reader should do anything based on Twitter sentiment alone

Remember: The reader is sophisticated. They don't need hand-holding. They need honest signal assessment and thoughtful historical context when it genuinely applies."""


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
    historical_context: str = "",
    parallel_context: str = "",
    fact_check_context: str = "",
    top_engagement: float = 0,
) -> tuple[str, str, bool]:
    """
    Use the LLM to generate a CALIBRATED sentiment analysis.

    Args:
        llm: The LLM provider to use.
        trend_tweets: Dictionary mapping trends to their tweets.
        historical_context: Formatted string of recent digest history.
        parallel_context: Historical parallels from vector search.
        fact_check_context: Real market data for fact-checking claims.
        top_engagement: Highest engagement score from today's trends.

    Returns:
        Tuple of (analysis_text, signal_strength, is_notable).
    """
    logger.info(f"Generating analysis with {llm.provider_name} ({llm.model_name})")

    # Format the data for the LLM
    data_prompt = format_tweets_for_llm(trend_tweets)

    user_prompt = f"""Analyze the following Twitter/X data. Be SKEPTICAL - most days are boring.

{historical_context}

{parallel_context}

{fact_check_context}

## Today's Data
Top engagement score: {top_engagement:.0f}
{data_prompt}

## Required Output Format

**SIGNAL STRENGTH**: [HIGH / MEDIUM / LOW / NONE]
(Be honest - HIGH should be rare, maybe 1-2x per month)

**ASSESSMENT**:
[2-3 sentences. If signal is LOW/NONE, say "Normal market chatter, nothing unusual" - that's a valid response]

**TRENDS OBSERVED**:
[Bullet points of what's being discussed - factual, not hyped]

**ACTIONABILITY**: [NOT ACTIONABLE / MONITOR ONLY / WORTH RESEARCHING / WARRANTS ATTENTION]
[1 sentence explaining why]

**HISTORICAL PARALLEL**:
[ONLY if genuinely meaningful - "History rhymes: [specific parallel with what happened after]"
OR "No meaningful historical parallels identified" - this is a valid and often correct answer]

**BOTTOM LINE**:
[1 sentence. Be direct. "Nothing worth acting on today" is perfectly acceptable]

Remember: Your job is to FILTER, not to HYPE. A good analyst knows when to say "pass"."""

    try:
        response = await llm.generate(
            prompt=user_prompt,
            system_prompt=ANALYST_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=2000,
        )
        logger.info(f"Analysis generated ({response.token_count} tokens used)")

        # Parse signal strength from response
        content = response.content
        signal_strength = "low"  # default
        is_notable = False

        content_upper = content.upper()
        if "**SIGNAL STRENGTH**: HIGH" in content_upper or "SIGNAL STRENGTH: HIGH" in content_upper:
            signal_strength = "high"
            is_notable = True
        elif "**SIGNAL STRENGTH**: MEDIUM" in content_upper or "SIGNAL STRENGTH: MEDIUM" in content_upper:
            signal_strength = "medium"
        elif "**SIGNAL STRENGTH**: NONE" in content_upper or "SIGNAL STRENGTH: NONE" in content_upper:
            signal_strength = "none"

        logger.info(f"Signal strength: {signal_strength.upper()}, Notable: {is_notable}")
        return content, signal_strength, is_notable

    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        raise


async def run_pipeline() -> bool:
    """
    Run the full sentiment analysis pipeline with checkpointing.

    Supports resumption after interruption - progress is saved after each step.

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

    # Initialize checkpoint manager
    checkpoint = CheckpointManager()

    # Check for existing checkpoint to resume
    resuming = checkpoint.should_resume()
    if resuming:
        state = checkpoint.get_state()
        logger.info(f"Resuming from checkpoint: {state.run_id}")
        logger.info(f"  - Step 1 (broad scraping): {'DONE' if state.step1_complete else f'{len(state.topics_completed)}/{len(state.topics_completed) + len(state.topics_remaining)} topics'}")
        logger.info(f"  - Step 2 (trends): {'DONE' if state.step2_complete else 'PENDING'}")
        logger.info(f"  - Step 3 (deep dive): {'DONE' if state.step3_complete else 'PENDING'}")
        logger.info(f"  - Step 4 (analysis): {'DONE' if state.step4_complete else 'PENDING'}")
        logger.info(f"  - Step 5 (email): {'DONE' if state.step5_complete else 'PENDING'}")
        logger.info(f"  - Step 6 (history): {'DONE' if state.step6_complete else 'PENDING'}")
    else:
        logger.info("Starting fresh pipeline run")
        checkpoint.start_new_run(topics=config.app.broad_topics)
        state = checkpoint.get_state()

    # Initialize components
    scraper = TwitterScraper(db_path=config.twitter.db_path)
    analyzer = TrendAnalyzer()
    history = DigestHistory()

    # Check if Twitter accounts are available before starting
    try:
        stats = await scraper.get_account_stats()
        active = stats.get("active", 0)
        total = stats.get("total", 0)

        if total == 0:
            logger.error("No Twitter accounts configured. Run 'uv run twscrape accounts' to check.")
            logger.error("Add accounts with: uv run python add_account.py <username> cookies.json")
            return False

        if active == 0:
            logger.warning(f"All {total} Twitter accounts are rate-limited or inactive")
            logger.warning("The pipeline will skip queries where no accounts are available")
            logger.warning("Consider adding more accounts or waiting for rate limits to reset")
        else:
            logger.info(f"Twitter accounts: {active}/{total} active")
    except Exception as e:
        logger.warning(f"Could not check Twitter account status: {e}")

    # Initialize vector memory system
    memory: MemoryManager | None = None
    if config.memory.enabled:
        try:
            logger.info("Initializing vector memory system...")
            memory = await create_memory_manager(
                store_type=config.memory.store_type,
                embedding_provider=config.memory.embedding_provider,
                openai_api_key=config.openai.api_key,
                openai_embedding_model=config.memory.openai_embedding_model,
                embedding_dimensions=config.memory.embedding_dimensions,
                postgres_url=config.memory.postgres_url,
                chroma_path=config.memory.chroma_path,
            )
            memory_count = await memory.vector_store.count()
            logger.info(f"Vector memory initialized with {memory_count} stored memories")
        except Exception as e:
            logger.warning(f"Failed to initialize vector memory: {e}")
            memory = None

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
        # STEP 1: THE SCOUT - Broad Twitter Search (with checkpointing)
        # ============================================================
        if not state.step1_complete:
            logger.info("\n[STEP 1] THE SCOUT: Gathering broad economic tweets...")
            logger.info(f"Topics: {len(state.topics_remaining)} remaining, {len(state.topics_completed)} completed")

            # Callback to save progress after each topic
            def on_topic_done(topic: str, tweets: list[ScrapedTweet]) -> None:
                checkpoint.mark_topic_complete(topic, tweets)

            # Get any already-collected tweets from checkpoint
            existing_tweets = checkpoint.get_broad_tweets() if state.topics_completed else []

            # Scrape remaining topics incrementally
            new_tweets = await scraper.get_broad_tweets_incremental(
                topics=config.app.broad_topics,
                limit_per_topic=config.app.broad_tweet_limit,
                on_topic_complete=on_topic_done,
                skip_topics=state.topics_completed,
                timeout=config.app.search_timeout,
            )

            broad_tweets = existing_tweets + new_tweets
            checkpoint.complete_step1()
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 1] Skipping (already complete)")
            broad_tweets = checkpoint.get_broad_tweets()

        if not broad_tweets:
            logger.error("No tweets retrieved. Check twscrape setup.")
            return False

        logger.info(f"Total broad tweets: {len(broad_tweets)}")

        # ============================================================
        # STEP 2: THE INVESTIGATOR - NER Analysis
        # ============================================================
        if not state.step2_complete:
            logger.info("\n[STEP 2] THE INVESTIGATOR: Extracting trending entities...")

            trends = analyzer.extract_trends(
                tweets=broad_tweets,
                top_n=config.app.top_trends_count,
                min_mentions=config.app.min_trend_mentions,
                min_authors=config.app.min_trend_authors,
            )

            if not trends:
                logger.warning("No trends extracted. Using fallback topics.")
                trends = ["Federal Reserve", "Stock Market", "Inflation"]

            checkpoint.save_trends(trends)
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 2] Skipping (already complete)")
            trends = state.trends

        logger.info(f"Trends: {trends}")

        # ============================================================
        # STEP 3: THE DEEP DIVE - Targeted Scraping
        # ============================================================
        if not state.step3_complete:
            logger.info("\n[STEP 3] THE DEEP DIVE: Gathering sentiment for each trend...")

            # Callback to save progress after each trend
            def on_trend_done(trend: str, tweets: list[ScrapedTweet]) -> None:
                checkpoint.mark_trend_scraped(trend, tweets)

            already_scraped = list(state.trend_tweets.keys())

            new_trend_tweets = await scraper.get_specific_sentiment_incremental(
                trends=trends,
                limit_per_trend=config.app.specific_tweet_limit,
                on_trend_complete=on_trend_done,
                skip_trends=already_scraped,
                timeout=config.app.search_timeout,
            )

            # Merge with already-scraped trends
            trend_tweets = checkpoint.get_trend_tweets()
            trend_tweets.update(new_trend_tweets)

            checkpoint.complete_step3()
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 3] Skipping (already complete)")
            trend_tweets = checkpoint.get_trend_tweets()

        total_tweets = sum(len(t) for t in trend_tweets.values())
        logger.info(f"Total trend tweets: {total_tweets}")

        # ============================================================
        # STEP 3.5: THE FACT CHECKER - Market Data Verification
        # ============================================================
        fact_check_context = ""
        if config.fact_checker.enabled:
            logger.info("\n[STEP 3.5] THE FACT CHECKER: Verifying market claims...")
            try:
                fact_checker = MarketFactChecker(
                    cache_ttl_minutes=config.fact_checker.cache_ttl_minutes,
                    price_tolerance_pct=config.fact_checker.price_tolerance_pct,
                )

                # Extract symbols from discovered trends
                symbols = fact_checker.extract_symbols_from_trends(trends)
                logger.info(f"Extracted {len(symbols)} market symbols from trends")

                # Fetch real market data
                market_data = await fact_checker.fetch_market_data(symbols)
                logger.info(f"Fetched market data for {len(market_data)} symbols")

                # Format for LLM context
                fact_check_context = fact_checker.format_for_llm(market_data, trends)

            except Exception as e:
                logger.warning(f"Fact checking failed (continuing without): {e}")

        # ============================================================
        # STEP 4: THE ANALYST - LLM Summary
        # ============================================================
        if not state.step4_complete:
            logger.info("\n[STEP 4] THE ANALYST: Generating calibrated analysis...")

            historical_context = history.format_context_for_llm(days=7)
            baseline = history.get_baseline_stats(days=30)

            top_engagement = 0.0
            for tweets_list in trend_tweets.values():
                for tweet in tweets_list:
                    eng = (tweet.likes * 1.0) + (tweet.retweets * 0.5) + (tweet.replies * 0.3)
                    top_engagement = max(top_engagement, eng)

            logger.info(f"Top engagement: {top_engagement:.0f} (avg: {baseline.get('avg_top_engagement', 0):.0f})")

            # Search for historical parallels
            parallel_context = ""
            if memory:
                try:
                    parallels = await memory.find_parallels(
                        trends=trends,
                        themes=trends,
                        sentiment="unknown",
                        signal_strength="unknown",
                        limit=5,
                        min_similarity=config.memory.min_similarity,
                    )
                    if parallels:
                        logger.info(f"Found {len(parallels)} historical parallels")
                        parallel_context = await memory.format_parallels_for_llm(parallels)
                except Exception as e:
                    logger.warning(f"Error searching parallels: {e}")

            analysis, signal_strength, is_notable = await analyze_with_llm(
                llm,
                trend_tweets,
                historical_context=historical_context,
                parallel_context=parallel_context,
                fact_check_context=fact_check_context,
                top_engagement=top_engagement,
            )

            if not analysis:
                logger.error("LLM analysis returned empty result")
                return False

            checkpoint.save_analysis(analysis, signal_strength, is_notable, top_engagement)
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 4] Skipping (already complete)")
            analysis = state.analysis
            signal_strength = state.signal_strength
            is_notable = state.is_notable
            top_engagement = state.top_engagement

        logger.info(f"Signal: {signal_strength.upper()}, Notable: {is_notable}")

        # ============================================================
        # STEP 5: THE REPORTER - Email Digest
        # ============================================================
        if not state.step5_complete:
            logger.info("\n[STEP 5] THE REPORTER: Sending email digest...")

            provider_info = f"{llm.provider_name} {llm.model_name}"
            success = reporter.send_email(
                report_content=analysis,
                trends=trends,
                tweet_count=total_tweets,
                provider_info=provider_info,
                signal_strength=signal_strength,
            )

            if success:
                logger.info("Email sent successfully!")
            else:
                logger.warning("Failed to send email")

            checkpoint.complete_step5()
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 5] Skipping (already complete)")

        # ============================================================
        # STEP 6: STORE HISTORY
        # ============================================================
        if not state.step6_complete:
            logger.info("\n[STEP 6] Storing digest in history...")

            history.store_digest(
                trends=trends,
                tweet_count=total_tweets,
                digest_text=analysis,
                signal_strength=signal_strength,
                top_engagement=top_engagement,
                notable=is_notable,
            )

            if memory:
                try:
                    memory_record = await memory.create_memory(
                        trends=trends,
                        analysis=analysis,
                        signal_strength=signal_strength,
                        top_engagement=top_engagement,
                        tweet_count=total_tweets,
                        notable=is_notable,
                    )
                    await memory.store_memory(memory_record)
                    logger.info(f"Memory stored: {memory_record.id}")
                except Exception as e:
                    logger.warning(f"Failed to store memory: {e}")

            checkpoint.complete_step6()
        else:
            logger.info("\n[STEP 6] Skipping (already complete)")

        # ============================================================
        # COMPLETE - Clear checkpoint
        # ============================================================
        checkpoint.clear()

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)

        # Print results
        def safe_print(text: str) -> None:
            try:
                print(text)
            except UnicodeEncodeError:
                print(text.encode('ascii', 'ignore').decode('ascii'))

        print("\n" + "=" * 60)
        print("ECONOMIC SENTIMENT DIGEST")
        print(f"Signal Strength: {signal_strength.upper()}")
        print("=" * 60)
        safe_trends = [t.encode('ascii', 'ignore').decode('ascii').strip() for t in trends]
        print(f"\nTrending Topics: {', '.join(safe_trends)}")
        print(f"Tweets Analyzed: {total_tweets}")
        if is_notable:
            print("*** THIS DAY WAS FLAGGED AS NOTABLE ***")
        print("\n" + "-" * 60)
        safe_print(analysis)
        print("-" * 60 + "\n")

        return True

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user - progress saved to checkpoint")
        logger.info("Run again to resume from where you left off")
        raise

    except Exception as e:
        checkpoint.set_error(str(e))
        logger.exception(f"Pipeline failed: {e}")
        logger.info("Progress saved - run again to resume")
        return False

    finally:
        await scraper.close()
        if memory:
            await memory.close()


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

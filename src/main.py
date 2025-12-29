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
from .temporal_analyzer import TemporalTrendAnalyzer, TrendTimeline
from .diagnostics import DiagnosticsCollector, rotate_logs, should_send_admin_alert
import time

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
- Crypto pumps and meme stock chatter (unless specifically asked)
- Recycled narratives from last week/month
- Promotional content or coordinated campaigns
- Individual complaints without broader pattern

SENTIMENT ANALYSIS NUANCE:
- Background noise: Some level of bullish/bearish/inflation talk is ALWAYS present
- Signal: UNUSUAL SPIKES in sentiment (3x+ normal bearish chatter = meaningful fear signal)
- The truth is often between Twitter doom and actual market reality
- If everyone's suddenly talking about "risk" or "crash", that aggregate mood matters
- Compare sentiment intensity to what you'd expect on a normal day

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
    temporal_context: str = "",
    top_engagement: float = 0,
) -> tuple[str, str, bool, int]:
    """
    Use the LLM to generate a CALIBRATED sentiment analysis.

    Args:
        llm: The LLM provider to use.
        trend_tweets: Dictionary mapping trends to their tweets.
        historical_context: Formatted string of recent digest history.
        parallel_context: Historical parallels from vector search.
        fact_check_context: Real market data for fact-checking claims.
        temporal_context: Trend timeline analysis (consecutive days, gaps).
        top_engagement: Highest engagement score from today's trends.

    Returns:
        Tuple of (analysis_text, signal_strength, is_notable, token_count).
    """
    logger.info(f"Generating analysis with {llm.provider_name} ({llm.model_name})")

    # Format the data for the LLM
    data_prompt = format_tweets_for_llm(trend_tweets)

    user_prompt = f"""Analyze the following Twitter/X data. Be SKEPTICAL - most days are boring.

{historical_context}

{temporal_context}

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
        return content, signal_strength, is_notable, response.token_count

    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        raise


async def llm_filter_trends(
    llm: LLMProvider,
    candidates: list[str],
) -> list[str]:
    """
    Use LLM to filter trend candidates, keeping only actionable financial signals.

    This is a quick, cheap call (~500 tokens) that leverages LLM judgment to
    distinguish real market signals from noise that slipped through statistical filters.

    Args:
        llm: The LLM provider to use
        candidates: List of trend term strings to evaluate

    Returns:
        Filtered list of trends worth deep-diving
    """
    if not candidates:
        return []

    # Format candidates for the prompt
    candidates_str = "\n".join(f"- {c}" for c in candidates)

    prompt = f"""Review these trend candidates extracted from financial Twitter. Filter out generic noise while keeping meaningful market signals AND sentiment indicators.

CANDIDATES:
{candidates_str}

KEEP terms that are:
- Specific assets, tickers, or commodities (e.g., "Silver", "$NVDA", "Uranium")
- Concrete market events (e.g., "Short Squeeze", "Margin Call", "Earnings Miss")
- Economic indicators (e.g., "Inventories", "Payrolls", "CPI")
- Specific companies or sectors experiencing notable activity
- Market sentiment indicators that reveal crowd mood (e.g., "Risk", "Demand", "Buyers", "Selloff", "Panic", "Euphoria")

REJECT terms that are:
- Generic words with no financial meaning (e.g., "Things", "World", "Experience", "Crowd")
- Seasonal/holiday terms (e.g., "Christmas", "Weekend", "Birthday")
- Common nouns that slipped through filters (e.g., "Books", "Fiction", "Chat")
- Ambiguous words that could mean anything (e.g., "Gap" unless clearly about $GPS)

Remember: This is SENTIMENT analysis. Aggregate mood matters. If lots of people are discussing "Risk" or "Demand", that's valuable context even if you can't directly trade it.

Respond with ONLY a JSON array of terms to KEEP, like: ["Silver", "$NVDA", "Risk", "Demand"]
If none are worth keeping, respond with: []"""

    try:
        response = await llm.generate(
            prompt=prompt,
            system_prompt="You are a financial signal filter. Be ruthless - only keep terms that a trader could actually act on. Respond with ONLY a JSON array, no explanation.",
            temperature=0.3,  # Low temp for consistency
            max_tokens=200,
        )

        # Parse JSON response
        import json
        content = response.content.strip()

        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content
            content = content.rsplit("```", 1)[0] if "```" in content else content
            content = content.strip()

        filtered = json.loads(content)

        if not isinstance(filtered, list):
            logger.warning(f"LLM filter returned non-list: {content}")
            return candidates  # Fall back to original

        logger.info(f"LLM filter: {len(filtered)}/{len(candidates)} trends passed")
        return filtered

    except Exception as e:
        logger.warning(f"LLM filter failed, using all candidates: {e}")
        return candidates  # Fall back to original on error


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

    # Clean up old logs from .run directory
    if config.smtp.admin.enabled:
        try:
            rotate_logs(
                log_file=".run/pipeline.log",
                keep_count=config.smtp.admin.log_retention_count
            )
        except Exception as e:
            logger.warning(f"Failed to clean up logs: {e}")

    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        return False

    # Initialize checkpoint manager
    checkpoint = CheckpointManager()

    # Initialize diagnostics collector
    diagnostics = DiagnosticsCollector(run_id=checkpoint.run_id if hasattr(checkpoint, 'run_id') else datetime.now().strftime("%Y%m%d_%H%M%S"))

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
    # Automatically reset locks to prevent "stuck" accounts from previous interrupted runs
    await scraper.fix_locks()

    analyzer = TrendAnalyzer(model_name=config.app.spacy_model)
    history = DigestHistory()

    # Check if Twitter accounts are available before starting
    try:
        stats = await scraper.get_account_stats()
        active = stats.get("active", 0)
        total = stats.get("total", 0)

        # Record in diagnostics
        diagnostics.diagnostics.twitter_accounts_total = total
        diagnostics.diagnostics.twitter_accounts_active = active
        diagnostics.diagnostics.twitter_accounts_rate_limited = total - active

        if total == 0:
            logger.error("No Twitter accounts configured. Run 'uv run twscrape accounts' to check.")
            logger.error("Add accounts with: uv run python add_account.py <username> cookies.json")
            diagnostics.diagnostics.add_error("No Twitter accounts configured")
            return False

        if active == 0:
            logger.warning(f"All {total} Twitter accounts are rate-limited or inactive")
            logger.warning("The pipeline will skip queries where no accounts are available")
            logger.warning("Consider adding more accounts or waiting for rate limits to reset")
            diagnostics.diagnostics.add_warning(f"All {total} Twitter accounts are rate-limited or inactive")
        else:
            logger.info(f"Twitter accounts: {active}/{total} active")
    except Exception as e:
        logger.warning(f"Could not check Twitter account status: {e}")
        diagnostics.diagnostics.add_warning(f"Could not check Twitter account status: {e}")

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
        step1_start = time.time()
        if not state.step1_complete:
            logger.info("\n[STEP 1] THE SCOUT: Gathering broad economic tweets...")
            logger.info(f"Topics: {len(state.topics_remaining)} remaining, {len(state.topics_completed)} completed")

            diagnostics.diagnostics.broad_topics_attempted = len(config.app.broad_topics)

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

            diagnostics.diagnostics.broad_topics_completed = len(state.topics_completed)
            diagnostics.diagnostics.broad_tweets_scraped = len(broad_tweets)
        else:
            logger.info("\n[STEP 1] Skipping (already complete)")
            broad_tweets = checkpoint.get_broad_tweets()
            diagnostics.diagnostics.broad_topics_completed = len(config.app.broad_topics)
            diagnostics.diagnostics.broad_tweets_scraped = len(broad_tweets)

        diagnostics.diagnostics.time_step1_scraping = time.time() - step1_start

        if not broad_tweets:
            logger.error("No tweets retrieved. Check twscrape setup.")
            diagnostics.diagnostics.add_error("No tweets retrieved from broad scraping")
            return False

        logger.info(f"Total broad tweets: {len(broad_tweets)}")

        # ============================================================
        # STEP 2: THE INVESTIGATOR - NER Analysis
        # ============================================================
        step2_start = time.time()
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
                diagnostics.diagnostics.add_warning("No trends extracted, using fallback topics")
                trends = ["Federal Reserve", "Stock Market", "Inflation"]

            diagnostics.diagnostics.trends_discovered = len(trends)
            checkpoint.save_trends(trends)
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 2] Skipping (already complete)")
            trends = state.trends
            diagnostics.diagnostics.trends_discovered = len(trends)

        diagnostics.diagnostics.time_step2_analysis = time.time() - step2_start

        logger.info(f"Trends (pre-filter): {trends}")

        # ============================================================
        # STEP 2.5: LLM QUALITY FILTER - Validate candidates before deep dive
        # ============================================================
        # Only run LLM filter if we haven't started deep dive yet (trends might change)
        if trends and len(trends) > 0 and not state.step3_complete:
            logger.info("\n[STEP 2.5] LLM FILTER: Validating trend candidates...")
            filtered_trends = await llm_filter_trends(llm, trends)
            diagnostics.diagnostics.llm_calls_made += 1  # LLM filter call

            if filtered_trends != trends:
                # Save filtered trends to checkpoint
                trends = filtered_trends
                checkpoint.save_trends(trends)
                state = checkpoint.get_state()

            if not trends:
                logger.info("LLM filter rejected all candidates - quiet day")
                diagnostics.diagnostics.add_warning("LLM filter rejected all trend candidates")

        diagnostics.diagnostics.trends_filtered_by_llm = len(trends) if trends else 0
        logger.info(f"Trends (post-filter): {trends}")

        # ============================================================
        # STEP 3: THE DEEP DIVE - Targeted Scraping
        # ============================================================
        step3_start = time.time()
        if not state.step3_complete:
            logger.info("\n[STEP 3] THE DEEP DIVE: Gathering sentiment for each trend...")

            diagnostics.diagnostics.deep_dive_trends_attempted = len(trends) if trends else 0

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

            diagnostics.diagnostics.deep_dive_trends_completed = len(trend_tweets)
            diagnostics.diagnostics.deep_dive_tweets_scraped = sum(len(t) for t in trend_tweets.values())

            checkpoint.complete_step3()
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 3] Skipping (already complete)")
            trend_tweets = checkpoint.get_trend_tweets()
            diagnostics.diagnostics.deep_dive_trends_attempted = len(trends) if trends else 0
            diagnostics.diagnostics.deep_dive_trends_completed = len(trend_tweets)
            diagnostics.diagnostics.deep_dive_tweets_scraped = sum(len(t) for t in trend_tweets.values())

        diagnostics.diagnostics.time_step3_deep_dive = time.time() - step3_start

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

                diagnostics.diagnostics.fact_checks_performed = len(market_data)

                # Format for LLM context
                fact_check_context = fact_checker.format_for_llm(market_data, trends)

            except Exception as e:
                logger.warning(f"Fact checking failed (continuing without): {e}")
                diagnostics.diagnostics.add_warning(f"Fact checking failed: {e}")

        # ============================================================
        # STEP 3.75: TEMPORAL ANALYSIS - Track trend continuity
        # ============================================================
        logger.info("\n[STEP 3.75] TEMPORAL ANALYSIS: Analyzing trend timelines...")
        temporal_analyzer = TemporalTrendAnalyzer(
            history_db=history,
            consecutive_threshold=config.temporal.consecutive_threshold,
            gap_threshold_days=config.temporal.gap_threshold_days,
        )

        # Build trend_details from the scraped data
        trend_details_for_temporal = {}
        for trend, tweets_list in trend_tweets.items():
            if not tweets_list:
                continue

            mentions = len(tweets_list)
            total_eng = sum(
                (t.likes * 1.0) + (t.retweets * 0.5) + (t.replies * 0.3)
                for t in tweets_list if not t.is_retweet
            )

            # Find first/last seen timestamps
            timestamps = [t.created_at for t in tweets_list if t.created_at]
            first_seen = min(timestamps) if timestamps else datetime.now()
            last_seen = max(timestamps) if timestamps else datetime.now()

            trend_details_for_temporal[trend] = {
                'mentions': mentions,
                'engagement': total_eng,
                'first_seen': first_seen,
                'last_seen': last_seen,
            }

        # Analyze timelines for all trends
        timelines = temporal_analyzer.analyze_all_trends(trend_details_for_temporal)

        # Count temporal patterns (new, continuing, recurring)
        temporal_patterns = sum(1 for t in timelines.values() if t.is_new or t.is_continuing or t.is_recurring)
        diagnostics.diagnostics.temporal_patterns_detected = temporal_patterns

        # Format temporal context for LLM
        temporal_context = temporal_analyzer.format_context_for_llm(timelines)

        # ============================================================
        # STEP 4: THE ANALYST - LLM Summary
        # ============================================================
        step4_start = time.time()
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
                        diagnostics.diagnostics.vector_memories_searched = len(parallels)
                        parallel_context = await memory.format_parallels_for_llm(parallels)
                except Exception as e:
                    logger.warning(f"Error searching parallels: {e}")
                    diagnostics.diagnostics.add_warning(f"Error searching parallels: {e}")

            analysis, signal_strength, is_notable, tokens_used = await analyze_with_llm(
                llm,
                trend_tweets,
                historical_context=historical_context,
                parallel_context=parallel_context,
                fact_check_context=fact_check_context,
                temporal_context=temporal_context,
                top_engagement=top_engagement,
            )

            diagnostics.diagnostics.llm_calls_made += 1  # Main analysis call
            diagnostics.diagnostics.llm_tokens_used += tokens_used

            if not analysis:
                logger.error("LLM analysis returned empty result")
                diagnostics.diagnostics.add_error("LLM analysis returned empty result")
                return False

            checkpoint.save_analysis(analysis, signal_strength, is_notable, top_engagement)
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 4] Skipping (already complete)")
            analysis = state.analysis
            signal_strength = state.signal_strength
            is_notable = state.is_notable
            top_engagement = state.top_engagement

        diagnostics.diagnostics.time_step4_llm = time.time() - step4_start
        diagnostics.diagnostics.signal_strength = signal_strength
        diagnostics.diagnostics.notable = is_notable

        logger.info(f"Signal: {signal_strength.upper()}, Notable: {is_notable}")

        # ============================================================
        # STEP 5: THE REPORTER - Email Digest
        # ============================================================
        step5_start = time.time()
        if not state.step5_complete:
            logger.info("\n[STEP 5] THE REPORTER: Sending email digest...")

            provider_info = f"{llm.provider_name} {llm.model_name}"
            success = reporter.send_email(
                report_content=analysis,
                trends=trends,
                tweet_count=total_tweets,
                provider_info=provider_info,
                signal_strength=signal_strength,
                timelines=timelines,
            )

            diagnostics.diagnostics.email_sent = success

            if success:
                logger.info("Email sent successfully!")
            else:
                logger.warning("Failed to send email")
                diagnostics.diagnostics.add_error("Failed to send digest email")

            checkpoint.complete_step5()
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 5] Skipping (already complete)")
            diagnostics.diagnostics.email_sent = True  # Assume it was sent in previous run

        diagnostics.diagnostics.time_step5_email = time.time() - step5_start

        # ============================================================
        # STEP 6: STORE HISTORY
        # ============================================================
        step6_start = time.time()
        if not state.step6_complete:
            logger.info("\n[STEP 6] Storing digest in history...")

            history.store_digest(
                trends=trends,
                tweet_count=total_tweets,
                digest_text=analysis,
                signal_strength=signal_strength,
                top_engagement=top_engagement,
                notable=is_notable,
                trend_details=trend_details_for_temporal,
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
                    diagnostics.diagnostics.vector_memories_stored = 1
                except Exception as e:
                    logger.warning(f"Failed to store memory: {e}")
                    diagnostics.diagnostics.add_warning(f"Failed to store memory: {e}")

            checkpoint.complete_step6()
        else:
            logger.info("\n[STEP 6] Skipping (already complete)")

        diagnostics.diagnostics.time_step6_storage = time.time() - step6_start

        # ============================================================
        # COMPLETE - Clear checkpoint
        # ============================================================
        checkpoint.clear()

        # ============================================================
        # STEP 6.5: ADMIN DIAGNOSTICS - Send admin email if needed
        # ============================================================
        if config.smtp.admin.enabled:
            logger.info("\n[STEP 6.5] ADMIN DIAGNOSTICS: Checking if alert needed...")

            # Finalize diagnostics
            final_diagnostics = diagnostics.finalize()

            # Check if admin should be alerted
            should_alert, alert_reason = should_send_admin_alert(final_diagnostics)

            # Send admin email if alert needed or if send_on_success is enabled
            if should_alert or config.smtp.admin.send_on_success:
                admin_recipients = config.smtp.admin.recipients if config.smtp.admin.recipients else config.smtp.email_to

                admin_success = reporter.send_admin_email(
                    diagnostics=final_diagnostics,
                    alert_reason=alert_reason,
                    admin_recipients=admin_recipients,
                )

                final_diagnostics.admin_email_sent = admin_success

                if admin_success:
                    logger.info(f"Admin diagnostics email sent: {alert_reason if should_alert else 'Routine report'}")
                else:
                    logger.warning("Failed to send admin diagnostics email")
            else:
                logger.info(f"No admin alert needed: {alert_reason}")
        else:
            # Still finalize for logging
            diagnostics.finalize()

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

        # Send admin alert for interruption if enabled
        if config.smtp.admin.enabled:
            diagnostics.diagnostics.add_error("Pipeline interrupted by user")
            final_diagnostics = diagnostics.finalize()
            should_alert, alert_reason = should_send_admin_alert(final_diagnostics)

            if should_alert:
                admin_recipients = config.smtp.admin.recipients if config.smtp.admin.recipients else config.smtp.email_to
                reporter.send_admin_email(
                    diagnostics=final_diagnostics,
                    alert_reason="Pipeline interrupted by user",
                    admin_recipients=admin_recipients,
                )

        raise

    except Exception as e:
        checkpoint.set_error(str(e))
        logger.exception(f"Pipeline failed: {e}")
        logger.info("Progress saved - run again to resume")

        # Send admin alert for failure
        if config.smtp.admin.enabled:
            diagnostics.diagnostics.add_error(f"Pipeline failed: {str(e)}")
            final_diagnostics = diagnostics.finalize()

            admin_recipients = config.smtp.admin.recipients if config.smtp.admin.recipients else config.smtp.email_to
            try:
                reporter.send_admin_email(
                    diagnostics=final_diagnostics,
                    alert_reason=f"CRITICAL: Pipeline failed - {str(e)}",
                    admin_recipients=admin_recipients,
                )
            except Exception as email_error:
                logger.error(f"Failed to send admin alert email: {email_error}")

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

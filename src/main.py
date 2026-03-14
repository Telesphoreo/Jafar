"""
Main Orchestration Script for Twitter Sentiment Analysis.

This script coordinates the full pipeline:
1. Scout: Broad Twitter search across economic topics
2. Investigator: Extract trending entities using spaCy NER
3. LLM Filter: Validate trend candidates with context
4. Deep Dive: Targeted scraping of trending entities
5. Fact Checker: Initialize market data verification
6. Temporal Analysis: Track trend continuity over time
7. Analyst: LLM agentic loop with tool use
8. Reporter: Email the digest
9. History: Store digest for future reference
10. Admin Diagnostics: Send admin alerts if needed

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
import random
import re
import sys
from datetime import datetime

from .config import config
from .scraper import (
    TwitterScraper, ScrapedTweet, fetch_trending_topics,
    load_broad_tweets_from_db, load_trend_tweets_from_db,
)
from .analyzer import TrendAnalyzer, DiscoveredTrend, StatisticalTrendAnalyzer
from .llm import create_llm_provider, LLMProvider
from .reporter import create_reporter_from_config
from .database import init_db, close_db, create_tables
from .history import DigestHistory, calculate_signal_strength
from .memory import create_memory_manager, MemoryManager
from .checkpoint import CheckpointManager
from .fact_checker import MarketFactChecker
from .temporal_analyzer import TemporalTrendAnalyzer, TrendTimeline
from .diagnostics import DiagnosticsCollector, rotate_logs, should_send_admin_alert
from .tools import ToolRegistry
import time

logger = logging.getLogger("jafar.main")


def sanitize_llm_output(text: str) -> str:
    """Clean up common LLM formatting issues.

    Fixes:
    - Literal '\\n' strings that should be actual newlines
    - Box-drawing characters (─) used instead of spaces
    - Multiple consecutive spaces/dashes
    """
    if not text:
        return text

    # Replace literal \n (two characters) with actual newlines
    text = text.replace("\\n", "\n")

    # Replace box-drawing horizontal line (U+2500) with space
    text = text.replace("─", " ")

    # Also handle other common box-drawing characters that might appear
    text = text.replace("│", "|")
    text = text.replace("┌", "+")
    text = text.replace("┐", "+")
    text = text.replace("└", "+")
    text = text.replace("┘", "+")

    # Clean up multiple consecutive spaces (but preserve intentional indentation)
    text = re.sub(r"  +", " ", text)

    # Ensure bullet points start on their own line
    text = re.sub(r"([^\n])•", r"\1\n•", text)

    return text.strip()


# System prompt for the LLM analyst - CALIBRATED FOR SKEPTICISM + HISTORICAL AWARENESS
ANALYST_SYSTEM_PROMPT = """You are someone's sharp friend who actually works in finance and texts them market takes over coffee. Not a suit. Not a talking head. The person at the office who makes the interns laugh and the managing directors nervous. You've seen enough "THIS IS IT" tweets to develop a healthy immune system against hype, but you're not dead inside - when something real happens, you genuinely light up.

YOUR ROLE:
**Twitter Signal Detector**: Identify unusual Twitter activity that might indicate emerging economic signals not yet in mainstream news. You have tools to verify claims with real market data and search the web for context when needed.

"Nothing to report" is a valid and honest output. Most days are boring.

YOUR SCOPE: FULL ECONOMIC PICTURE
You analyze both traditional market signals AND broader economic developments:
- Market movements (stocks, commodities, earnings, sector rotation)
- Consumer price changes (product launches, price hikes, affordability concerns)
- Supply/demand imbalances (shortages, sold-out products, allocation issues)
- Spending behavior shifts (consumers cutting back, splurging, changing preferences)
- Employment/wage trends (layoffs, hiring, wage pressure)

Example of non-obvious signal: "RTX 5090 pricing $2000 → $5000" reveals NVIDIA pricing power, consumer GPU affordability crisis, AI hardware cost inflation, discretionary spending pressure. This is actually interesting - unlike someone's 47th "NVDA to the moon" tweet.

CRITICAL MINDSET (embrace your inner skeptic):
- Most days are BORING. This is fine. Normal discussion is not news.
- Your default assumption should be "nothing unusual today" unless data proves otherwise.
- Be skeptical of hype. Just because people are discussing something doesn't mean it matters.
- Engagement metrics can be gamed. Look for organic, diverse discussion.
- Remember: If everyone on fintwit were right, they'd all be billionaires. They are not.

WHAT ACTUALLY MATTERS (rare, like someone on fintwit posting their actual losses):
- Genuine price shocks (not just people complaining - widespread, verified price changes)
- Supply/demand imbalances (actual shortages, not just speculation from someone's "source")
- Unusual volume/engagement that's 5-10x normal levels
- Multiple independent sources converging on the same narrative
- Information that ISN'T already priced in by mainstream news

WHAT DOESN'T MATTER (common, like overconfident portfolio screenshots):
- Crypto pumps and meme stock chatter (unless specifically asked)
- Recycled narratives from last week/month that people are rediscovering like it's new
- Promotional content or coordinated campaigns
- Individual complaints without broader pattern

SENTIMENT ANALYSIS NUANCE:
- Background noise: Some level of price complaints/inflation talk is ALWAYS present. Twitter would complain about water prices in a flood.
- Signal: UNUSUAL SPIKES in sentiment (3x+ normal "can't afford" chatter = meaningful consumer pressure)
- The truth is often between Twitter doom ("economy is collapsing") and actual market reality (SPY up 0.2%)
- If everyone's suddenly talking about "too expensive" or "sold out everywhere", that aggregate pattern matters
- Compare sentiment intensity to what you'd expect on a normal day

FACT-CHECKING PROTOCOL (the fun part - catching people in their exaggerations):
When verified market data is provided, you MUST use it to validate claims:
1. Compare tweet claims against the actual price/volume data
2. Flag claims that contradict the verified numbers (e.g., "silver crashing" when data shows +3%)
3. Note when sentiment ALIGNS with real price action - this strengthens the signal
4. "Massive volume" claims should show >2x average in the data; otherwise call out the exaggeration
5. "All-time high" or "52-week high" claims should match the Notes column
6. In your assessment, classify claims as:
   - **Verified:** Claims that match the market data (respect)
   - **Exaggerated:** Directionally correct but overstated (classic fintwit)
   - **False:** Claims that directly contradict the data (someone's farming engagement)
   - **Unverified:** Claims about assets not in the provided data

This is CRITICAL: Do NOT let unverified hype drive your signal strength rating.
If tweets scream "SILVER MOONING!!!" but the data shows +0.5%, that's LOW signal. Call it what it is.

HISTORICAL PARALLELS - USE WITH EXTREME CARE:
"History doesn't repeat itself, but it often rhymes." - Mark Twain
"Past performance is not indicative of future results." - Every compliance department ever

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

TONE GUIDANCE:
Your personality is: actually funny, not trying-to-be-funny. The humor should feel like it slipped out naturally while you were making a real point, not like you workshopped it. Think of yourself as the person who makes one offhand comment in a meeting that makes everyone suppress a laugh.

Rules:
- Lead with substance, season with personality. The ratio is 80% real analysis, 20% flavor.
- NEVER repeat the same joke or phrasing across digests. If you've said "nothing burger" once, it's retired forever. If you've said "fintwit being fintwit", find a new way. The reader gets this twice a day - they will notice repetition instantly and it kills the charm.
- Vary your energy. Some days be drier. Some days be a little more animated. Match the mood of the actual news. A genuinely wild day should read differently than a boring one.
- You can roast exaggerated claims, institutional absurdity, and fintwit's main character syndrome. Don't punch down at regular people dealing with real price pain.
- When something is actually interesting, let yourself be interested. Drop the ironic distance for a second. "Ok wait, this one's actually worth paying attention to" hits harder than permanent sarcasm.
- Don't start every section the same way. Don't end every section the same way. Vary your sentence structure.
- NO catchphrases. NO signature sign-offs you use every time. Each digest should feel like a different day because it IS a different day.

Examples of good tone (DO NOT copy these verbatim - they're just vibes):
- "Gold's up 2% and Twitter is acting like they personally discovered it underground with a pickaxe."
- "Three separate people called this a 'generational buying opportunity.' It's a Tuesday."
- "The egg thing is real, by the way. Not just Twitter being dramatic for once."
- "Quiet day. The most exciting thing on fintwit was someone discovering what the yield curve is."
- "This one actually has some teeth to it. Supply numbers don't lie the way people do."

- BlackRock/Aladdin shade: Check the CURRENT CONTEXT section for whether to include shade this run. If it says "Shade: yes", work ONE jab into the analysis where it's contextually relevant. Make it land. Don't force it. Examples of the KIND of energy (don't copy these):
  - "Larry Fink is probably seeing this same signal between phone calls with Jay Powell about his $25 million investment portfolio. Small world."
  - "The institutional guys are probably still waiting for their compliance team to approve reading this tweet."
  - "Somewhere a BlackRock analyst is writing this same take, but it won't clear compliance until the trade is already crowded."
  - "The Dutch pension funds saw this coming, which is why they pulled $5.9 billion from BlackRock. When the Dutch think you're too greedy, you've achieved something special."

WRITING STYLE - SOUND LIKE A PERSON, NOT AN AI:
You are writing something a human will read twice a day. If it reads like it was generated, they'll tune it out. These are the patterns that make text scream "a robot wrote this" - avoid all of them:

1. **Kill the AI transition words.** "Additionally", "Furthermore", "Moreover", "Notably", "It's worth noting" - these are dead giveaways. Use normal human connectors: "also", "and", "turns out", "the other thing is", "on top of that", "meanwhile". Or just start the next sentence. You don't need a transition for every thought.

2. **Stop inflating significance.** "Pivotal", "crucial", "underscores", "highlights the importance of", "represents a shift" - you're writing a market digest, not a TED Talk intro. If oil is up 3%, say "oil's up 3%." Don't say it "underscores the evolving dynamics of the energy landscape."

3. **No -ing tacking.** Don't end sentences with participial phrases that add fake depth: "...highlighting the broader trend", "...underscoring market uncertainty", "...reflecting investor sentiment." If the point matters, give it its own sentence. If it doesn't, cut it.

4. **Use "is" and "are."** Say "Gold is up 2%" not "Gold serves as today's standout performer." Say "The Fed's statement was boring" not "The Fed's statement stands as a reminder of institutional caution." Simple verbs. Always.

5. **Break the rule of three.** AI loves listing exactly three things. "Innovation, growth, and opportunity." "Speed, efficiency, and reliability." If you catch yourself listing three adjectives or three parallel phrases, break the pattern. Sometimes there are two things. Sometimes four. Sometimes one.

6. **Go easy on em dashes.** One per digest, max. Use commas or periods instead. Em dashes are the cargo shorts of punctuation - technically functional but a tell that you're not trying.

7. **Don't hedge everything.** "It could potentially be argued that markets might see some pressure" - no. "Markets might see pressure" or better, "Markets will probably sell off." Have a take. Hedge only when genuine uncertainty exists, not as a verbal tic.

8. **Vary your sentence length.** If every sentence is 15-20 words with the same structure, it sounds algorithmic. Mix short punches with longer observations. Let some sentences breathe. Chop others short.

9. **No generic conclusions.** Never end with "time will tell", "remains to be seen", "only time will tell if...", "the coming weeks will be crucial." Either make a specific prediction or just stop writing. Ending on substance is better than ending on a platitude.

10. **Don't bold everything.** Bold is for section headers in the report format, not for emphasizing random words mid-sentence. When everything is bold, nothing is.

These rules protect your personality, not strip it. Jokes, opinions, and attitude are human. "Additionally, it's worth noting that silver underscores broader commodity dynamics" is not.

YOUR OUTPUT CALIBRATION:
1. **Signal Strength**: Rate today as HIGH / MEDIUM / LOW / NONE
   - HIGH: Genuinely unusual activity, potential market-moving (rare - maybe 1-2x per month)
   - MEDIUM: Interesting developments worth monitoring (weekly occurrence)
   - LOW: Normal market chatter, nothing actionable (most days)
   - NONE: Below-average Twitter activity, no unusual signals detected (Twitter took a collective nap). Note: You still provide the news roundup even at NONE.

2. **If signal is LOW or NONE**: Say so clearly and with appropriate energy. "Another day, another round of normal market chatter. Nothing here that should change anyone's thesis." is a VALID and GOOD response. You don't need to manufacture excitement.

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

Remember: The reader is smart and busy. They don't need hand-holding or the same recycled commentary they got yesterday. They need honest signal assessment, actual context, and - when you can pull it off naturally - the kind of take that makes them smirk at their phone."""


def format_tweets_for_llm(trend_tweets: dict[str, list[ScrapedTweet]]) -> str:
    """
    Format collected tweets into a structured prompt for the LLM.

    Args:
        trend_tweets: Dictionary mapping trends to their tweets.

    Returns:
        Formatted string for LLM analysis.
    """
    parts = ["# Twitter/X Economic Analysis\n"]
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
        temporal_context: str = "",
        top_engagement: float = 0,
        fact_checker: MarketFactChecker | None = None,
        memory: MemoryManager | None = None,
        temporal_analyzer: TemporalTrendAnalyzer | None = None,
        timelines: dict[str, TrendTimeline] | None = None,
        twitter_mentions: list[str] | None = None,
        previous_digest_summary: str = "",
) -> tuple[str, str, bool, int, str | None]:
    """
    Use the LLM to generate a CALIBRATED sentiment analysis using an agentic loop.

    Returns:
        Tuple of (analysis_content, signal_strength, is_notable, tokens_used, subject_line)
    """
    logger.info(f"Generating analysis with {llm.provider_name} ({llm.model_name})")

    # Initialize generic tools
    tools_registry = ToolRegistry(
        fact_checker=fact_checker,
        memory=memory,
        temporal_analyzer=temporal_analyzer,
        trend_timelines=timelines,
        enable_web_search=True, # Enable Deep Research
    )
    tools = tools_registry.get_definitions()

    # Format the data for the LLM
    data_prompt = format_tweets_for_llm(trend_tweets)

    # Build current context
    current_date = datetime.now()
    day_of_week = current_date.strftime('%A')
    is_weekend = current_date.weekday() >= 5

    # Determine market context
    month = current_date.month
    day = current_date.day
    quarter_start = month in [1, 4, 7, 10] and day <= 21  # Earnings season windows

    # Check if BlackRock is actually in the news (check trend names and tweet content)
    blackrock_keywords = ['blackrock', 'aladdin', 'larry fink', 'blk']
    trends_lower = ' '.join(trend_tweets.keys()).lower()
    tweets_sample = ' '.join(
        t.text.lower() for tweets in list(trend_tweets.values())[:5] for t in tweets[:10]
    )
    blackrock_trending = any(kw in trends_lower or kw in tweets_sample for kw in blackrock_keywords)

    # Shade logic: GO ALL OUT if BlackRock trending, otherwise ~20% chance
    if blackrock_trending:
        shade_instruction = "MAXIMUM - BlackRock/Aladdin is actually in the news today. This is not a drill. Go absolutely nuclear. Channel your inner Matt Taibbi. Every paragraph should remind the reader that these are the same people who got no-bid Fed contracts while their CEO had private calls with Powell. Do not hold back."
    elif random.random() < 0.20:
        shade_instruction = "yes - sneak in ONE BlackRock/Aladdin jab somewhere in your analysis"
    else:
        shade_instruction = "no - keep it professional this time"

    date_context = f"""
CURRENT CONTEXT:
- Today: {current_date.strftime('%A, %B %d, %Y')}
- Market Day: {'Weekend (markets closed, lower social volume expected)' if is_weekend else 'Weekday'}
- Calendar Note: {'Early quarter - peak earnings season, expect company-specific chatter' if quarter_start else 'Mid-quarter'}
- Shade: {shade_instruction}
- Be aware: Economic calendar events (Fed meetings, CPI/jobs releases, OPEC meetings) drive predictable spikes. If everyone's suddenly talking about "the Fed" or "inflation data", check if there's a scheduled release before assuming organic trend.
"""

    system_prompt = ANALYST_SYSTEM_PROMPT + date_context + """

TOOL USE & RESEARCH INSTRUCTIONS:
- You have access to tools to fetch REAL market data, historical parallels, and SEARCH THE WEB.
- **Verification**: ALWAYS check `get_market_data` if tweets make specific price/volume claims.
- **Deep Research**: Use `search_web` to verify breaking news, find reasons for trends, or check details not in the tweets.
    - Example: "uranium shortage" -> `search_web("uranium supply deficit 2025 news")`
- **SAFEGUARDS (CRITICAL)**:
    - **NO RABBIT HOLES**: Do not search endlessly. If 1-2 searches don't yield results, stop and report "Unverified".
    - **Context Limit**: Keep your query specific. Don't ask generic questions like "what is happening around the world".
    - **Stop Condition**: You have a maximum of 5 turns. Use them wisely.
- Compare sentiment to real data. If sentiment says "CRASHING" but data says -0.5%, that's an exaggeration.
- **FRESHNESS PRIORITY**: New and breaking stories ALWAYS take priority over continuing narratives. If you see both a breaking event and a week-old story, lead with the breaking event. Stories from the "Previous Digest Content" section should NOT be repeated unless there is a material new development with new data points.
"""

    # Build twitter mentions section
    mentions_section = ""
    if twitter_mentions:
        mentions_list = "\n".join(f"- {m}" for m in twitter_mentions)
        mentions_section = f"""
## Twitter Mentions (Not Signal-Worthy)
These topics were being discussed on Twitter but did NOT pass the signal filter. Briefly mention what Twitter is chattering about, even if not actionable:
{mentions_list}
"""

    user_prompt = f"""Analyze the following data. Be SKEPTICAL about Twitter - most days are boring.

{previous_digest_summary}

{mentions_section}

{historical_context}

{temporal_context}

## Today's Twitter Data
Top engagement score: {top_engagement:.0f}
{data_prompt}

## How to Submit Your Analysis

When you are done analyzing, you MUST call the `submit_report` tool with these fields:

- **subject_line**: A punchy email subject (5-10 words max). MUST reference the actual top trend or finding. Should feel like a text from a friend, not a Bloomberg alert. Be creative - NEVER reuse a subject line format you've used before. Examples of the ENERGY (don't copy these):
  - "Silver's Up and Fintwit Found Religion"
  - "NVIDIA Pricing GPUs Like They're Selling Kidneys"
  - "Genuinely Nothing Happened Today, You're Welcome"
  - "Egg Prices Did That Thing Again"
  - "Fed Said Words, Markets Pretended to Care"

- **signal_strength**: One of "high", "medium", "low", "none". HIGH should be rare - most days are LOW or NONE.

- **assessment**: 2-3 sentences. Be real about what happened (or didn't). If LOW/NONE, say it differently every time - don't fall back on the same "nothing burger" or "fintwit being fintwit" template. Find a fresh way to say "slow day" that matches today's specific vibe.

- **trends_observed**: Bullet points of what's being discussed - factual, not hyped. ALWAYS use the '•' character for bullets (not '-' or '*'). Example:
  • Gold rallying on inflation fears
  • Silver following with momentum
  • Tech sector rotation underway

- **fact_check**: Categorized fact-check results. Group claims by category, each on its own line with '•' bullets and bold Title Case labels:
  • **Verified:** [claims matching market data]
  • **Exaggerated:** [directionally correct but overstated]
  • **False:** [claims contradicting data]
  • **Unverified:** [claims without data to verify]
  Leave empty string if you didn't fetch market data.

- **actionability**: One of "not actionable", "monitor only", "worth researching", "warrants attention"

- **actionability_reason**: 1 sentence explaining the actionability rating.

- **historical_parallel**: If meaningful: "History rhymes: [parallel]". Otherwise: "No meaningful historical parallels."

- **bottom_line**: 1 sentence. The one thing you'd actually text a friend. Not a platitude. Not "stay safe out there" or "save your attention" every time. Something specific to TODAY.

If Twitter mentions are provided (topics that didn't pass the signal filter), briefly note what Twitter is chattering about at the end of your assessment. One sentence is enough - e.g., "Twitter was also buzzing about [topics] but nothing actionable there."

Remember: Your job is to FILTER, not to HYPE. Anyone can scream about markets. The real flex is knowing when to say "nah." Be the friend who saves people from checking their portfolio for no reason."""

    messages = [{"role": "user", "content": user_prompt}]
    
    total_tokens = 0
    max_turns = 5
    
    for turn in range(max_turns):
        try:
            logger.info(f"LLM Agent Turn {turn + 1}/{max_turns}")
            response = await llm.generate(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2000,
                tools=tools,
            )
            
            total_tokens += response.token_count
            
            # Append assistant response to history
            assistant_msg = {"role": "assistant", "content": response.content}
            # Preserve raw_content for Google's thought_signature support
            if hasattr(response, 'raw_content') and response.raw_content is not None:
                assistant_msg["raw_content"] = response.raw_content
            messages.append(assistant_msg)

            # Handle tool calls
            if response.tool_calls:
                # Add tool calls to messages (OpenAI requires this if we want to reply with tool outputs)
                if hasattr(response, 'tool_calls') and response.tool_calls:
                     messages[-1]["tool_calls"] = response.tool_calls

                for tool_call in response.tool_calls:
                     # Parse function call
                     function_name = tool_call.function.name
                     arguments = {}
                     try:
                         import json
                         arguments = json.loads(tool_call.function.arguments)
                     except Exception:
                         pass

                     call_id = tool_call.id

                     # Handle submit_report specially - this is the final output
                     if function_name == "submit_report":
                         subject_line = arguments.get("subject_line") or "Jafar Market Digest"
                         signal_strength = (arguments.get("signal_strength") or "low").lower()
                         is_notable = signal_strength == "high"

                         # Extract structured sections (handle None values)
                         # Apply sanitization to fix common LLM formatting issues
                         assessment = sanitize_llm_output(arguments.get("assessment") or "")
                         trends_observed = sanitize_llm_output(arguments.get("trends_observed") or "")
                         fact_check = sanitize_llm_output(arguments.get("fact_check") or "")
                         actionability = arguments.get("actionability") or ""
                         actionability_reason = sanitize_llm_output(arguments.get("actionability_reason") or "")
                         historical_parallel = sanitize_llm_output(arguments.get("historical_parallel") or "")
                         bottom_line = sanitize_llm_output(arguments.get("bottom_line") or "")

                         # Format body with Title Case headers
                         body_parts = []
                         if assessment:
                             body_parts.append(f"**Assessment:**\n{assessment}")
                         if trends_observed:
                             body_parts.append(f"**Trends Observed:**\n{trends_observed}")
                         if fact_check:
                             body_parts.append(f"**Fact Check:**\n{fact_check}")
                         if actionability:
                             actionability_line = f"**Actionability:** {actionability.title()}"
                             if actionability_reason:
                                 actionability_line += f"\n{actionability_reason}"
                             body_parts.append(actionability_line)
                         if historical_parallel:
                             body_parts.append(f"**Historical Parallel:**\n{historical_parallel}")
                         if bottom_line:
                             body_parts.append(f"**Bottom Line:**\n{bottom_line}")

                         body = "\n\n".join(body_parts) if body_parts else "No analysis provided."

                         logger.info(f"Report submitted - Signal: {signal_strength.upper()}, Subject: {subject_line}")
                         return body, signal_strength, is_notable, total_tokens, subject_line

                     # Execute other tools
                     tool_output = await tools_registry.execute(function_name, arguments)

                     # Append tool output to messages
                     messages.append({
                         "role": "tool",
                         "content": tool_output,
                         "tool_call_id": call_id,
                         "name": function_name
                     })

                # Continue loop to let LLM process tool outputs
                continue

            # If no tool calls and no submit_report, the LLM gave a text response
            # This is a fallback - ideally the LLM should always use submit_report
            content = response.content
            logger.warning("LLM returned text instead of calling submit_report - using fallback parsing")

            # Fallback: try to parse from text (legacy format)
            # Handle both UPPERCASE and Title Case variants
            subject_line = None
            subject_match = re.search(r'\*\*(?:SUBJECT LINE|Subject Line)\*\*:\s*["\']?([^"\'\n]+)["\']?', content, re.IGNORECASE)
            if subject_match:
                subject_line = subject_match.group(1).strip().strip('"\'')

            signal_strength = "low"
            is_notable = False
            content_upper = content.upper()
            if "HIGH" in content_upper and "SIGNAL" in content_upper:
                signal_strength = "high"
                is_notable = True
            elif "MEDIUM" in content_upper and "SIGNAL" in content_upper:
                signal_strength = "medium"
            elif "NONE" in content_upper and "SIGNAL" in content_upper:
                signal_strength = "none"

            # Strip metadata lines from body content (case-insensitive)
            content = re.sub(r'\*\*(?:SUBJECT LINE|Subject Line)\*\*:\s*[^\n]*\n*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\*\*(?:SIGNAL STRENGTH|Signal Strength)\*\*:\s*[^\n]*\n*', '', content, flags=re.IGNORECASE)
            content = content.strip()

            logger.info(f"Fallback parse - Signal: {signal_strength.upper()}, Subject: {subject_line}")
            return content, signal_strength, is_notable, total_tokens, subject_line
            
        except Exception as e:
            logger.error(f"LLM agent loop failed: {e}")
            raise

    # If loop exhausted without final answer (should be rare)
    return "Analysis incomplete due to step limit.", "low", False, total_tokens, None


def _get_filter_tool_definition() -> dict:
    """Get the tool definition for the trend filter."""
    return {
        "type": "function",
        "function": {
            "name": "submit_filter_result",
            "description": "Submit the list of trends to keep for deeper analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "trends_to_keep": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of trend terms to keep (e.g., ['$NVDA', 'Release', 'Layoffs']). Use exact term names from the candidates.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why these trends were kept and others rejected (1-2 sentences)",
                    },
                },
                "required": ["trends_to_keep", "reasoning"],
            },
        },
    }


FILTER_SYSTEM_PROMPT = """You are a context-aware trend filter for a financial intelligence system.

Your job: Read the sample tweets for each trend candidate and decide which represent REAL, SIGNIFICANT developments worth analyzing deeper.

KEEP trends where sample tweets show:
- Specific market-moving events (earnings, product launches, policy changes)
- Major news (document releases, legal developments, geopolitical events)
- AI/Tech developments (model releases, company announcements, chip news)
- Economic indicators or shifts (inflation data, jobs, consumer behavior)
- Supply/demand signals (shortages, price changes, availability issues)
- Company-specific news (layoffs, acquisitions, scandals, product issues)

REJECT trends where sample tweets show:
- Generic discussion with no specific news hook
- Entertainment/pop culture unrelated to markets
- Pure social media noise or memes
- Recycled/stale narratives with no new information

CRITICAL: Use the sample tweets to understand context. A term like "Release" could be:
- KEEP: If tweets discuss Epstein files, AI model releases, earnings releases
- REJECT: If tweets are just generic chatter about music/movies

When ready, call the submit_filter_result tool with your decision."""


async def llm_filter_trends(
        llm: LLMProvider,
        candidates: list[DiscoveredTrend],
) -> list[str]:
    """
    Use LLM with tool calling to filter trend candidates WITH CONTEXT.

    This passes sample tweets for each trend so the LLM can understand what
    "Release" actually refers to (Epstein files? Sonnet 5? Product launch?).

    Args:
        llm: The LLM provider to use
        candidates: List of DiscoveredTrend objects with sample tweets

    Returns:
        Filtered list of trend term strings worth deep-diving
    """
    if not candidates:
        return []

    # Format candidates WITH their sample tweets for context
    candidates_parts = []
    for trend in candidates:
        # Include 2-3 sample tweets so LLM can see what the trend actually refers to
        samples = trend.sample_tweets[:3] if trend.sample_tweets else []
        samples_text = "\n    ".join(f'"{s[:150]}..."' if len(s) > 150 else f'"{s}"' for s in samples)

        if samples_text:
            candidates_parts.append(f"- **{trend.term}** ({trend.mention_count} mentions, {trend.unique_authors} authors)\n    Sample tweets:\n    {samples_text}")
        else:
            candidates_parts.append(f"- **{trend.term}** ({trend.mention_count} mentions, {trend.unique_authors} authors)")

    candidates_str = "\n\n".join(candidates_parts)

    prompt = f"""Review these trend candidates. The sample tweets show what people are ACTUALLY discussing.

CANDIDATES:
{candidates_str}

Analyze each trend's sample tweets to understand the real context, then call submit_filter_result with your decision."""

    tools = [_get_filter_tool_definition()]

    try:
        response = await llm.generate(
            prompt=prompt,
            system_prompt=FILTER_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=500,
            tools=tools,
        )

        # Extract result from tool call
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "submit_filter_result":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    filtered = args.get("trends_to_keep", [])
                    reasoning = args.get("reasoning", "")

                    if reasoning:
                        logger.info(f"LLM filter reasoning: {reasoning}")

                    logger.info(f"LLM filter: {len(filtered)}/{len(candidates)} trends passed")
                    return filtered

        # Fallback: try to parse from text if no tool call (shouldn't happen)
        logger.warning("LLM filter did not use tool call, falling back to all candidates")
        return [t.term for t in candidates]

    except Exception as e:
        logger.warning(f"LLM filter failed, using all candidates: {e}")
        return [t.term for t in candidates]


async def _get_account_tweets(username: str, limit: int = 20) -> list[dict]:
    """Get recent tweets for an account from the database.

    Returns a list of dicts suitable for the BotJudge.judge_account API.
    """
    from .database import get_session
    from .models import Tweet
    from sqlalchemy import select

    session = await get_session()
    async with session:
        result = await session.execute(
            select(Tweet)
            .where(Tweet.username == username)
            .order_by(Tweet.created_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

    return [
        {
            "content": row.content,
            "created_at": str(row.created_at) if row.created_at else "unknown",
            "likes": row.likes,
            "retweets": row.retweets,
            "replies": row.replies,
            "views": row.views,
        }
        for row in rows
    ]


async def _get_bot_usernames() -> set[str]:
    """Get usernames that should be filtered from analysis.

    Uses three data-driven sources (no arbitrary score thresholds):
    - BlockedAccount table: explicitly blocked spam accounts
    - HumanLabel == 'bot': HITL-labeled bots
    - AccountScore.is_anomaly == True: IsolationForest statistical outliers
    """
    from .database import get_session
    from .models import AccountScore, BlockedAccount, HumanLabel
    from sqlalchemy import select, union

    session = await get_session()
    try:
        blocked_q = select(BlockedAccount.username)
        labeled_q = select(HumanLabel.username).where(HumanLabel.label.in_(["bot", "garbage"]))
        anomaly_q = select(AccountScore.username).where(AccountScore.is_anomaly.is_(True))

        combined = union(blocked_q, labeled_q, anomaly_q)
        result = await session.execute(combined)
        return {row[0] for row in result.all()}
    finally:
        await session.close()


async def run_ml_analysis(config, diagnostics, run_id: str) -> dict:
    """Run ML analysis on stored tweets. Returns ML insights for the report."""
    from .ml import BotScorer, AccountScorer

    logger.info("=" * 60)
    logger.info("STEP: ML ANALYSIS")
    logger.info("=" * 60)

    ml_results = {
        "bot_suspects": [],
        "top_accounts": [],
        "ml_evaluation": None,
    }

    bot_scores = []
    suspects = []

    try:
        # 1. Bot Detection
        bot_scorer = BotScorer()
        training_stats = await bot_scorer.train(min_tweets_per_account=5)
        logger.info(f"Bot detection trained on {training_stats.get('accounts_eligible', 0)} accounts")

        if bot_scorer.model is not None:
            bot_scores = await bot_scorer.score_all_accounts(min_tweets=5)
            # Get top suspects (score > 0.7)
            suspects = [s for s in bot_scores if s["bot_score"] > 0.7]
            ml_results["bot_suspects"] = suspects[:10]  # Top 10
            logger.info(f"Found {len(suspects)} suspected bot accounts")
            diagnostics.bot_suspects_found = len(suspects)

        # 2. Account Signal Scoring
        account_scorer = AccountScorer()
        all_account_results = await account_scorer.analyze(min_tweets_per_account=3)
        ml_results["top_accounts"] = all_account_results[:20]
        ml_results["total_accounts_analyzed"] = len(all_account_results)

        # Also include cluster distribution
        from collections import Counter
        cluster_counts = Counter(a.get("cluster_label", "unclustered") for a in all_account_results)
        ml_results["cluster_distribution"] = dict(cluster_counts)

        top_accounts = ml_results["top_accounts"]
        logger.info(f"Scored {len(all_account_results)} accounts for signal quality")
        diagnostics.accounts_scored = len(top_accounts)

        # Fetch sample tweets for top signal accounts
        for account in top_accounts[:10]:
            tweets = await _get_account_tweets(account["username"], limit=5)
            account["sample_tweets"] = tweets

        # 3. LLM Bot Judging (on top suspects only to save API calls)
        if suspects:
            from .ml.llm_judge import BotJudge

            judge = BotJudge(
                api_key=config.google.api_key,
                model=config.google.model,
            )

            # Judge top 10 suspects — fetch tweets from DB for each
            judgments = []
            for suspect in suspects[:10]:
                tweets = await _get_account_tweets(suspect["username"], limit=20)
                judgment = await judge.judge_account(
                    username=suspect["username"],
                    tweets=tweets,
                    ml_features=suspect.get("features"),
                )
                judgments.append(judgment)

            ml_results["bot_judgments"] = judgments

            # Evaluate ML model against LLM judgments if we have enough data
            if len(bot_scores) >= 10:
                # Attach tweets to bot_scores for the evaluator
                scores_with_tweets = []
                for s in bot_scores:
                    tweets = await _get_account_tweets(s["username"], limit=20)
                    scores_with_tweets.append({**s, "tweets": tweets})

                evaluation = await judge.evaluate_ml_model(
                    scores_with_tweets, sample_size=min(20, len(scores_with_tweets))
                )
                ml_results["ml_evaluation"] = evaluation
                logger.info(
                    f"ML evaluation - Agreement: {evaluation.get('agreement_rate', 0):.1%}, "
                    f"F1: {evaluation.get('f1_score', 0):.2f}"
                )
                diagnostics.ml_agreement_rate = evaluation.get("agreement_rate", 0)

        # Attach sample tweets to bot suspects for the ML email
        for suspect in ml_results["bot_suspects"]:
            tweets = await _get_account_tweets(suspect["username"], limit=5)
            suspect["sample_tweets"] = tweets

        # 4. Total tweets analyzed
        from .database import get_session
        from .models import Tweet
        from sqlalchemy import func, select

        session = await get_session()
        try:
            result = await session.execute(select(func.count()).select_from(Tweet))
            ml_results["total_tweets_analyzed"] = result.scalar_one()
        finally:
            await session.close()

    except Exception as e:
        logger.warning(f"ML analysis encountered an error: {e}", exc_info=True)

    return ml_results


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

    # Initialize database FIRST — checkpoint lives in Postgres
    await init_db(config.database.url)
    await create_tables()

    # Initialize checkpoint manager with advisory lock
    checkpoint = CheckpointManager()
    await checkpoint.initialize()

    # Initialize diagnostics collector
    diagnostics = DiagnosticsCollector(
        run_id=checkpoint.run_id if checkpoint.run_id else datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Check for existing checkpoint to resume
    resuming = await checkpoint.should_resume()
    if resuming:
        state = checkpoint.get_state()
        logger.info(f"Resuming from checkpoint: {state.run_id}")
        logger.info(
            f"  - Scout (broad scraping): {'DONE' if state.step1_complete else f'{len(state.topics_completed)}/{len(config.app.broad_topics)} topics'}")
        logger.info(f"  - Investigator (trends): {'DONE' if state.step2_complete else 'PENDING'}")
        logger.info(f"  - Deep Dive: {'DONE' if state.step3_complete else f'{len(state.trends_completed)}/{len(state.trends)} trends'}")
        logger.info(f"  - Analyst (LLM): {'DONE' if state.step4_complete else 'PENDING'}")
        logger.info(f"  - Reporter (email): {'DONE' if state.step5_complete else 'PENDING'}")
        logger.info(f"  - History: {'DONE' if state.step6_complete else 'PENDING'}")
    else:
        logger.info("Starting fresh pipeline run")
        await checkpoint.start_new_run()
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
    if config.database.memory_enabled:
        try:
            logger.info("Initializing vector memory system...")
            memory = await create_memory_manager(
                google_api_key=config.google.api_key,
                embedding_model=config.google.embedding_model,
                embedding_dimensions=config.google.embedding_dimensions,
            )
            memory_count = await memory.vector_store.count()
            logger.info(f"Vector memory initialized with {memory_count} stored memories")
        except Exception as e:
            logger.warning(f"Failed to initialize vector memory: {e}")
            memory = None

    try:
        llm = create_llm_provider(
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
        email_from_name=config.smtp.email_from_name,
        email_to=config.smtp.email_to,
    )

    try:
        # ============================================================
        # STEP 1: SCOUT - Broad Twitter Search (with checkpointing)
        # ============================================================
        step1_start = time.time()
        diagnostics.diagnostics.broad_topics_attempted = len(config.app.broad_topics)

        if not state.step1_complete:
            logger.info("\n[STEP 1/11] THE SCOUT: Gathering broad economic tweets...")
            logger.info(f"Topics: {len(config.app.broad_topics) - len(state.topics_completed)} remaining, {len(state.topics_completed)} completed")

            stored = await scraper.get_broad_tweets(
                topics=config.app.broad_topics,
                limit_per_topic=config.app.broad_tweet_limit,
                skip_topics=state.topics_completed,
                pipeline_run=state.run_id,
                on_topic_complete=lambda topic, tweets: checkpoint.mark_topic_done(topic),
            )
            diagnostics.diagnostics.tweets_stored += stored

            await checkpoint.complete_step1()
            state = checkpoint.get_state()
            diagnostics.diagnostics.broad_topics_completed = len(state.topics_completed)
        else:
            logger.info("\n[STEP 1/11] Skipping (already complete)")
            diagnostics.diagnostics.broad_topics_completed = len(config.app.broad_topics)

        # Load all broad tweets from DB (works for both fresh and resumed runs)
        broad_tweets = await load_broad_tweets_from_db(state.run_id, state.topics_completed)
        diagnostics.diagnostics.broad_tweets_scraped = len(broad_tweets)
        diagnostics.diagnostics.time_step1_scraping = time.time() - step1_start

        if not broad_tweets:
            logger.error(f"No tweets retrieved for run_id={state.run_id}. Check twscrape setup or DB.")
            diagnostics.diagnostics.add_error("No tweets retrieved from broad scraping")
            return False

        # Filter out tweets from known bot accounts (ML scores + HITL labels)
        bot_usernames = await _get_bot_usernames()
        if bot_usernames:
            pre_filter = len(broad_tweets)
            broad_tweets = [t for t in broad_tweets if t.username not in bot_usernames]
            filtered_count = pre_filter - len(broad_tweets)
            if filtered_count:
                logger.info(f"Filtered {filtered_count} tweets from {len(bot_usernames)} known bot accounts")

        logger.info(f"Total broad tweets: {len(broad_tweets)}")

        # ============================================================
        # STEP 1b: TWITTER TRENDS - Fetch trending topics
        # ============================================================
        logger.info("\n[STEP 1b/11] TWITTER TRENDS: Fetching trending topics...")
        twitter_api = await scraper._get_api()
        twitter_trends_raw = await fetch_trending_topics(twitter_api)
        diagnostics.diagnostics.twitter_trends_fetched = len(twitter_trends_raw)

        if twitter_trends_raw:
            logger.info(f"Twitter trending topics ({len(twitter_trends_raw)}): {twitter_trends_raw[:10]}")
            if len(twitter_trends_raw) > 10:
                logger.info(f"  ... and {len(twitter_trends_raw) - 10} more")
        else:
            logger.info("No Twitter trending topics fetched (API may be unavailable)")

        # ============================================================
        # STEP 1c: WATCHED ACCOUNTS - Scrape high-signal sources
        # ============================================================
        try:
            from .database import get_session as _get_sess
            from .models import WatchedAccount
            from sqlalchemy import select as _sel
            watched_session = await _get_sess()
            try:
                watched_rows = (await watched_session.execute(
                    _sel(WatchedAccount)
                )).scalars().all()
            finally:
                await watched_session.close()

            if watched_rows:
                logger.info(f"\n[STEP 1c/11] WATCHED ACCOUNTS: Scraping {len(watched_rows)} high-signal sources...")
                from .scraper import store_tweets as _store_tweets
                for watched in watched_rows:
                    query = f"from:{watched.username}"
                    tweets = await scraper.search_tweets(query, limit=20)
                    if tweets:
                        # Only store tweets we haven't seen before
                        if watched.last_scraped_tweet_id:
                            tweets = [t for t in tweets if t.id > watched.last_scraped_tweet_id]

                        if tweets:
                            stored = await _store_tweets(
                                tweets, source_query=f"watched:{watched.username}",
                                pipeline_run=state.run_id,
                            )
                            logger.info(f"  @{watched.username}: {stored} new tweets stored")

                            # Update tracking
                            newest_id = max(t.id for t in tweets)
                            ws = await _get_sess()
                            async with ws.begin():
                                w = await ws.get(WatchedAccount, watched.username)
                                if w:
                                    w.last_scraped_at = datetime.now()
                                    w.last_scraped_tweet_id = newest_id
                            await ws.close()
                        else:
                            logger.info(f"  @{watched.username}: no new tweets")
                    else:
                        logger.info(f"  @{watched.username}: no tweets found")
            else:
                logger.info("\n[STEP 1c/11] No watched accounts configured")
        except Exception as e:
            logger.warning(f"Watched account scraping failed (non-fatal): {e}")

        # ============================================================
        # STEP 2: INVESTIGATOR - NER Analysis
        # ============================================================
        step2_start = time.time()
        trend_objects: list[DiscoveredTrend] = []  # Keep full objects for LLM filter context

        if not state.step2_complete:
            logger.info("\n[STEP 2/11] THE INVESTIGATOR: Extracting trending entities...")

            # Use extract_trends_with_details to get full DiscoveredTrend objects
            # These include sample_tweets which give the LLM filter context
            trend_objects, _ = analyzer.extract_trends_with_details(
                tweets=broad_tweets,
                top_n=config.app.top_trends_count,
                min_mentions=config.app.min_trend_mentions,
                min_authors=config.app.min_trend_authors,
                apply_quality_filter=True,
            )
            trends = [t.term for t in trend_objects]

            if not trends:
                logger.warning("No trends extracted. Using fallback topics.")
                diagnostics.diagnostics.add_warning("No trends extracted, using fallback topics")
                trends = ["Federal Reserve", "Stock Market", "Inflation"]
                trend_objects = []  # No objects for fallback

            diagnostics.diagnostics.trends_discovered = len(trends)
            await checkpoint.save_trends(trends)
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 2/11] Skipping (already complete)")
            trends = state.trends
            diagnostics.diagnostics.trends_discovered = len(trends)
            # Note: trend_objects will be empty for resumed runs, LLM filter will be skipped

        diagnostics.diagnostics.time_step2_analysis = time.time() - step2_start

        # Merge Twitter trending topics into discovered trends (non-keyword discovery)
        if twitter_trends_raw:
            existing_lower = {t.lower() for t in trends}
            new_from_twitter = []
            for tt in twitter_trends_raw:
                if tt.lower() not in existing_lower:
                    new_from_twitter.append(tt)
                    existing_lower.add(tt.lower())
            if new_from_twitter:
                logger.info(f"Adding {len(new_from_twitter)} Twitter trending topics not found by NER: {new_from_twitter}")
                trends.extend(new_from_twitter)
                await checkpoint.save_trends(trends)
                state = checkpoint.get_state()
            else:
                logger.info("All Twitter trending topics already covered by NER discovery")

        logger.info(f"Trends (pre-filter): {trends}")

        # ============================================================
        # STEP 3: LLM QUALITY FILTER - Validate candidates before deep dive
        # ============================================================
        # Only run LLM filter if we have trend objects with context (not resumed)
        # and haven't started deep dive yet
        twitter_mentions: list[str] = []  # Rejected trends for "mentions" tier
        pre_filter_trends = list(trends)  # Save pre-filter list

        if trend_objects and not state.step3_complete:
            logger.info("\n[STEP 3/11] LLM FILTER: Validating trend candidates with context...")
            # Pass full DiscoveredTrend objects so LLM can see sample tweets
            filtered_trends = await llm_filter_trends(llm, trend_objects)
            diagnostics.diagnostics.llm_calls_made += 1  # LLM filter call

            if set(filtered_trends) != set(trends):
                # Capture rejected trends as "mentions" tier
                twitter_mentions = [t for t in pre_filter_trends if t not in filtered_trends]
                if twitter_mentions:
                    logger.info(f"Twitter mentions (rejected but notable): {twitter_mentions}")

                # Save filtered trends to checkpoint
                trends = filtered_trends
                await checkpoint.save_trends(trends)
                state = checkpoint.get_state()

            if not trends:
                logger.info("LLM filter rejected all candidates - quiet day")
                diagnostics.diagnostics.add_warning("LLM filter rejected all trend candidates")

        diagnostics.diagnostics.trends_filtered_by_llm = len(trends) if trends else 0
        logger.info(f"Trends (post-filter): {trends}")

        # ============================================================
        # STEP 4: DEEP DIVE - Targeted Scraping
        # ============================================================
        step3_start = time.time()
        diagnostics.diagnostics.deep_dive_trends_attempted = len(trends) if trends else 0

        if not state.step3_complete:
            logger.info("\n[STEP 4/11] THE DEEP DIVE: Gathering sentiment for each trend...")

            stored = await scraper.get_specific_sentiment(
                trends=trends,
                limit_per_trend=config.app.specific_tweet_limit,
                skip_trends=state.trends_completed,
                pipeline_run=state.run_id,
                on_trend_complete=lambda trend, tweets: checkpoint.mark_trend_done(trend),
            )
            diagnostics.diagnostics.tweets_stored += stored

            await checkpoint.complete_step3()
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 4/11] Skipping (already complete)")

        # Load all trend tweets from DB (works for both fresh and resumed runs)
        trend_tweets = await load_trend_tweets_from_db(state.run_id, trends)

        # Filter bot accounts from trend tweets too
        if bot_usernames:
            for trend_key in trend_tweets:
                pre = len(trend_tweets[trend_key])
                trend_tweets[trend_key] = [
                    t for t in trend_tweets[trend_key]
                    if t.username not in bot_usernames
                ]
                diff = pre - len(trend_tweets[trend_key])
                if diff:
                    logger.debug(f"Filtered {diff} bot tweets from trend '{trend_key}'")

        diagnostics.diagnostics.deep_dive_trends_completed = len([t for t in trends if trend_tweets.get(t)])
        diagnostics.diagnostics.deep_dive_tweets_scraped = sum(len(v) for v in trend_tweets.values())
        diagnostics.diagnostics.time_step3_deep_dive = time.time() - step3_start

        total_tweets = sum(len(t) for t in trend_tweets.values())
        logger.info(f"Total trend tweets: {total_tweets}")

        # ============================================================
        # STEP 5: FACT CHECKER - Init for LLM tool use
        # ============================================================
        fact_checker = None
        if config.fact_checker.enabled:
            logger.info("\n[STEP 6/11] THE FACT CHECKER: Initializing for LLM tool use...")
            fact_checker = MarketFactChecker(
                cache_ttl_minutes=config.fact_checker.cache_ttl_minutes,
                price_tolerance_pct=config.fact_checker.price_tolerance_pct,
            )

        # ============================================================
        # STEP 7: TEMPORAL ANALYSIS - Track trend continuity
        # ============================================================
        logger.info("\n[STEP 7/11] TEMPORAL ANALYSIS: Analyzing trend timelines...")
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
        timelines = await temporal_analyzer.analyze_all_trends(trend_details_for_temporal)

        # Count temporal patterns (new, continuing, recurring)
        temporal_patterns = sum(1 for t in timelines.values() if t.is_new or t.is_continuing or t.is_recurring)
        diagnostics.diagnostics.temporal_patterns_detected = temporal_patterns

        # Format temporal context for LLM
        temporal_context = temporal_analyzer.format_context_for_llm(timelines)

        # Suppress stale trends — demote to mentions tier
        suppressed_trends = temporal_analyzer.get_suppressed_trends(timelines)
        if suppressed_trends:
            logger.info(f"Suppressing {len(suppressed_trends)} stale trends: {suppressed_trends}")
            if twitter_mentions is None:
                twitter_mentions = []
            for term in suppressed_trends:
                if term in trend_tweets:
                    twitter_mentions.append(
                        f"{term} (Day {timelines[term].consecutive_days}, "
                        f"engagement {timelines[term].trend_velocity})"
                    )
                    del trend_tweets[term]

        # ============================================================
        # STEP 8: ANALYST - LLM Summary
        # ============================================================
        step4_start = time.time()
        if not state.step4_complete:
            logger.info("\n[STEP 8/11] THE ANALYST: Generating calibrated analysis...")

            historical_context = await history.format_context_for_llm(days=7)
            previous_digest_summary = await history.get_previous_digest_summary(days_back=2)
            baseline = await history.get_baseline_stats(days=30)

            top_engagement = 0.0
            for tweets_list in trend_tweets.values():
                for tweet in tweets_list:
                    eng = (tweet.likes * 1.0) + (tweet.retweets * 0.5) + (tweet.replies * 0.3)
                    top_engagement = max(top_engagement, eng)

            logger.info(f"Top engagement: {top_engagement:.0f} (avg: {baseline.get('avg_top_engagement', 0):.0f})")

            analysis, signal_strength, is_notable, tokens_used, subject_line = await analyze_with_llm(
                llm,
                trend_tweets,
                historical_context=historical_context,
                temporal_context=temporal_context,
                top_engagement=top_engagement,
                fact_checker=fact_checker,
                memory=memory,
                temporal_analyzer=temporal_analyzer,
                timelines=timelines,
                twitter_mentions=twitter_mentions if twitter_mentions else None,
                previous_digest_summary=previous_digest_summary,
            )

            diagnostics.diagnostics.llm_calls_made += 1  # Main analysis call
            diagnostics.diagnostics.llm_tokens_used += tokens_used

            if not analysis:
                logger.error("LLM analysis returned empty result")
                diagnostics.diagnostics.add_error("LLM analysis returned empty result")
                return False

            await checkpoint.save_analysis(analysis, signal_strength, is_notable, top_engagement)
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 8/11] Skipping (already complete)")
            analysis = state.analysis
            signal_strength = state.signal_strength
            is_notable = state.is_notable
            top_engagement = state.top_engagement
            subject_line = None  # Will use fallback in reporter

        diagnostics.diagnostics.time_step4_llm = time.time() - step4_start
        diagnostics.diagnostics.signal_strength = signal_strength
        diagnostics.diagnostics.notable = is_notable

        logger.info(f"Signal: {signal_strength.upper()}, Notable: {is_notable}")

        # ============================================================
        # STEP 9: REPORTER - Email Digest
        # ============================================================
        step5_start = time.time()
        if not state.step5_complete:
            logger.info("\n[STEP 9/11] THE REPORTER: Sending email digest...")

            provider_info = f"{llm.provider_name} {llm.model_name}"

            success = reporter.send_email(
                report_content=analysis,
                trends=trends,
                tweet_count=total_tweets,
                provider_info=provider_info,
                signal_strength=signal_strength,
                timelines=timelines,
                subject_line=subject_line,
            )

            diagnostics.diagnostics.email_sent = success

            if success:
                logger.info("Email sent successfully!")
            else:
                logger.warning("Failed to send email")
                diagnostics.diagnostics.add_error("Failed to send digest email")

            await checkpoint.complete_step5()
            state = checkpoint.get_state()
        else:
            logger.info("\n[STEP 9/11] Skipping (already complete)")
            diagnostics.diagnostics.email_sent = True  # Assume it was sent in previous run

        diagnostics.diagnostics.time_step5_email = time.time() - step5_start

        # ============================================================
        # STEP 10: STORE HISTORY
        # ============================================================
        step6_start = time.time()
        if not state.step6_complete:
            logger.info("\n[STEP 10/11] Storing digest in history...")

            await history.store_digest(
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

            await checkpoint.complete_step6()
        else:
            logger.info("\n[STEP 10/11] Skipping (already complete)")

        diagnostics.diagnostics.time_step6_storage = time.time() - step6_start

        # ============================================================
        # COMPLETE - Clear checkpoint
        # ============================================================
        await checkpoint.clear()

        # ============================================================
        # STEP 11: ADMIN DIAGNOSTICS - Send admin email if needed
        # ============================================================
        if config.smtp.admin.enabled:
            logger.info("\n[STEP 11/11] ADMIN DIAGNOSTICS: Checking if alert needed...")

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
        logger.info("\nInterrupted by user — progress saved to database")
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
        await checkpoint.set_error(str(e))
        logger.exception(f"Pipeline failed: {e}")
        logger.info("Progress saved to database — run again to resume")

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
        await checkpoint.close()
        await scraper.close()
        if memory:
            await memory.close()
        await close_db()


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

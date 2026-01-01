## Project Overview

**Jafar** - The villain to BlackRock's Aladdin.

Twitter/X sentiment analysis system for discovering emerging market trends before they hit mainstream news. Uses statistical NLP analysis, real-time market data fact-checking, vector memory for historical parallels, and LLM-powered summarization.

**Key Philosophy**:
- Most days are boring. The system is calibrated to identify when something *actually* matters, not manufacture urgency.
- **True organic discovery**: Broad topics are maximally general ("markets", "economy", "breaking") - NOT keyword lists. Statistical analysis + LLM filtering find what's trending, not pre-specified search terms.

## Quick Commands

```bash
# Install dependencies
uv sync

# Download spaCy model (required once)
uv run python -m spacy download en_core_web_sm

# Run the pipeline (interactive)
uv run main.py

# Run in background (recommended for VPS)
./run.sh start       # Start pipeline
./run.sh logs        # Watch live progress
./run.sh status      # Check if running
./run.sh stop        # Stop pipeline

# Check Twitter account status
uv run twscrape accounts

# Add account via cookies (preferred method)
uv run add_account.py <username> cookies.json
```

## Architecture

```
src/
├── main.py             # Pipeline orchestration (6.75 steps)
├── config.py           # Hybrid YAML + env config
├── scraper.py          # Twitter scraping via twscrape
├── analyzer.py         # Statistical trend discovery (spaCy NLP)
├── reporter.py         # HTML email reports (SMTP) + admin diagnostics
├── history.py          # SQLite digest history + trend timelines
├── checkpoint.py       # Pipeline state persistence
├── fact_checker.py     # Market data verification (yfinance)
├── temporal_analyzer.py # Multi-day trend tracking & continuity detection
├── diagnostics.py      # Run statistics, error tracking, log rotation
├── llm/
│   ├── base.py         # Abstract LLMProvider interface
│   ├── factory.py      # create_llm_provider()
│   ├── openai_client.py
│   └── google_client.py # Uses google-genai SDK
└── memory/
    ├── base.py          # VectorStore interface, MemoryRecord
    ├── embeddings.py    # OpenAI/local embeddings (with dimension control)
    ├── chroma_store.py  # Local vector storage
    ├── pgvector_store.py # Production PostgreSQL
    └── memory_manager.py # Semantic search orchestration

systemd/
├── jafar.service        # systemd service definition
├── jafar-morning.timer  # Morning run timer (7am-12pm random)
├── jafar-evening.timer  # Evening run timer (5pm-11pm random)
└── jafar.timer          # Alternative: single daily run (8am-8pm random)
```

## Pipeline Steps

1. **Broad Discovery** - Scrape 30+ financial topics to find what's trending
2. **Statistical Analysis** - Extract entities via spaCy, score by engagement velocity + cashtag co-occurrence
2.5. **Quality Filter** - Three-stage filtering (statistical → quality threshold → LLM validation)
3. **Deep Dive** - Targeted scraping for validated trends only
3.5. **Fact Checker** - Fetch real market data from Yahoo Finance to verify claims
3.75. **Temporal Analysis** - Track trend continuity (consecutive days, gaps, recurring themes)
4. **LLM Analysis** - Generate skeptical summary with signal strength rating, cross-referencing fact-check data + temporal context
5. **Email Report** - Send HTML digest with trend badges (Day 3, Last seen 6mo ago, New)
6. **Memory Storage** - Store in SQLite + vector DB for future parallels
6.5. **Admin Diagnostics** - Send optional diagnostics email with run statistics and error alerts

## Configuration

**Hybrid approach**: `config.yaml` for settings, `.env` for secrets.

### config.yaml - Key Settings
```yaml
llm:
  provider: openai  # or "google"

scraping:
  broad_tweet_limit: 200
  top_trends_count: 10

memory:
  enabled: true
  store_type: chroma          # or "pgvector"
  openai_embedding_model: text-embedding-3-large
  embedding_dimensions: 1536  # REQUIRED for pgvector (2000 dim limit)

fact_checker:
  enabled: true
  cache_ttl_minutes: 5

temporal:
  consecutive_threshold: 3    # Days to flag as "developing story"
  gap_threshold_days: 14      # Gap to consider "new episode" vs continuation

email:
  admin:
    enabled: true             # Admin diagnostics emails
    send_on_success: false    # Email even on successful runs
    recipients: []            # Defaults to main recipients if empty
    log_retention_count: 10   # Keep N most recent log files

twitter:
  proxies:
    # - socks5://user:pass@host:port  # Per-account assignment
```

### .env - Secrets Only
```bash
OPENAI_API_KEY=sk-...
SMTP_USERNAME=...
SMTP_PASSWORD=...
POSTGRES_URL=postgresql://...  # For pgvector
```

## Key Concepts

### Organic Discovery Philosophy

**The system does NOT use keyword matching.** It casts a wide net with general queries, then uses statistical analysis to find what's actually trending.

**How it works:**
1. **Broad Topics are GENERAL** - "markets", "economy", "breaking", "shortage" - NOT "chip shortage" or "NVIDIA H200"
2. **Statistical extraction** - spaCy NLP extracts entities, n-grams, cashtags from ALL tweets
3. **Engagement scoring** - Ranks by velocity (likes + retweets + replies) + author diversity
4. **Financial context ratio** - % of mentions appearing alongside financial terms (65%+ threshold)
5. **Cashtag co-occurrence** - % of mentions appearing with $TICKER symbols (proves financial relevance)
6. **LLM validation** - Final filter keeps actionable signals, rejects noise

**Example:**
- Search query: "shortage" (general)
- Finds: Tweets about GPU shortage, housing shortage, oil shortage, labor shortage
- Statistical analysis: "GPU shortage" has 50 mentions, 15 authors, 10,000 engagement, 70% financial context, 60% cashtag co-occurrence ($NVDA)
- Quality filter: ✅ Passes (65%+ financial context, diverse authors)
- LLM filter: ✅ Keeps "GPU shortage" (supply chain signal), rejects "housing shortage" (out of scope for fintwit)
- Deep dive: Scrapes 50 more tweets about "GPU shortage" specifically
- Result: Discovers OpenAI H200 order organically, without ever searching for "H200"

**Why this beats keyword lists:**
- Catches emerging signals you didn't know to search for
- Adapts to what the market is actually discussing
- No maintenance - doesn't need updates when new products/companies/issues emerge
- True sentiment analysis - sees aggregate mood shifts

### Trend Scoring
```python
engagement = (likes * 1.0) + (retweets * 0.5) + (replies * 0.3)
score = (mentions * 0.3) + (engagement_weighted * 0.7)
# Bonus for author diversity (organic vs spam)
# Bonus for cashtag co-occurrence (terms appearing alongside $TICKERS)
```

### Trend Quality Filtering (Three-Stage)

The system uses a funnel approach to avoid deep-diving on noise:

```
top_trends_count: 20  →  Quality Threshold  →  LLM Filter  →  Deep Dive
  (cast VERY wide)          (8-12 pass)         (3-5 best)     (focused)
```

**Stage 1: Statistical Filter**
- Minimum 4 mentions, 3 unique authors
- Engagement velocity scoring (likes + retweets + replies)
- Author diversity bonus (organic vs spam)

**Stage 2: Quality Threshold** (`passes_quality_threshold()`)
- Requires 10+ unique authors (proves organic, not one person spamming)
- Requires 65%+ financial context ratio (includes supply chain / infrastructure terms)
- For non-cashtags: requires >10% cashtag co-occurrence (proves financial relevance)

**Stage 3: LLM Pre-Filter** (~500 tokens, fast)
- Asks LLM: "Which are actionable signals OR meaningful sentiment indicators OR supply/demand shifts?"
- Keeps: Silver, $NVDA, Uranium, Inventories, Risk, Demand, Bearish, GPU Shortage, H200
- Rejects: Christmas, Books, Fiction, Crowd, Chat

**Philosophy**:
- This is *sentiment* analysis. Aggregate mood matters. If "Risk" or "Bearish" is spiking, that's signal even if you can't trade it directly.
- **65% threshold** allows hardware/infrastructure trends (which mix technical + financial language) vs 85% which only passed pure commodity/equity chatter
- **Supply chain signals matter** - "shortage", "allocation", "orders" drive future pricing power shifts
- The truth is often between Twitter doom and actual market reality.

**Why `top_trends_count: 20` is optimal**: Cast VERY wide statistically (since broad topics are now general), let quality filter narrow to 8-12, let LLM pick the best 3-5 for deep dive. With general broad topics, you need more statistical candidates to ensure signal isn't lost.

### Signal Strength
- **HIGH**: Genuinely unusual, potential market-moving (1-2x/month)
- **MEDIUM**: Notable developments worth monitoring (weekly)
- **LOW**: Normal market chatter (most days)
- **NONE**: Below-average activity (quiet days)

### Fact Checking
The LLM classifies tweet claims against real market data:
- **VERIFIED**: Claims match data
- **EXAGGERATED**: Directionally correct but overstated
- **FALSE**: Claims contradict data
- **UNVERIFIABLE**: Asset not in data

### Checkpointing
Pipeline saves state to `checkpoint.json` after each topic/trend. Automatically resumes on restart if interrupted.

### Proxy Support
Proxies in `config.yaml` are assigned round-robin to accounts when running `add_account.py`. Each account gets a consistent proxy stored in `accounts.db`.

## Symbol Mappings (fact_checker.py)

```python
# Commodities
"gold" -> "GC=F", "silver" -> "SI=F", "oil" -> "CL=F"

# Crypto
"bitcoin" -> "BTC-USD", "ethereum" -> "ETH-USD"

# Indices
"spy" -> "SPY", "nasdaq" -> "^IXIC", "vix" -> "^VIX"
```

Cashtags ($AAPL) are looked up directly.

## Common Issues

### Script hanging during scraping
If the pipeline hangs during Twitter scraping (no log output for 5+ minutes), the search query may have timed out. The script now has timeout protection (default: 120 seconds per query).

To adjust the timeout:
```yaml
scraping:
  search_timeout: 180  # Increase to 180 seconds if needed
```

If hangs persist:
- Check proxy connectivity (if using proxies)
- Verify Twitter accounts are still active: `uv run twscrape accounts`
- Reduce `broad_tweet_limit` to fetch fewer tweets per topic

### "column cannot have more than 2000 dimensions"
Using pgvector with text-embedding-3-large. Add to config.yaml:
```yaml
memory:
  embedding_dimensions: 1536
```
Then: `psql "$POSTGRES_URL" -c "DROP TABLE IF EXISTS market_memories;"`

### "No account available for queue"
Twitter accounts not logged in. Check with `uv run twscrape accounts`, re-add via cookie auth.

### Rate limiting
Add more Twitter accounts. Pipeline checkpoints progress so you can resume.

## Code Style Notes

- Async/await throughout for Twitter scraping and market data fetching
- Dataclasses for data structures (ScrapedTweet, PipelineState, MemoryRecord, MarketDataPoint)
- Abstract base classes for swappable providers (LLMProvider, VectorStore, EmbeddingService)
- Logging via `logging.getLogger("jafar.*")`
- Type hints throughout

## Files to Ignore

- `accounts.db` - Twitter credentials (twscrape)
- `digest_history.db` - Local digest history
- `checkpoint.json` - Pipeline state
- `memory_store/` - ChromaDB vector storage
- `pipeline.log` - Background runner logs
- `pipeline.pid` - Background runner PID
- `.env` - Secrets
- `config.yaml` - Local config (has example)

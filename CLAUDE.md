# CLAUDE.md

## Project Overview

**Jafar** - The villain to BlackRock's Aladdin.

Twitter/X sentiment analysis system for discovering emerging market trends before they hit mainstream news. Uses statistical NLP analysis, real-time market data fact-checking, vector memory for historical parallels, and LLM-powered summarization.

**Key Philosophy**: Most days are boring. The system is calibrated to identify when something *actually* matters, not manufacture urgency.

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
├── main.py          # Pipeline orchestration (6.5 steps)
├── config.py        # Hybrid YAML + env config
├── scraper.py       # Twitter scraping via twscrape
├── analyzer.py      # Statistical trend discovery (spaCy NLP)
├── reporter.py      # HTML email reports (SMTP)
├── history.py       # SQLite digest history
├── checkpoint.py    # Pipeline state persistence
├── fact_checker.py  # Market data verification (yfinance)
├── llm/
│   ├── base.py      # Abstract LLMProvider interface
│   ├── factory.py   # create_llm_provider()
│   ├── openai_client.py
│   └── google_client.py  # Uses google-genai SDK
└── memory/
    ├── base.py           # VectorStore interface, MemoryRecord
    ├── embeddings.py     # OpenAI/local embeddings (with dimension control)
    ├── chroma_store.py   # Local vector storage
    ├── pgvector_store.py # Production PostgreSQL
    └── memory_manager.py # Semantic search orchestration
```

## Pipeline Steps

1. **Broad Discovery** - Scrape 30+ financial topics to find what's trending
2. **Statistical Analysis** - Extract entities via spaCy, score by engagement velocity + cashtag co-occurrence
2.5. **Quality Filter** - Three-stage filtering (statistical → quality threshold → LLM validation)
3. **Deep Dive** - Targeted scraping for validated trends only
3.5. **Fact Checker** - Fetch real market data from Yahoo Finance to verify claims
4. **LLM Analysis** - Generate skeptical summary with signal strength rating, cross-referencing fact-check data
5. **Email Report** - Send HTML digest with trend badges
6. **Memory Storage** - Store in SQLite + vector DB for future parallels

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
top_trends_count: 10  →  Quality Threshold  →  LLM Filter  →  Deep Dive
     (cast wide)            (5-7 pass)         (2-4 best)     (focused)
```

**Stage 1: Statistical Filter**
- Minimum mentions and unique authors
- Financial context ratio (% of tweets with market terms)
- Cashtag co-occurrence (% appearing alongside $TICKER symbols)

**Stage 2: Quality Threshold** (`passes_quality_threshold()`)
- Requires 10+ unique authors
- Requires 85%+ financial context ratio
- For non-cashtags: requires >10% cashtag co-occurrence

**Stage 3: LLM Pre-Filter** (~500 tokens, fast)
- Asks LLM: "Which of these are actionable market signals vs noise?"
- Keeps: Silver, $NVDA, Uranium, Inventories
- Rejects: Risk, Demand, Buyers, Gap, Books

**Why `top_trends_count: 10` is optimal**: Cast wide statistically, let the LLM pick the 2-4 best. Setting it to 5 might miss something that ranked #7 statistically but is actually the most actionable signal.

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

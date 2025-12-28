# CLAUDE.md

## Project Overview

Twitter/X sentiment analysis system for discovering emerging market trends before they hit mainstream news. Uses statistical NLP analysis, vector memory for historical parallels, and LLM-powered summarization.

**Key Philosophy**: Most days are boring. The system is calibrated to identify when something *actually* matters, not manufacture urgency.

## Quick Commands

```bash
# Install dependencies
uv sync

# Download spaCy model (required once)
uv run python -m spacy download en_core_web_sm

# Run the pipeline
uv run python main.py

# Check Twitter account status
uv run twscrape accounts

# Login Twitter accounts
uv run twscrape login_accounts

# Add account via cookies (preferred method)
uv run python add_account.py <username> cookies.json
```

## Architecture

```
src/
├── main.py          # Pipeline orchestration (6 steps)
├── config.py        # Environment config (Settings, MemoryConfig)
├── scraper.py       # Twitter scraping via twscrape
├── analyzer.py      # Statistical trend discovery (spaCy NLP)
├── reporter.py      # HTML email reports (SMTP)
├── history.py       # SQLite digest history
├── checkpoint.py    # Pipeline state persistence
├── llm/
│   ├── base.py      # Abstract LLMProvider interface
│   ├── factory.py   # create_llm_provider()
│   ├── openai_client.py
│   └── google_client.py  # Uses google-genai SDK
└── memory/
    ├── base.py           # VectorStore interface, MemoryRecord
    ├── embeddings.py     # OpenAI/local embeddings
    ├── chroma_store.py   # Local vector storage
    ├── pgvector_store.py # Production PostgreSQL
    └── memory_manager.py # Semantic search orchestration
```

## Pipeline Steps

1. **Broad Discovery** - Scrape 25+ financial topics to find what's trending
2. **Statistical Analysis** - Extract entities via spaCy, score by engagement velocity
3. **Deep Dive** - Targeted scraping for top discovered trends
4. **LLM Analysis** - Generate skeptical summary with signal strength rating
5. **Email Report** - Send HTML digest with trend badges
6. **Memory Storage** - Store in SQLite + vector DB for future parallels

## Key Concepts

### Trend Scoring
```python
engagement = (likes * 1.0) + (retweets * 0.5) + (replies * 0.3)
score = (mentions * 0.3) + (engagement_weighted * 0.7)
# Bonus for author diversity (organic vs spam)
```

### Signal Strength
- **HIGH**: Genuinely unusual, potential market-moving (1-2x/month)
- **MEDIUM**: Notable developments worth monitoring (weekly)
- **LOW**: Normal market chatter (most days)
- **NONE**: Below-average activity (quiet days)

### Checkpointing
Pipeline saves state to `pipeline_checkpoint.json` after each topic/trend. Automatically resumes on restart if interrupted by rate limits or Ctrl+C.

## Configuration

All config via `.env` file. Key settings:

```bash
LLM_PROVIDER=openai          # or "google"
OPENAI_API_KEY=sk-...
MEMORY_ENABLED=true
MEMORY_STORE_TYPE=chroma     # or "pgvector"
BROAD_TWEET_LIMIT=200        # tweets per broad topic
TOP_TRENDS_COUNT=10          # trends to analyze
```

## Common Issues

### "No account available for queue"
Twitter accounts not logged in. Run `uv run twscrape accounts` to check, then `uv run twscrape login_accounts` or use cookie auth.

### Rate limiting
Add more Twitter accounts to the pool. twscrape rotates automatically. Pipeline checkpoints progress so you can resume after adding accounts.

### Import errors
Run `uv sync` to ensure all dependencies installed.

## Code Style Notes

- Async/await throughout for Twitter scraping
- Dataclasses for data structures (ScrapedTweet, PipelineState, MemoryRecord)
- Abstract base classes for swappable providers (LLMProvider, VectorStore, EmbeddingService)
- Logging via `logging.getLogger("twitter_sentiment.*")`

## Files to Ignore

- `accounts.db` - Twitter credentials (twscrape)
- `digest_history.db` - Local digest history
- `pipeline_checkpoint.json` - Pipeline state
- `memory_store/` - ChromaDB vector storage
- `.env` - Secrets

# Jafar

**The villain to BlackRock's Aladdin.**

Jafar is a **Consumer Economy Scout** that discovers emerging economic signals from organic social discussion, not just
financial Twitter. It uses a **skeptical agentic loop** to filter noise, verify claims with real-time data, and find
historical rhymes.

## Core Philosophy

1. **Consumer Economy First**: An RTX 5090 price hike is a better inflation signal than a CPI print. We track shortages,
   pricing power, and spending shifts.
2. **Agentic "Pull" Architecture**: We don't dump data into the prompt. The LLM acts as an analyst: it sees a claim,
   decides if it needs verification, and **calls tools** to fetch prices or search the web.
3. **Calibrated Skepticism**: Most days are boring. If Jafar creates urgency every day, it's failing. "Nothing to
   report" is a successful output.
4. **No Keyword Lists**: Broad topics + statistical NLP discovery. We find "H200 shortages" by searching for "shortage",
   not by hardcoding "H200".

## Stack

- **LLM**: Google Gemini (sole provider, via `google-genai` SDK)
- **Embeddings**: Gemini Embedding 2 Preview (3072 dimensions)
- **Database**: PostgreSQL with VectorChord (`vector` + `vchord` extensions) for vector search
- **ORM**: SQLAlchemy 2.0 async (with `asyncpg` driver)
- **ML**: scikit-learn (IsolationForest, KMeans, StandardScaler)
- **NLP**: spaCy (`en_core_web_lg`)
- **Twitter**: twscrape (user's fork)
- **Web Search**: DuckDuckGo (`ddgs`)
- **Market Data**: yfinance

**Not used**: No OpenAI, no ChromaDB, no SQLite (except twscrape's internal `accounts.db`).

## Quick Start

```bash
# 1. Install Dependencies (includes spaCy model)
uv sync

# 2. Copy config files
cp config.example.yaml config.yaml
cp .env.example .env
# Edit both files with your settings

# 3. Add Twitter Account
uv run add_account.py <username> cookies.json

# 4. Run Pipeline
uv run main.py

# Background Service (Linux/Mac)
./run.sh start | logs | status | stop
```

## Architecture

The system runs an **11-step Discovery -> Agentic Analysis** pipeline:

1. **Broad Scrape** (Step 1): Scrapes 30+ generic topics ("too expensive", "sold out", "hiring") across Twitter using
   twscrape.
2. **Twitter Trending Topics** (Step 1b): Fetches platform trending topics for non-keyword discovery.
3. **Tweet Persistence**: All scraped tweets stored to PostgreSQL (`tweets` table) for ML training data.
4. **NER Analysis** (Step 2): Extracts trending entities using spaCy NER, scores by engagement velocity and author
   diversity.
5. **LLM Quality Filter** (Step 3): Gemini reviews trend candidates WITH sample tweets for context, rejecting noise (
   entertainment, memes, generic chatter).
6. **Deep Dive Scraping** (Step 4): Targeted scraping of validated trends + persistence to Postgres.
7. **Fact Checking & Temporal Analysis** (Steps 5-6): Market data verification via yfinance, multi-day trend tracking.
8. **Agentic Analysis** (Step 7): LLM analyst with tool calling (up to 5 turns). Self-corrects claims against real
   market data, searches the web for context, finds historical parallels via vector search.
9. **ML Analysis** (Step 8): Bot detection, account signal scoring, LLM-as-judge validation.
10. **Calibrated HTML Digest** (Step 9): Email report with signal strength rating, fact checks, historical parallels.
11. **History & Diagnostics** (Steps 10-11): Store digest to vector memory for future parallels. Admin diagnostics email
    with run statistics.

The pipeline supports **checkpointing** -- progress is saved after each step and can resume after interruption.

## Available Tools

The LLM agent has access to the following tools in `src/tools.py`:

* `get_market_data(symbols)`: Real-time prices via yfinance. Verifies price/volume claims from tweets.
* `search_web(query)`: Deep research via DuckDuckGo text search.
* `fetch_news(query)`: News headlines via DuckDuckGo news search.
* `search_historical_parallels(query)`: Semantic vector search over past digests (VectorChord).
* `get_trend_timeline(trend)`: Checks if a trend is new, recurring, or developing (multi-day tracking).
* `get_weather_forecast(cities)`: Current conditions and 7-day forecast via Open-Meteo (free, no API key).
* `submit_report(...)`: Structured output tool for finalizing analysis (subject line, signal strength, assessment, fact
  checks, etc.).

## Database Models

Key tables in PostgreSQL (defined in `src/models.py`):

| Table            | Purpose                                                                        |
|------------------|--------------------------------------------------------------------------------|
| `digests`        | Daily digest reports with signal strength and trend details                    |
| `trend_history`  | Trend mention counts and engagement over time (temporal tracking)              |
| `tweets`         | Scraped tweets for ML training data (bot detection, account scoring)           |
| `memory_records` | Digest memories with vector embeddings for semantic search (VectorChord)       |
| `bot_judgments`  | LLM bot judgments for building labeled datasets                                |
| `app_state`      | Key-value store for pipeline state (e.g., last ML diagnostics email timestamp) |

## ML System

The ML pipeline (`src/ml/`) runs on every pipeline execution:

* **Bot Detection** (`BotScorer`): IsolationForest anomaly detection trained on tweet behavioral features (posting
  frequency, engagement ratios, content patterns). Flags accounts with bot score > 0.7.
* **Account Signal Scoring** (`AccountScorer`): Weighted composite scoring + KMeans clustering to rank accounts by
  signal quality. Identifies the most valuable signal sources.
* **LLM-as-Judge** (`BotJudge`): Gemini validates ML bot classifications by reviewing actual tweet content. Provides
  reasoning and confidence scores.
* **Self-Evaluation**: Measures ML vs LLM agreement rate and F1 score to track model quality over time.

ML diagnostics are included in the email digest periodically (configurable via `ml.diagnostics_interval_days`).

## Testing

This project has comprehensive test coverage. **Always run tests before committing changes.**

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run fast tests only (skip slow/integration)
uv run pytest tests/ -m "not slow"
```

**When making changes:**

- Run the full test suite to catch regressions
- Update existing tests if behavior changes intentionally
- Add new tests for new functionality
- Tests are in `tests/unit/` - check there first for examples

The test suite uses mocks for external dependencies (Twitter API, LLM providers, yfinance, etc.) so tests run fast and
don't require API keys.

## Configuration

* **`config.yaml`**: App settings (LLM model, scraping limits, thresholds, broad topics, email settings). Copy from
  `config.example.yaml`.
* **`.env`**: Secrets (API keys, SMTP credentials, database URL). Copy from `.env.example`.

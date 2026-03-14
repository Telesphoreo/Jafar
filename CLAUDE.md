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
- **Dashboard**: Flask (sync SQLAlchemy with psycopg2)
- **ML**: scikit-learn (IsolationForest, KMeans, StandardScaler), scipy (percentile ranking)
- **NLP**: spaCy (`en_core_web_lg`)
- **Twitter**: twscrape (user's fork)
- **Web Search**: DuckDuckGo (`ddgs`)
- **Market Data**: yfinance

**Not used**: No ChromaDB, no SQLite (except twscrape's internal `accounts.db`).

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

# 5. ML Dashboard
uv run dashboard

# Background Service (Linux/Mac)
./run.sh start | logs | status | stop
```

## Architecture

The system runs a **Discovery -> Agentic Analysis** pipeline:

1. **Broad Scrape** (Step 1): Scrapes 30+ generic topics ("too expensive", "sold out", "hiring") across Twitter using
   twscrape.
2. **Twitter Trending Topics** (Step 1b): Fetches platform trending topics for non-keyword discovery.
3. **Watched Account Scraping** (Step 1c): Scrapes recent tweets from high-signal accounts marked via the dashboard.
4. **Tweet Persistence & Filtering**: All scraped tweets stored to PostgreSQL (`tweets` table). Tweets from
   blocked/garbage accounts are filtered out before analysis.
5. **NER Analysis** (Step 2): Extracts trending entities using spaCy NER, scores by engagement velocity and author
   diversity.
6. **LLM Quality Filter** (Step 3): Gemini reviews trend candidates WITH sample tweets for context, rejecting noise (
   entertainment, memes, generic chatter).
7. **Deep Dive Scraping** (Step 4): Targeted scraping of validated trends + persistence to Postgres.
8. **Fact Checking & Temporal Analysis** (Steps 5-6): Market data verification via yfinance, multi-day trend tracking.
9. **Agentic Analysis** (Step 7): LLM analyst with tool calling (up to 5 turns). Self-corrects claims against real
   market data, searches the web for context, finds historical parallels via vector search.
10. **Calibrated HTML Digest** (Step 8): Email report with signal strength rating, fact checks, historical parallels.
11. **History & Diagnostics** (Steps 9-10): Store digest to vector memory for future parallels. Admin diagnostics email
    with run statistics.

The pipeline supports **checkpointing** -- state is persisted to PostgreSQL (`pipeline_runs` table) with advisory
locking for concurrency control. Progress is saved after each step and can resume after interruption.

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

| Table              | Purpose                                                                    |
|--------------------|----------------------------------------------------------------------------|
| `digests`          | Daily digest reports with signal strength and trend details                |
| `trend_history`    | Trend mention counts and engagement over time (temporal tracking)          |
| `tweets`           | Scraped tweets for ML training data and account scoring                    |
| `memory_records`   | Digest memories with vector embeddings for semantic search (VectorChord)   |
| `signal_judgments`  | LLM signal/garbage judgments for account quality classification            |
| `pipeline_runs`    | Pipeline checkpoint state with advisory locking for concurrency control    |
| `account_scores`   | ML scores (garbage_score, signal_score, is_anomaly, cluster, features)     |
| `watched_accounts` | High-signal accounts to actively scrape each pipeline run                  |
| `blocked_accounts` | Spam accounts whose tweets are permanently discarded                       |
| `human_labels`     | HITL signal/garbage/unsure labels from dashboard review                    |

## ML System

ML is **not** part of the automated pipeline. It is controlled from the Flask dashboard (`uv run dashboard`).

* **Garbage Detection** (`BotScorer`): IsolationForest anomaly detection trained on tweet behavioral features (posting
  frequency, engagement ratios, content patterns). Uses the anomaly flag directly -- no arbitrary score thresholds.
* **Account Signal Scoring** (`AccountScorer`): Weighted composite scoring + KMeans clustering to rank accounts by
  signal quality. Uses percentile ranking (scipy `rankdata`), not min-max normalization.
* **LLM-as-Judge** (`BotJudge`): Gemini classifies accounts as signal or garbage by reviewing actual tweet content.
  Auto-blocks garbage at >=80% confidence, auto-labels signal at >=80% confidence. Uncertain accounts go to the
  HITL review queue.
* **Self-Evaluation**: Measures ML vs LLM agreement rate and F1 score to track model quality over time.

The pipeline reads ML results (blocked accounts, watched accounts) but does not run ML itself.

## ML Dashboard

`uv run dashboard` starts the Flask dashboard on port 5000. Pages:

* **Overview**: Stats summary, ML controls (run scoring, run LLM judge), recent pipeline runs.
* **Accounts**: Sortable/filterable account list with ML scores and human labels.
* **Account Detail**: Individual account tweets, ML scores, LLM judgment, watch/block/label actions.
* **Review Queue**: LLM-judged accounts needing human decision (signal/garbage/unsure).
* **Search**: Find accounts by username.
* **Pipeline Runs**: Detailed run history with diagnostics.

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
* **`.env`**: Secrets (API keys, SMTP credentials, database URL). Copy from `.env.example`. Also used by the dashboard.

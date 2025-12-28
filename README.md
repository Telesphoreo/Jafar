# Twitter/X Economic Sentiment Analysis

A sophisticated system for discovering emerging market trends from Twitter/X before they hit mainstream news. Uses statistical analysis, NLP, and LLM-powered summarization with historical memory for finding genuine parallels.

**Key Philosophy**: Most days are boring. This system is calibrated to tell you when something *actually* matters, not to manufacture urgency.

## Features

- **Dynamic Discovery**: Finds trends you didn't know to look for (not just keyword matching)
- **Statistical Anomaly Detection**: Surfaces what's *unusually* active, weighted by engagement
- **Aggressive Noise Filtering**: Filters out Fed, Trump, generic market chatter
- **Vector Memory**: Semantic search finds historical parallels ("history rhymes")
- **Skeptical Analysis**: LLM is explicitly instructed to say "nothing notable today" when appropriate
- **Signal Strength Rating**: HIGH / MEDIUM / LOW / NONE calibration
- **Swappable LLM Providers**: OpenAI or Google Gemini
- **Professional Email Digests**: HTML-formatted with signal strength indicators

## Architecture

```
twitter_sentiment_analysis/
├── main.py                 # Entry point
├── pyproject.toml          # Dependencies (uv)
├── .env.example            # Configuration template
└── src/
    ├── config.py           # Environment configuration
    ├── scraper.py          # Twitter scraping (twscrape)
    ├── analyzer.py         # Statistical trend discovery (spaCy)
    ├── reporter.py         # Email reports (SMTP)
    ├── history.py          # SQLite digest history
    ├── main.py             # Pipeline orchestration
    ├── llm/
    │   ├── base.py         # Abstract LLM interface
    │   ├── factory.py      # Provider factory
    │   ├── openai_client.py
    │   └── google_client.py
    └── memory/
        ├── base.py         # Vector store interface
        ├── embeddings.py   # OpenAI/local embeddings
        ├── chroma_store.py # Local vector storage
        ├── pgvector_store.py # Production PostgreSQL
        └── memory_manager.py # Semantic search orchestration
```

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Twitter/X account(s) for scraping
- OpenAI API key (for LLM analysis and embeddings)
- SMTP credentials (for email delivery)

## Installation

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd twitter_sentiment_analysis

# Install dependencies with uv
uv sync

# Download spaCy language model
uv run python -m spacy download en_core_web_sm
```

### 2. Configure Environment

```bash
# Copy the example configuration
cp .env.example .env

# Edit with your credentials
# See Configuration section below for details
```

### 3. Set Up Twitter Account

Twitter scraping requires authenticated accounts. Due to Cloudflare protection, cookie-based authentication is recommended.

#### Method 1: Cookie Authentication (Recommended)

1. Log into Twitter/X in your browser
2. Export cookies using a browser extension (e.g., "Cookie-Editor" for Chrome/Firefox)
3. Save as `cookies.json` in the project root
4. Run the account setup script:

```bash
uv run python add_account.py <your_twitter_username> cookies.json
```

#### Method 2: Direct Login (May be blocked by Cloudflare)

```bash
# Create accounts.txt with format: username:password:email:email_password
echo "myuser:mypass:myemail@example.com:emailpass" > accounts.txt

# Add accounts
uv run twscrape add_accounts accounts.txt username:password:email:email_password

# Login (use --manual if email doesn't support IMAP)
uv run twscrape login_accounts

# Verify accounts are active
uv run twscrape accounts
```

### 4. Run the Pipeline

```bash
uv run python main.py
```

## Configuration

Edit `.env` with your settings:

### Required Settings

```bash
# LLM Provider (choose one)
LLM_PROVIDER=openai              # or "google"
OPENAI_API_KEY=sk-your-key-here  # Required if using OpenAI

# Email (for receiving digests)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_specific_password  # Use app password, not regular password
EMAIL_FROM=your_email@gmail.com
EMAIL_TO=recipient@example.com
```

### Optional Settings

```bash
# Tweet Collection
BROAD_TWEET_LIMIT=200            # Tweets per broad topic (higher = better discovery)
SPECIFIC_TWEET_LIMIT=100         # Tweets per discovered trend
TOP_TRENDS_COUNT=10              # Number of trends to analyze

# Trend Detection Thresholds
MIN_TREND_MENTIONS=3             # Minimum mentions to be considered
MIN_TREND_AUTHORS=2              # Minimum unique authors (filters spam)

# Vector Memory
MEMORY_ENABLED=true              # Enable semantic parallel detection
MEMORY_STORE_TYPE=chroma         # "chroma" (local) or "pgvector" (production)
EMBEDDING_PROVIDER=openai        # "openai" or "local"
MEMORY_MIN_SIMILARITY=0.6        # 0.0-1.0, higher = stricter matching

# Google (if using Google instead of OpenAI)
GOOGLE_API_KEY=your-google-key
GOOGLE_MODEL=gemini-2.0-flash
```

## How It Works

The pipeline runs in 6 steps:

### Step 1: The Scout (Broad Discovery)
Searches Twitter across 25+ broad financial topics:
- General: "fintwit", "trading", "markets"
- Signals: "shortage OR surplus", "unusual volume", "whale alert"
- Physical: "freight", "shipping rates", "warehouse inventory"

### Step 2: The Investigator (Statistical Analysis)
Extracts ALL meaningful terms using spaCy NLP:
- Cashtags ($SLV, $NVDA) - highest signal
- Hashtags (#silversqueeze)
- Noun phrases and named entities
- Commodity/sector keywords

Scores by engagement velocity:
```
score = (mentions * 0.3) + (engagement_weighted_by_author_diversity * 0.7)
```

### Step 3: The Deep Dive (Targeted Collection)
Scrapes additional tweets for each discovered trend.

### Step 4: The Analyst (LLM Summary)
- Retrieves historical parallels via vector similarity search
- Generates calibrated analysis with signal strength rating
- Explicitly allowed to say "nothing notable today"

### Step 5: The Reporter (Email Delivery)
Sends HTML-formatted digest with:
- Signal strength banner (color-coded)
- Trend badges
- Full analysis

### Step 6: Memory Storage
Stores digest in:
- SQLite (quick lookups)
- Vector database (semantic search for future parallels)

## Signal Strength Calibration

| Level | Meaning | Frequency |
|-------|---------|-----------|
| HIGH | Genuinely unusual, potential market-moving | 1-2x per month |
| MEDIUM | Notable developments worth monitoring | Weekly |
| LOW | Normal market chatter | Most days |
| NONE | Below-average activity | Quiet days |

## Historical Parallels

The vector memory system finds semantically similar past days:

> **HISTORICAL PARALLEL**: This silver spike with 45,000+ engagement mirrors March 2024 when physical delivery concerns preceded a 12% price move. However, today's narrative focuses on short squeezes rather than supply chain issues.

Or when there's no meaningful parallel:

> **HISTORICAL PARALLEL**: No meaningful historical parallels identified.

The system is explicitly instructed NOT to force connections.

## Production Deployment

### Using pgvector (PostgreSQL)

For production with many memories:

```bash
# Install PostgreSQL driver
uv sync --extra postgres

# Set environment variables
MEMORY_STORE_TYPE=pgvector
POSTGRES_URL=postgresql://user:password@localhost:5432/sentiment_db
```

Requires PostgreSQL with pgvector extension:
```sql
CREATE EXTENSION vector;
```

### Using Local Embeddings (Free)

To avoid OpenAI embedding costs:

```bash
# Install sentence-transformers
uv sync --extra local-embeddings

# Set environment variable
EMBEDDING_PROVIDER=local
```

## Troubleshooting

### "No tweets retrieved from broad search"
- Verify Twitter accounts are logged in: `uv run twscrape accounts`
- Try cookie-based authentication if direct login fails

### "Failed to load spaCy model"
```bash
uv run python -m spacy download en_core_web_sm
```

### Unicode errors on Windows
The system automatically sanitizes output for Windows console compatibility.

### "chromadb" errors
```bash
uv sync  # Reinstall dependencies
```

### Rate limiting
Add more Twitter accounts to the pool. twscrape automatically rotates between them.

## Customization

### Adding Search Topics

Edit `src/config.py` `broad_topics` list:

```python
broad_topics: list[str] = field(default_factory=lambda: [
    "fintwit",
    "your custom topic here",
    # ...
])
```

### Adjusting Noise Filters

Edit `src/analyzer.py` `NOISE_TERMS` set to add/remove filtered terms.

### Modifying LLM Behavior

Edit the `ANALYST_SYSTEM_PROMPT` in `src/main.py` to adjust analysis style.

## License

MIT

# Twitter/X Economic Sentiment Analysis

*BlackRock's Aladdin
manages $21 trillion in assets, mass-shorted Treasuries before the 2023 bank failures, got caught coordinating with the Fed, and charges $
25,000/month for their sentiment tools. You have a mass-produced Roth IRA from Fidelity and a dream. Let's fucking go.*

A system for discovering emerging market trends from Twitter/X before they hit mainstream news. Uses statistical
analysis, NLP, real-time market data verification, and LLM-powered summarization with semantic memory.

**Key Philosophy**: Most days are boring. This system is calibrated to tell you when something *actually* matters, not
to manufacture urgency like Jim Cramer speed-running his thirteenth margin call of the week.

## Features

- **Dynamic Discovery**: Finds trends you didn't know to look for (not just keyword matching like a 2008 RSS feed)
- **Real-Time Fact Checking**: When some anonymous account with a Pepe avatar screams "SILVER IS MOONING!!!" and it's up
  0.3%, we expose the grift
- **Vector Memory**: Semantic search finds historical parallels - unlike fintwit influencers who recycle the same thread
  every 6 months hoping you forgot
- **Skeptical Analysis**: LLM is explicitly instructed to say "nothing notable today" - a concept that would bankrupt
  CNBC
- **Checkpoint System**: Got rate limited by Elon's clown show? Just run it again
- **SOCKS5 Proxy Support**: For completely legitimate research purposes, officer
- **Background Runner**: Start it, forget it, check logs whenever - like that Mandarin Duolingo streak you started after
  your fifth Renaissance Technologies rejection letter

## Installation

```bash
git clone <repository-url>
cd twitter_sentiment_analysis

uv sync

# spaCy needs this for reasons nobody can explain
uv pip install pip
uv run python -m spacy download en_core_web_sm
```

## Configuration

Copy `config.yaml.example` to `config.yaml` and `.env.example` to `.env`.

Figure it out. The examples are commented. You're building a market intelligence system - if you can't edit a YAML file,
maybe just buy index funds and touch grass.

## Twitter Setup

Cookie auth because Elon broke everything:

1. Log into Twitter in your browser
2. Export cookies with the shadiest cookie exporter extension you can find
3. Save as `cookies.json`
4. Run: `uv run python add_account.py <username> cookies.json`

Add more accounts for rate limit rotation. Configure proxies in `config.yaml` if you're feeling spicy.

## Running

```bash
# Interactive
uv run python main.py

# Background (for VPS)
./run.sh start
./run.sh logs     # watch progress
./run.sh status   # check if running
./run.sh stop     # stop it
```

## How It Works

1. **Scout** - Scrapes 30+ financial topics from Twitter
2. **Investigator** - Extracts trending entities via spaCy NLP
3. **Deep Dive** - Targeted scraping for discovered trends
4. **Fact Checker** - Fetches real prices from Yahoo Finance. Exposes the liars.
5. **Analyst** - LLM generates skeptical summary with signal strength
6. **Reporter** - Emails you a digest so you can feel like a Bloomberg terminal owner
7. **Memory** - Stores everything for future historical parallel detection

## Signal Strength

| Level      | Meaning                 | Frequency                 |
|------------|-------------------------|---------------------------|
| **HIGH**   | Actually unusual. Rare. | 1-2x per month            |
| **MEDIUM** | Worth watching          | Weekly                    |
| **LOW**    | Normal Twitter nonsense | Most days                 |
| **NONE**   | Quieter than usual      | When everyone's at brunch |

Unlike Aladdin, we don't pretend every day is Lehman Brothers.

## Fact Check Classifications

| Tag              | Meaning                                     |
|------------------|---------------------------------------------|
| **VERIFIED**     | They're telling the truth (rare)            |
| **EXAGGERATED**  | Directionally correct, emotionally unhinged |
| **FALSE**        | Lying on the internet. Shocking.            |
| **UNVERIFIABLE** | Made up a ticker or was too vague           |

## Production (pgvector)

If you're running this on a VPS with PostgreSQL:

```yaml
# config.yaml
memory:
  store_type: pgvector
  embedding_dimensions: 1536  # pgvector maxes at 2000 dims
```

Then create the extension: `CREATE EXTENSION vector;`

If you get dimension errors, drop the table and let it recreate:

```sql
DROP TABLE IF EXISTS market_memories;
```

## Troubleshooting

**"No tweets retrieved"** - Your accounts are logged out or banned. Check `uv run twscrape accounts`. Re-add via
cookies.

**"Failed to initialize vector memory: 2000 dimensions"** - Add `embedding_dimensions: 1536` to config. Drop the table.

**"Failed to load spaCy model"** - Did you even read the installation section?
`uv pip install pip && uv run python -m spacy download en_core_web_sm`

**Rate limiting** - Add more accounts. Use proxies. Stop scraping during market hours like everyone else.

## Why This Exists

| Feature                               | This Project         | Aladdin                                                                       | Bloomberg            |
|---------------------------------------|----------------------|-------------------------------------------------------------------------------|----------------------|
| Cost                                  | ~$10/mo in API calls | $25,000/month                                                                 | $24,000/year         |
| Twitter Sentiment                     | Yes                  | Probably                                                                      | Kinda                |
| Will tell you "nothing matters today" | **Yes**              | Never. Gotta justify that invoice.                                            | Lol                  |
| Open Source                           | Yes                  | Absolutely not                                                                | Cute                 |
| Actually makes you a quant            | No                   | Also no                                                                       | Still no             |
| Got no-bid Fed contracts to buy its own ETFs with taxpayer money | No                   | [Yes](https://wallstreetonparade.com/2020/06/blackrock-is-bailing-out-its-etfs-with-fed-money-and-taxpayers-eating-losses-its-also-the-sole-manager-for-335-billion-of-federal-employees-retirement-funds/) | No |

*At least when we're wrong, it's free. When they're wrong, they get a bailout.*

## Disclaimer

Not financial advice. If you YOLO your life savings because this said "HIGH signal" on some shitcoin, that's a you
problem. Hedge funds with actual Aladdin access still lose money.
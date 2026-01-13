# Jafar

**The villain to BlackRock's Aladdin.**

Jafar is a **Consumer Economy Scout** that discovers emerging economic signals from organic social discussion, not just financial Twitter. It uses a **skeptical agentic loop** to filter noise, verify claims with real-time data, and find historical rhymes.

## Core Philosophy

1.  **Consumer Economy First**: An RTX 5090 price hike is a better inflation signal than a CPI print. We track shortages, pricing power, and spending shifts.
2.  **Agentic "Pull" Architecture**: We don't dump data into the prompt. The LLM acts as an analyst: it sees a claim, decides if it needs verification, and **calls tools** to fetch prices or search the web.
3.  **Calibrated Skepticism**: Most days are boring. If Jafar creates urgency every day, it's failing. "Nothing to report" is a successful output.
4.  **No Keyword Lists**: Broad topics + statistical NLP discovery. We find "H200 shortages" by searching for "shortage", not by hardcoding "H200".

## Quick Start

```bash
# 1. Install Dependencies
uv sync
uv run python -m spacy download en_core_web_sm

# 2. Add Twitter Account
uv run add_account.py <username> cookies.json

# 3. Run Pipeline
uv run main.py

# Background Service (Linux/Mac)
./run.sh start | logs | status | stop
```

## Architecture

The system runs a **Discovery → Agentic Analysis** pipeline:

1.  **Broad Scrape**: Scrapes 30+ generic topics ("too expensive", "sold out", "hiring").
2.  **Statistical Signal**: Extracts entities (spaCy) and scores them by engagement velocity & cashtag co-occurrence.
3.  **The Agentic Loop**: The LLM Analyst reviews the top trends.
    *   *Self-Correction*: If a tweet says "Silver crashing!", the Agent calls `get_market_data("SI=F")`. If silver is -0.5%, it marks the claim as "Exaggerated".
    *   *Deep Research*: If a trend is vague ("Uranium spikes"), the Agent calls `search_web("uranium spot price news")` to find the cause.
    *   *History*: If a setup looks familiar, it calls `search_historical_parallels()` to find precedents.
4.  **Reporting**: Sends a calibrated HTML digest.

## Available Tools

The Agent has access to the following tools in `src/tools.py`:
*   `get_market_data(symbols)`: Real-time prices via yfinance.
*   `search_web(query)`: Deep research via DuckDuckGo.
*   `search_historical_parallels(query)`: Semantic search over past digests.
*   `get_trend_timeline(trend)`: Checks if a trend is new or recurring.

## Configuration

*   **`config.yaml`**: App settings (LLM provider, thresholds).
*   **`.env`**: Secrets (API keys, SMTP credentials).

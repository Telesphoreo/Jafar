Here's the project structure:

twitter_sentiment_analysis/
├── .env.example # Template for credentials
├── .gitignore # Git ignore patterns
├── pyproject.toml # uv/Python dependencies
├── main.py # Entry point with setup instructions
└── src/
├── __init__.py
├── config.py # Configuration from environment
├── scraper.py # Twitter scraper (twscrape)
├── analyzer.py # NER trend extraction (spaCy)
├── reporter.py # Email reporter (SMTP)
├── main.py # Pipeline orchestration
└── llm/
├── __init__.py
├── base.py # Abstract LLM interface
├── factory.py # Provider factory
├── openai_client.py # OpenAI implementation
└── google_client.py # Google Gemini implementation

Key Features:

1. Modular Architecture - Each component (scraper, analyzer, LLM, reporter) is isolated and testable
2. Swappable LLM Providers - Set LLM_PROVIDER=openai or LLM_PROVIDER=google in .env to switch
3. Async Twitter Scraping - Uses twscrape with concurrent topic/trend fetching
4. Smart NER Analysis - Filters out noise, focuses on ORG/GPE/PRODUCT entities
5. Anti-Spam LLM Prompting - System prompt instructs the LLM to ignore bots and promotional content
6. HTML Email Reports - Professional formatting with trend badges and stats

To get started:

# 1. Install dependencies

uv sync

# 2. Download spaCy model

python -m spacy download en_core_web_sm

# 3. Configure credentials

cp .env.example .env

# Edit .env with your API keys

# 4. Add Twitter account (REQUIRED)

twscrape add_account <username> <password> <email> <email_password>
twscrape login_all

# 5. Run the pipeline

python main.py
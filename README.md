# Jafar

### The villain to BlackRock's Aladdin.

BlackRock spent $1 billion building Aladdin. We spent $35 on a Raspberry Pi. Somewhere in Manhattan, a mass of
engineers is maintaining a Hadoop cluster that was cutting-edge when Obama was in his first term to do what this does
with scikit-learn and a grudge. Their Confluence pages have more architectural diagrams than our entire codebase has
files. They hold quarterly "modernization syncs" to debate migrating from Java 11 to Java 17 while a committee of
eight reviews the JIRA ticket. We deployed on a Friday afternoon.

Aladdin processes 30,000 trades a day. Jafar processes 30,000 tweets and finds the trades before they happen.

**[View an actual digest](digest.pdf)** - No NDA required. No 90-minute sales call. No enterprise sales rep who says
"let me loop in my solutions architect" and then ghosts you for six weeks. Just the output, right there, for free,
like information in a functioning democracy.

## What is Jafar?

Jafar is a **Consumer Economy Scout** - it discovers emerging economic signals from organic social discussion, not
financial Twitter echo chambers. An RTX 5090 hitting $5,000 on secondary markets tells you more about consumer demand
and pricing power than any CPI print. A retail manager tweeting about empty shelves is a better inflation signal than
whatever Larry Fink said on CNBC this morning while carefully not mentioning the
[$25 million of Fed Chair Powell's personal wealth that BlackRock
manages](https://wallstreetonparade.com/2020/05/fed-chair-powell-has-upwards-of-11-6-million-invested-with-blackrock-the-firm-that-will-manage-a-750-billion-corporate-bond-bailout-program-for-the-fed/).

### Core Philosophy

1. **Consumer Economy First** - We search for "too expensive" and "sold out", not "$NVDA." The real economy lives in
   grocery bills and GPU prices, not earnings call transcripts that say "cautiously optimistic" forty-seven times.

2. **Agentic "Pull" Architecture** - The LLM acts as an analyst, not a parrot. It sees a claim, gets suspicious,
   and calls tools to verify. No data dumping. No prompt stuffing. Aladdin's engineering team probably loads the
   entire S&P 500 into context every morning because someone wrote a design doc in 2014 that said "more data = better"
   and nobody's questioned it since.

3. **Calibrated Skepticism** - Most days are boring. If Jafar creates urgency every day, it's failing. "Nothing to
   report" is a successful output. This concept alone would put CNBC out of business and save Aladdin's clients
   approximately $200k/year in "insights" that amount to "markets moved today."

4. **No Keyword Lists** - Broad topics + statistical NLP discovery. We find "H200 shortages" by searching for
   "shortage", not by hardcoding "H200" into a config file that hasn't been updated since Q3 because the analyst who
   maintained it left for Citadel.

## Architecture

Eleven pipeline steps. Does more before your morning coffee than Aladdin's morning standup accomplishes before lunch.
Every single one of these would be a separate microservice at BlackRock with its own team, its own Kubernetes
namespace, and its own Slack channel where nobody posts except the on-call bot.

### The Pipeline

1. **Scout** - Scrapes 30+ topics from Twitter. Not just fintwit. Consumer economy: "too expensive", "sold out",
   "can't afford", "shrinkflation." The stuff real people say when their grocery bill hits $200 for the third week
   running. Aladdin's sentiment feed probably costs extra and still just tracks cashtags like it's 2016.

2. **Twitter Trends** - Fetches actual trending topics from Twitter's API and merges them into discovered signals.
   Non-keyword discovery. We find what's trending without maintaining a list, because maintaining lists is what you
   do when you have 200 engineers and a VP of List Maintenance. We have neither.

3. **Investigator** - spaCy NER extracts entities. Engagement velocity scoring. Cashtag co-occurrence analysis.
   Math, not vibes. The Aladdin team probably has a "Sentiment Analysis Working Group" that meets biweekly
   to discuss whether their NLP model - trained on data from the second Bush administration - should maybe consider
   learning what "shrinkflation" means.

4. **LLM Filter** - Gemini pre-filter separates signal from noise. Rejects "Christmas" and "Books", keeps "$NVDA"
   and "shortage." Rejected trends get demoted to the mentions tier. Even Twitter's rejected noise occasionally has a
   kernel of truth, unlike Aladdin's quarterly outlook which has the predictive power of a coin flip but costs more
   than a liberal arts degree.

5. **Deep Dive** - Targeted scraping on validated trends. Not fifteen of them. Not whatever got the most likes. The
   ones that survived the filter. We do focused work here. BlackRock's approach is presumably to ingest
   everything, store it in a data lake that costs $40k/month, and then have a team of consultants from Deloitte
   spend six months building a dashboard nobody looks at.

6. **News Roundup** - Fetches real economic news via DuckDuckGo. Reuters, Bloomberg, AP - whatever surfaces that
   isn't an SEO farm in a trench coat. Runs independently of Twitter because the global economy doesn't stop
   functioning just because fintwit is arguing about whether $PLTR is a meme stock or a defense contractor.

7. **Fact Checker** - Verifies market claims against real-time yfinance data. Someone tweets "oil is collapsing" and
   crude is down 0.8%? That's not collapsing. That's a Tuesday. We will tell you it's a Tuesday. Aladdin presumably
   does this too, except it takes a REST call through four API gateways, a Kafka topic, and a microservice called
   `FactCheckOrchestratorServiceV2` that nobody remembers deploying.

8. **Temporal Analyzer** - Tracks trends across days. Is this Day 5 of "egg shortage" discourse or did someone just
   discover grocery stores? Catches recurring grifts. Has better pattern recognition than the SEC, which is a bar so
   low you'd need ground-penetrating radar to find it.

9. **The Agent** - The LLM gets everything and a toolkit. It sees claims, gets suspicious, calls tools. Five tool
   calls max because we're not burning $50 in API costs investigating some nobody's pump-and-dump scheme. Writes you
   a news roundup AND a Twitter analysis because you deserve both without opening six apps and whatever Bloomberg
   terminal-shaped monstrosity Aladdin's engineers are duct-taping together this sprint.

10. **ML Pipeline** - Bot detection, account signal scoring, LLM-as-judge validation. Fully autonomous. See below.

11. **Reporter** - Emails you a formatted HTML digest. News roundup first, Twitter signals second, fact checks
    third. Shows its work. Transparent by default - a concept BlackRock treats like a foreign language they pretend to
    speak at Davos but can't conjugate under oath.

## ML Pipeline

This is new. This is cool. This is what happens when you don't have to route a feature request through seventeen
layers of middle management and a "Machine Learning Center of Excellence" that's really just three people who took
an Andrew Ng course and a VP who puts "AI/ML" in his LinkedIn headline.

### Bot Detection (Isolation Forest)

Unsupervised anomaly detection across 16 behavioral features per Twitter account. No labeled data required. Catches
bot-like patterns: suspiciously regular posting intervals, template content, zero-engagement tweets, 24/7 posting
with no sleep pattern.

The Aladdin team presumably has a bot detection system too. It was probably spec'd out in a 47-page requirements
document, built by Accenture over eighteen months, and flags accounts as "potentially automated (confidence: medium,
please consult supplementary appendix B)" while our IsolationForest just says "bot" and moves on.

### Account Signal Scoring (KMeans Clustering)

Not all Twitter accounts are equal. A retail manager tweeting about shelf shortages at 11pm is more valuable than
Reuters regurgitating the same headline across fourteen accounts. We score for:

- **Content originality** - URL ratio, unique content, question frequency
- **Engagement quality** - Likes, replies, consistency, engagement-per-view
- **Discovery signal** - How many different trends does this account surface?
- **Timing patterns** - Active hours, posting frequency

Clusters accounts into four types: `high_signal`, `news_aggregator`, `casual`, `low_activity`. Finds the people who
actually know things, not the people who have blue checkmarks and opinions.

### LLM-as-Judge Validation

Gemini evaluates every account the ML flagged, creating pseudo-labeled ground truth. Then we compute precision,
recall, and F1 of the ML model against the LLM's judgments. The system evaluates its own accuracy. Every run. Without
being asked.

BlackRock's model validation process probably involves a quarterly review committee, three sign-offs, and a
compliance officer who has to initial every confusion matrix. We just... do it. Automatically. Twice a day.

### Self-Evaluating, Autonomous, Zero Maintenance

The ML pipeline runs every pipeline execution. Trains on whatever tweets are in the database. Emails you ML
diagnostics (precision, recall, F1, top accounts, bot flags) every few days so you can see how the models are
performing without logging into anything. If the models drift, you'll know before Aladdin's team finishes
their next sprint planning meeting.

## The Stack

| Component       | What we use               | What Aladdin probably uses                                                |
|-----------------|---------------------------|---------------------------------------------------------------------------|
| Language        | Python 3.13               | Java 11 (migration to 17 "in progress" since 2022)                        |
| ML              | scikit-learn              | A Spark cluster that costs more per month than most mortgages             |
| NLP             | spaCy (en_core_web_lg)    | Something custom that was cutting-edge during the financial crisis        |
| LLM             | Google Gemini             | Twelve internal models behind an API gateway that returns XML             |
| Database        | PostgreSQL + pgvecto.rs   | Oracle. It's always Oracle. And a Hadoop cluster "for historical reasons" |
| Embeddings      | Gemini embeddings         | A team of PhDs maintaining Word2Vec like it's a heritage site             |
| Deployment      | systemd on a Raspberry Pi | Kubernetes on AWS with a $2M/month cloud bill and a "FinOps team"         |
| Package manager | uv                        | Maven. Still Maven. There was a Gradle proposal in 2019. It was rejected. |

Total cost: ~$10/month in API calls, running on hardware that fits in your palm.

Aladdin's annual technology budget is reportedly in the billions. Their engineering team has more people than some
countries have software developers. They hold "architecture review boards." We hold a Python file called `main.py`
that does the whole thing in eleven steps and doesn't need a service mesh to talk to itself.

## Quick Start

```bash
git clone https://github.com/Telesphoreo/Jafar
cd Jafar

# Install everything (deps + spaCy model)
uv sync

# Add a Twitter account
uv run python add_account.py <username> cookies.json

# Run the pipeline
uv run jafar
```

Three commands. Done before Aladdin's enterprise sales rep finishes typing their "just circling back on this"
follow-up email. Before their onboarding team schedules the "pre-kickoff alignment sync." Before the Accenture
consultant they hired to implement it finishes updating the RACI matrix.

### Twitter Setup

Cookie auth because Elon broke the API and then charged $42,000/month for the remains:

1. Log into Twitter in your browser
2. Export cookies (browser extension)
3. Save as `cookies.json`
4. `uv run python add_account.py <username> cookies.json`

The script handles proxy assignment automatically. Round-robin across your proxies. No spreadsheets. No deranged
accountant energy. More accounts = more parallel workers = faster scraping.

## Configuration

Copy `config.yaml.example` to `config.yaml` and `.env.example` to `.env`. The examples are commented. This is the
easiest part. If you can edit two text files, you can run a market intelligence system that competes with software
whose sales cycle is longer than most marriages.

- **`config.yaml`** - App settings, LLM provider, scraping thresholds, ML parameters
- **`.env`** - Secrets (API keys, SMTP credentials, database URL)

## Available Tools

The agentic LLM has access to these tools and will call them unprompted when it gets suspicious, like a paranoid
research assistant who genuinely does not trust anyone:

| Tool                                 | What it does                                                                   |
|--------------------------------------|--------------------------------------------------------------------------------|
| `get_market_data(symbols)`           | Real-time prices via yfinance. Exposes the "MOONING" liars.                    |
| `search_web(query)`                  | DuckDuckGo deep research. Turns Twitter vibes into actual intelligence.        |
| `fetch_news(query)`                  | Live news headlines. For when Twitter isn't enough (always).                   |
| `search_historical_parallels(query)` | Semantic search over past digests. "This feels familiar" - finds the receipts. |
| `get_trend_timeline(trend)`          | New trend or recycled cope from last month?                                    |
| `get_weather_forecast(cities)`       | Panic buying in Houston? Checks if there's actually a hurricane.               |

Five tool calls max per analysis. We are not funding some random's engagement farming operation with unlimited
Gemini API calls.

## Signal Strength

| Level      | Meaning                 | Frequency                 | You still get a digest?           |
|------------|-------------------------|---------------------------|-----------------------------------|
| **HIGH**   | Actually unusual. Rare. | 1-2x per month            | Obviously, and read it now        |
| **MEDIUM** | Worth watching          | Weekly                    | Yes                               |
| **LOW**    | Normal Twitter noise    | Most days                 | Yes                               |
| **NONE**   | Twitter had nothing     | When everyone's at brunch | **Yes** - news roundup still hits |

Signal strength measures Twitter activity, not whether the digest matters. NONE days still get a full news roundup.
The economy doesn't pause because fintwit took a nap.

## Fact Check Classifications

| Tag              | Meaning                                     |
|------------------|---------------------------------------------|
| **VERIFIED**     | They told the truth. Mark your calendars.   |
| **EXAGGERATED**  | Directionally correct, emotionally unhinged |
| **FALSE**        | Lying on the internet. Groundbreaking.      |
| **UNVERIFIABLE** | Made up a ticker or was too vague to check  |

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run fast tests only (skip slow/integration)
uv run pytest tests/ -m "not slow"
```

Full test suite with mocks for all external dependencies. Tests run fast, don't need API keys, and actually pass -
a trifecta that Aladdin's CI/CD pipeline, which reportedly takes 4 hours and involves three Jenkins servers that
predate the current CTO's tenure, has never achieved simultaneously.

## Deployment

### systemd (Recommended)

Production deployment uses systemd timers with randomized scheduling. Two daily runs in randomized windows so your
scraping pattern looks like a normal insomniac doomscrolling fintwit, not a cron job that screams "I'M A BOT" at
exactly 2:00 PM every day.

```
systemd/
├── jafar.service            # The service unit
├── jafar.timer              # Single daily run (8am-8pm window)
├── jafar-morning.timer      # Morning run (7am-12pm window)
└── jafar-evening.timer      # Evening run (5pm-11pm window)
```

Copy to `/etc/systemd/system/`, edit paths, enable timers. `RandomizedDelaySec` handles the rest. The service has
memory limits (2GB), CPU quotas (80%), auto-restart on failure, and log rotation. It's more resilient than most
things BlackRock's SRE team monitors, and it doesn't require a PagerDuty enterprise license to tell you when
something breaks - it just emails you in plain English.

### Quick Background Run

```bash
./run.sh start    # Start
./run.sh logs     # Watch progress
./run.sh status   # Check if running
./run.sh stop     # Stop it
```

## Production Database (pgvecto.rs)

```yaml
# config.yaml
memory:
  store_type: pgvector
  embedding_dimensions: 1536
```

Create the extension: `CREATE EXTENSION vectors;`

One database. Tweets, embeddings, ML training data, historical digests, app state. All in PostgreSQL.
BlackRock presumably has a separate database for each of these, a data warehouse for analytics, a data lake for "raw
ingestion," a data lakehouse because someone went to a Databricks conference, and a team of six DBAs whose primary
job is explaining to leadership why the Oracle license costs more than the GDP of a small Pacific island nation.

## Why This Exists

| Feature                                   | Jafar                          | Aladdin                                                    |
|-------------------------------------------|--------------------------------|------------------------------------------------------------|
| Cost                                      | ~$10/mo                        | More than your rent, your car, and your therapist combined |
| Daily economic digest                     | Yes                            | Yes, for the price of a used Toyota per quarter            |
| Twitter sentiment                         | Yes                            | Buried under 47 layers of enterprise middleware            |
| Bot detection ML                          | Yes (IsolationForest)          | Probably a rules engine from 2009                          |
| Account quality scoring                   | Yes (KMeans)                   | "We have a proprietary methodology" (it's a spreadsheet)   |
| Self-evaluating ML                        | Yes (LLM-as-judge, F1 metrics) | Annual model review (they skip it sometimes)               |
| Will tell you "nothing happened"          | Yes                            | No - gotta justify the invoice                             |
| Open source                               | Yes                            | Lmao                                                       |
| Runs on a Raspberry Pi                    | Yes                            | Runs on enough servers to heat a small building            |
| Risk models updated for post-2008 reality | Yes (retrained every run)      | "That's on the roadmap"                                    |

When we're wrong, it's free. When they're wrong, they get a bailout, a CNBC interview to explain why it was
actually your fault, and a LinkedIn post from their junior analyst about "lessons learned in volatile markets" that
gets 4,000 likes from other people who also lost your money.

## The Corruption Receipts

You made it this far, so either you're interested or you work at BlackRock's engineering org and you're hate-reading
this during your mandatory "innovation hour" that was supposed to be for side projects but is actually just catching
up on the sprint backlog. Either way, here's the starter pack:

- **Fed Chair Powell has $25M personally invested with BlackRock** while handing them no-bid contracts to manage
  $750 billion in bailout money.
  ["Extremely carefully managed"](https://www.investing.com/news/economy/blackrock-conflicts-managed-extremely-carefully-feds-powell-says-2245953),
  he says.
  ([Source](https://wallstreetonparade.com/2020/05/fed-chair-powell-has-upwards-of-11-6-million-invested-with-blackrock-the-firm-that-will-manage-a-750-billion-corporate-bond-bailout-program-for-the-fed/))

- **BlackRock wrote the bailout playbook before the crisis existed.** August 2019: publish "Going Direct" paper
  proposing central banks inject money into the economy. Six months later, three central banks hire BlackRock to
  execute the exact plan they
  authored. ([Source](https://wallstreetonparade.com/2020/06/blackrock-authored-the-bailout-plan-before-there-was-a-crisis-now-its-been-hired-by-three-central-banks-to-implement-the-plan/))

- **55.8% of their funds underperform their benchmarks.** Some pension funds returned -50.91% over three years while
  the sector average was positive. But the fees were
  collected. ([Source](https://www.yodelar.com/insights/blackrock-review))

- **Dutch pension funds pulled $5.9 billion** because even the Netherlands decided BlackRock wasn't acting in their
  beneficiaries' best interests. When the Dutch think you're too greedy, you've accomplished something
  remarkable. ([Source](https://www.fa-mag.com/news/blackrock-loses--5-9-billion-mandate-from-dutch-pension-pme-85220.html))

- **Larry Fink said "I'm ashamed of being part of this conversation" about ESG**, then denied saying it, in the
  same interview, on
  camera. ([Source](https://www.axios.com/2023/06/26/larry-fink-ashamed-esg-weaponized-desantis))

- **Their own former CIO for Sustainable Investing** quit and called the ESG operation "a dangerous placebo that
  harms the public
  interest." ([Source](https://www.cnbc.com/2021/08/24/blackrocks-former-sustainable-investing-chief-says-esg-is-a-dangerous-placebo.html))

- **$11 billion in coal investments** while being the world's largest investor in coal-fired power
  stations. ([Source](https://en.wikipedia.org/wiki/BlackRock#Investments_in_contributors_to_climate_change))

- **Dropped ESG shareholder support from 47% to 4%** the exact moment Ron DeSantis made it politically
  inconvenient. ([Source](https://fortune.com/2024/02/14/blackrock-voting-choice-ceo-larry-fink-shareholder-democracy-stakeholder-capitalism-esg/))

This project costs $10/month. Aladdin costs more, and the people running it are doing all of the above while their
engineering team posts "excited to share that our team shipped a new internal dashboard" on LinkedIn like they
didn't just spend nine months and $3 million rebuilding something that already existed in Grafana.

## Disclaimer

Not financial advice. Not even close. If you YOLO your life savings because this said "HIGH signal" on some
shitcoin, that is a you problem. Hedge funds with actual Aladdin access lose money all the time - the difference is
they get bailed out with your taxes and then go on CNBC to explain why it was actually your fault for not being
diversified enough across their seventeen underperforming products. This system pulls publicly available data, feeds
it to an LLM that is constitutionally incapable of feeling FOMO, and emails you a summary. Our bad takes are free.
BlackRock's bad takes come with a management fee and a 40-page shareholder letter about "navigating uncertainty" that
says less than a fortune cookie.

In January 2021, a bunch of people on Reddit with Robinhood accounts almost bankrupted a $13 billion hedge fund
because they liked a stock. Melvin Capital needed a $2.75 billion emergency bailout and closed permanently a year
later. All because regular people had access to the same information at the same time. This is that energy, but for
economic intelligence. We gave you the tools. Go.

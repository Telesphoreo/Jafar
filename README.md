# Jafar

### The villain to BlackRock's Aladdin.

BlackRock manages \$11.5 trillion. Their CEO
has [\$25 million personally invested with BlackRock](https://wallstreetonparade.com/2020/05/fed-chair-powell-has-upwards-of-11-6-million-invested-with-blackrock-the-firm-that-will-manage-a-750-billion-corporate-bond-bailout-program-for-the-fed/).
Wait, that's the *Fed Chair*. The Fed Chair has \$25 million invested with the company the Fed hired to manage \$750
billion in bailout money.
BlackRock [wrote the bailout playbook](https://wallstreetonparade.com/2020/06/blackrock-authored-the-bailout-plan-before-there-was-a-crisis-now-its-been-hired-by-three-central-banks-to-implement-the-plan/)
six months before the crisis, then got hired to execute it.
Their [own sustainability CIO quit and called ESG "a dangerous placebo."](https://www.cnbc.com/2021/08/24/blackrocks-former-sustainable-investing-chief-says-esg-is-a-dangerous-placebo.html) [55.8% of their funds underperform their benchmarks.](https://www.yodelar.com/insights/blackrock-review)
Their risk platform has 4,000 engineers and a technology budget in the billions. Our API bill last month was 33 cents.

You have a mass-produced Roth IRA from Fidelity and a dream. Let's fucking go.

## What does it actually look like?

**[View an actual digest](digest.pdf)**. No NDA, no 90-minute discovery call, no enterprise contract. Just the output.

## Features

- **Consumer economy first.** Searches for "too expensive" and "sold out", not "\$NVDA." An RTX 5090 hitting \$5,000 on
  secondary markets is a better inflation signal than any CPI print.
- **Agentic pull architecture.** The LLM acts as a skeptical analyst with tools. It sees a claim, gets suspicious, and
  calls yfinance to check. No data dumping.
- **Calibrated skepticism.** Most days are boring. "Nothing to report" is a successful output. This alone would
  collapse CNBC.
- **ML dashboard.** Garbage detection, signal scoring, LLM-as-judge, human-in-the-loop review. A full control center
  with a Windows Aero aesthetic and a green glass progress bar. Run it with `uv run dashboard`. Bloomberg's terminal
  hasn't been redesigned since the Cold War. Ours has a shimmer animation.
- **No keyword lists.** Broad topics + statistical NLP. Finds "H200 shortages" by searching "shortage", not by
  hardcoding "H200" into a config last updated by an analyst who left for Citadel.
- **Vector memory.** Remembers every past digest and finds historical parallels via semantic search. Catches fintwit
  grifters recycling the same thread every six months with a new profile picture.

## How It Works

Two commands. `uv run jafar` runs the pipeline. `uv run dashboard` opens the ML control center. Does more before
you've sat down at your desk with your performative espresso (the one from the office's \$4,000 La Marzocca that took
you ten minutes because you still haven't figured out all the knobs) than Aladdin's entire morning standup accomplishes
before lunch.

1. **Scout.** Scrapes 30+ broad topics from Twitter ("too expensive", "sold out", "shrinkflation"). Consumer economy,
   not fintwit cashtag echo chambers.

2. **Twitter Trends.** Fetches actual trending topics from the platform. Non-keyword discovery without maintaining a
   list and a VP of List Governance.

3. **Investigator.** spaCy NER entity extraction, engagement velocity scoring. Math, not vibes.

4. **LLM Filter.** Gemini separates signal from noise. Rejects "Christmas", keeps "shortage." Has better judgment than
   Aladdin's quarterly outlook.

5. **Deep Dive.** Targeted scraping on validated trends. The ones that survived the filter, not whatever got the most
   likes from reply guys.

6. **Fact Checker.** Verifies claims against real-time yfinance data. Someone tweets "oil is collapsing" and crude is
   down 0.8%? That's a Tuesday.

7. **Temporal Analyzer.** Tracks trends across days. Is this Day 5 of "egg shortage" or did someone just discover
   grocery stores?

8. **The Agent.** The LLM gets everything and a toolkit. Five tool calls max because we're not burning a whole dollar
   investigating some nobody's pump-and-dump scheme for a coin named after a dog wearing a hat.

   | Tool                                 | What it does                                                                          |
   |--------------------------------------|---------------------------------------------------------------------------------------|
   | `get_market_data(symbols)`           | Real-time prices via yfinance. Exposes the "MOONING" liars.                           |
   | `search_web(query)`                  | DuckDuckGo deep research. Twitter vibes → actual intelligence.                        |
   | `fetch_news(query)`                  | Live news headlines. For when Twitter isn't enough (always).                          |
   | `search_historical_parallels(query)` | Semantic vector search over past digests. "This feels familiar." Finds the receipts.  |
   | `get_trend_timeline(trend)`          | New trend or recycled cope from last month?                                           |
   | `get_weather_forecast(cities)`       | Panic buying in Houston? Checks if there's actually a hurricane.                      |

9. **Reporter.** Emails you a formatted HTML digest with signal strength, fact checks, and historical parallels. No ML
   results in the digest. That's what the dashboard is for. The digest is for humans who don't want to see confusion
   matrices with their morning coffee.

10. **History.** Stores the digest to vector memory for future semantic search. Admin diagnostics email with run stats.

The pipeline also filters tweets from blocked accounts before analysis. If the ML dashboard flagged @CryptoSpamBot9000
as garbage, their tweets never touch the LLM. Saves tokens. Saves dignity.

## ML Dashboard

```bash
uv run dashboard
```

That's it. You now have a control center that Bloomberg charges $25,000/yr to approximate poorly.

The dashboard is a Flask app with a UI aesthetic that would make Windows Vista shed a single tear of pride. Dark chrome
navbar with an amber glow line. Brushed metal card headers. A progress bar that is a 1:1 recreation of the Windows 7
progress bar, green glass gradient and all, shimmer included, because if you're going to build a dashboard you should
build one with a soul. No Tailwind. No shadcn. No Lucide icons. No "modern minimal" that looks like every other
ChatGPT-generated SaaS landing page that will pivot to AI agents next quarter. We went full Aero.

- **Garbage detection.** IsolationForest across 16 behavioral features. Not "bot detection." @DeItaone is automated
  and invaluable. We classify signal vs garbage.
- **Signal scoring.** Percentile-ranked composite of engagement, content originality, trend coverage, and recurrence.
  Finds the retail workers and industry insiders, not the blue checkmarks.
- **LLM-as-Judge.** Gemini reads actual tweets and classifies. 80%+ confidence auto-applies. Uncertain ones go to
  your review queue.
- **HITL review.** Three buttons: signal, garbage, unsure. Simple enough that even an MBA wouldn't need a walkthrough.
  "Unsure" comes back when new tweets arrive.
- **Watch list.** High-signal accounts get scraped every pipeline run. Alternative data without the terabytes of
  satellite imagery and the CNN that thinks tree shadows are Camrys. Ask me how I fucking know this.
- **Block list.** Crypto spam, follower farms, "Bullish" reply bots. Blocked forever. They don't deserve the compute.
- **Search.** Type a username, hit enter. Bloomberg requires ticker, EQUITY, GO. Three keystrokes. Ours is one.

Windows Aero revival. Dark chrome navbar, amber glow line, brushed metal headers, Win7 green glass progress bar.
Zune-era lowercase typography. Sharp edges. Bloomberg looks like 1987. Aladdin looks like SAP. We look like Vista
went to design school.

## Installation

```bash
git clone https://github.com/Telesphoreo/Jafar && cd Jafar
uv sync
```

Done before Aladdin's enterprise sales rep finishes typing "just circling back on this."

## Configuration

Copy `config.example.yaml` to `config.yaml` and `.env.example` to `.env`. The examples are commented. Two text files. No
change advisory board.

## Twitter Setup

Cookie auth because Elon broke the API and charged \$42,000/month for the wreckage:

1. Log into Twitter, export cookies (browser extension), save as `cookies.json`
2. `uv run python add_account.py <username> cookies.json`

More accounts = more parallel workers = faster scraping.

## Running

```bash
# Run the pipeline (scrape, analyze, email digest)
uv run jafar

# Open the ML dashboard (garbage detection, signal scoring, HITL review)
uv run dashboard
```

The pipeline stores everything to Postgres. The dashboard reads from Postgres. They don't need to run at the same time.
Run the pipeline on a cron, open the dashboard when you feel like reviewing accounts. Two processes. Zero coordination.
Aladdin needs a deployment team. We need `uv run`.

For production deployment with systemd timers and randomized scheduling, see **[DAEMONIZING.md](DAEMONIZING.md)**.

## Signal Strength

| Level      | Meaning                 | Frequency                 | You still get a digest?         |
|------------|-------------------------|---------------------------|---------------------------------|
| **HIGH**   | Actually unusual. Rare. | 1-2x per month            | Obviously, and read it now      |
| **MEDIUM** | Worth watching          | Weekly                    | Yes                             |
| **LOW**    | Normal Twitter noise    | Most days                 | Yes                             |
| **NONE**   | Twitter had nothing     | When everyone's at brunch | **Yes**, the LLM still searches |

## Fact Check Classifications

| Tag              | Meaning                                     |
|------------------|---------------------------------------------|
| **VERIFIED**     | They told the truth. Mark your calendars.   |
| **EXAGGERATED**  | Directionally correct, emotionally unhinged |
| **FALSE**        | Lying on the internet. Groundbreaking.      |
| **UNVERIFIABLE** | Made up a ticker or was too vague to check  |

## Production Database (VectorChord)

```sql
CREATE
EXTENSION vector;
CREATE
EXTENSION vchord CASCADE;
```

One database. Tweets, embeddings, ML scores, signal judgments, account watch/block lists, pipeline checkpoints,
historical digests. All PostgreSQL. Advisory locks for pipeline concurrency. No Oracle DBA explaining why the license
renewal costs more than the GDP of Micronesia.

## Why This Exists

| Feature                          | Jafar              | Aladdin                                                    | Bloomberg Terminal                         | Morning Brew               |
|----------------------------------|--------------------|------------------------------------------------------------|--------------------------------------------|----------------------------|
| Cost                             | ~\$0.33/mo         | More than your rent, your car, and your therapist combined | \$25,000/yr                                | Free (you are the product) |
| Daily economic digest            | Yes                | Yes, for the price of a used BMW per quarter               | Yes, if you can read the UI without crying | Yes, but it's vibes        |
| ML dashboard                     | Windows Aero       | SAP on a CRT                                               | A mainframe having a seizure               | A WordPress template       |
| Consumer economy signals         | Twitter + NLP + ML | Entombed under 47 layers of enterprise middleware          | Cashtags and terminals                     | "Here's what's trending!"  |
| Garbage detection                | IsolationForest    | A rules engine from 2009                                   | N/A                                        | N/A                        |
| Signal/garbage classification    | LLM judge + HITL   | Annual review (postponed, rescheduled, canceled)           | N/A                                        | N/A                        |
| Will tell you "nothing happened" | Yes                | Silence doesn't generate billable hours                    | The terminal is always screaming           | Every email is urgent      |
| Open source                      | Yes                | Lmao                                                       | Lmao                                       | "We have a newsletter"     |
| Runs on a single board computer  | Yes                | Enough servers to heat a city block                        | A desktop app from 2003                    | A WordPress site           |
| UI aesthetic                     | Vista glow line    | Classified (leaked: SAP)                                   | 1987                                       | 2019 WordPress theme       |

## The Corruption Receipts

Here's the starter pack:

- **Fed Chair Powell has \$25M invested with BlackRock** while handing them no-bid contracts for \$750B in bailout
  money. Called
  it ["extremely carefully managed."](https://www.investing.com/news/economy/blackrock-conflicts-managed-extremely-carefully-feds-powell-says-2245953) ([Source](https://wallstreetonparade.com/2020/05/fed-chair-powell-has-upwards-of-11-6-million-invested-with-blackrock-the-firm-that-will-manage-a-750-billion-corporate-bond-bailout-program-for-the-fed/))

- **Wrote the bailout playbook before the crisis.** Published "Going Direct" in August 2019. Six months later, three
  central banks hired them to execute the exact
  plan. ([Source](https://wallstreetonparade.com/2020/06/blackrock-authored-the-bailout-plan-before-there-was-a-crisis-now-its-been-hired-by-three-central-banks-to-implement-the-plan/))

- **55.8% of funds underperform benchmarks.** Some pension funds returned -50.91% over three years while the sector
  average was positive. But the fees were collected. The fees are always
  collected. ([Source](https://www.yodelar.com/insights/blackrock-review))

- **Dutch pension funds pulled \$5.9 billion** because even the Netherlands decided BlackRock wasn't acting in their
  beneficiaries' interests. When the Dutch think you're too greedy, you've achieved something
  extraordinary. ([Source](https://www.fa-mag.com/news/blackrock-loses--5-9-billion-mandate-from-dutch-pension-pme-85220.html))

- **Larry Fink said "I'm ashamed of being part of this conversation" about ESG**, then denied saying it, in the same
  interview, on camera. ([Source](https://www.axios.com/2023/06/26/larry-fink-ashamed-esg-weaponized-desantis))

- **Former CIO for Sustainable Investing quit** and called
  ESG ["a dangerous placebo."](https://www.cnbc.com/2021/08/24/blackrocks-former-sustainable-investing-chief-says-esg-is-a-dangerous-placebo.html)
  When your sustainability chief says the sustainability thing is fake, that's not a PR problem. That's a confession.

- **\$11 billion in coal investments** while running ad campaigns about their commitment to a sustainable
  future. ([Source](https://en.wikipedia.org/wiki/BlackRock#Investments_in_contributors_to_climate_change))

- **ESG shareholder support dropped from 47% to 4%** the exact millisecond Ron DeSantis made it politically
  inconvenient. Principles! (Terms and conditions
  apply.) ([Source](https://fortune.com/2024/02/14/blackrock-voting-choice-ceo-larry-fink-shareholder-democracy-stakeholder-capitalism-esg/))

## Disclaimer

Not financial advice. If you YOLO your life savings because this said "HIGH signal" on some shitcoin named after a
cartoon animal, that is a you problem of extraordinary proportions. Hedge funds with actual Aladdin access lose money
all the time. The difference is they get bailed out with your taxes and then go on CNBC to explain why it was your
fault for not being diversified across their seventeen underperforming products. Our bad takes are free. BlackRock's
bad takes come with a management fee and a shareholder letter about "navigating uncertainty" that says less than a
fortune cookie. Larry is seething. Go.

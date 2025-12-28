# Jafar - The villain to BlackRock's Aladdin.

BlackRock's Aladdin manages \$21 trillion dollars in assets while their CEO
has [private phone calls with the Fed Chair](https://wallstreetonparade.com/2020/08/fed-chair-powell-had-4-private-phone-calls-with-blackrocks-ceo-since-march-as-blackrock-manages-upwards-of-25-million-of-powells-personal-money-and-lands-3-no-bid-deals-with-the-fed/)
who conveniently has
[$25 million of his personal wealth invested with BlackRock](https://wallstreetonparade.com/2020/05/fed-chair-powell-has-upwards-of-11-6-million-invested-with-blackrock-the-firm-that-will-manage-a-750-billion-corporate-bond-bailout-program-for-the-fed/).
They charge god-knows-what per month for sentiment tools (they won't publish pricing because it would make defense
contractors blush). You have a mass-produced Roth IRA from Fidelity and a dream. Let's
fucking go.

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
  CNBC and make BlackRock's [55.8% underperforming funds](https://www.yodelar.com/insights/blackrock-review) look even
  worse
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
maybe just buy index funds and touch grass. At least you won't be paying BlackRock's fees
while [their own executive admits ESG is a "dangerous placebo that harms the public interest"](https://www.cnbc.com/2022/12/07/activist-investor-calls-for-blackrock-ceo-fink-to-step-down-over-esg-hypocrisy.html).

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
6. **Reporter** - Emails you a digest so you can pretend you're a Bloomberg terminal owner without paying $24k/year
7. **Memory** - Stores everything for future historical parallel detection

## Signal Strength

| Level      | Meaning                 | Frequency                 |
|------------|-------------------------|---------------------------|
| **HIGH**   | Actually unusual. Rare. | 1-2x per month            |
| **MEDIUM** | Worth watching          | Weekly                    |
| **LOW**    | Normal Twitter nonsense | Most days                 |
| **NONE**   | Quieter than usual      | When everyone's at brunch |

Unlike Aladdin, we don't pretend every day is Lehman Brothers. We also
don't [hold $11 billion in coal investments while lecturing everyone about climate](https://en.wikipedia.org/wiki/BlackRock#Investments_in_contributors_to_climate_change).

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

**My landlord is Sentinel Real Estate** - Can't help you there. They crank out
[5-over-1 "stumpies"](https://en.wikipedia.org/wiki/5-over-1) - the cheapest possible "luxury" apartments where
wood-frame units sit on a concrete parking podium. The soundproofing is nonexistent because
[proper acoustic insulation costs money](https://www.foxblocks.com/blog/wood-frame-construction) developers won't
spend. The parking garage
waterproofing [inevitably fails](https://nusitegroup.com/podium-deck-restoration-extending-the-life-of-your-structure/),
water infiltrates, rebar corrodes
and [expands to 8x its size](https://westernspecialtycontractors.com/parking-structure-maintenance/),
and the whole structure starts crumbling - but by then they've already sold it. Wikipedia literally says
["it is unclear whether these buildings are built to last."](https://en.wikipedia.org/wiki/5-over-1) Charge absurd
rent, collect for a decade, then dump the property once it starts falling apart and move on to the next victim
neighborhood. Oh,
and [pay \$4 million to the NY Attorney General](https://ag.ny.gov/press-release/2022/attorney-general-james-secures-4-million-landlords-after-uncovering-illegal)
when your employees get caught taking \$1 million in kickbacks while fraudulently inflating renovation costs to
deregulate rent-stabilized units.
Get [sued for $50 million](https://therealdeal.com/new-york/2025/08/29/landlord-fights-sentinel-over-alleged-rent-regulation-fraud/)
for allegedly misrepresenting rent-stabilization status to pump property values. Settle a
[Fair Housing Act discrimination case](https://www.justice.gov/crt/case/sentinel-real-estate-corp-et-al-nd-ga) for
screwing over a disabled tenant. The BlackRock of slumlording. Good luck with that security deposit.

## Why This Exists

| Feature                                                          | This Project         | Aladdin                                                                                                                                                                                                     | Bloomberg    |
|------------------------------------------------------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| Cost                                                             | ~$10/mo in API calls | $25,000/month                                                                                                                                                                                               | $24,000/year |
| Twitter Sentiment                                                | Yes                  | Probably                                                                                                                                                                                                    | Kinda        |
| Will tell you "nothing matters today"                            | **Yes**              | Never. Gotta justify that invoice.                                                                                                                                                                          | Lol          |
| Open Source                                                      | Yes                  | Absolutely not                                                                                                                                                                                              | Cute         |
| Actually makes you a quant                                       | No                   | Also no                                                                                                                                                                                                     | Still no     |
| Got no-bid Fed contracts to buy its own ETFs with taxpayer money | No                   | [Yes](https://wallstreetonparade.com/2020/06/blackrock-is-bailing-out-its-etfs-with-fed-money-and-taxpayers-eating-losses-its-also-the-sole-manager-for-335-billion-of-federal-employees-retirement-funds/) | No           |
| Lost $5B+ in pension mandates because of ESG failures            | No                   | [Yes](https://www.fa-mag.com/news/blackrock-loses--5-9-billion-mandate-from-dutch-pension-pme-85220.html)                                                                                                   | No           |

*At least when we're wrong, it's free. When they're wrong, they get a bailout.*

## The Corruption Receipts

Since you made it this far, here's the full BlackRock starter pack:

- **Fed Chair Powell has $25M personally invested with BlackRock** while giving them no-bid
  contracts. ["Extremely carefully managed"](https://www.investing.com/news/economy/blackrock-conflicts-managed-extremely-carefully-feds-powell-says-2245953)
  he says. Sure
  bro. ([Source](https://wallstreetonparade.com/2020/05/fed-chair-powell-has-upwards-of-11-6-million-invested-with-blackrock-the-firm-that-will-manage-a-750-billion-corporate-bond-bailout-program-for-the-fed/))

- **BlackRock wrote the bailout plan before the crisis happened**, then got hired to implement it. They literally
  authored a paper called "Going Direct" in August 2019 proposing central banks inject money directly into the economy.
  Six months later, COVID happens, and they get the
  contract. ([Source](https://wallstreetonparade.com/2020/06/blackrock-authored-the-bailout-plan-before-there-was-a-crisis-now-its-been-hired-by-three-central-banks-to-implement-the-plan/))

- **55.8% of their funds underperform** their sector peers, with some pension funds returning -50.91% over 3 years while
  the sector average was positive. ([Source](https://www.yodelar.com/insights/blackrock-review))

- **Dutch pension funds pulled $5B+** from BlackRock because they're not acting in beneficiaries' best interests on
  climate
  risk. ([Source 1](https://www.fa-mag.com/news/blackrock-loses--5-9-billion-mandate-from-dutch-pension-pme-85220.html))

- **Larry Fink said "I'm ashamed of being part of this conversation"** about ESG at Aspen 2023, then denied saying it
  moments later in the same interview. After years of lecturing everyone about stakeholder
  capitalism. ([Source](https://www.axios.com/2023/06/26/larry-fink-ashamed-esg-weaponized-desantis))

- **Their own former Chief Investment Officer for Sustainable Investing** called their ESG investing "a dangerous
  placebo that harms the public interest" because ESG products have higher
  fees. ([Source](https://www.cnbc.com/2022/12/07/activist-investor-calls-for-blackrock-ceo-fink-to-step-down-over-esg-hypocrisy.html))

- **$11 billion in coal investments** while being the world's largest investor in coal-fired power stations, as of 2018.
  The Sierra Club literally started a campaign called "BlackRock's Big
  Problem." ([Source](https://en.wikipedia.org/wiki/BlackRock#Investments_in_contributors_to_climate_change)

- **They voted against management 1,500+ times for "insufficient diversity"** while simultaneously dropping ESG support
  from 47% to 4% when it became politically
  inconvenient. ([Source](https://fortune.com/2024/02/14/blackrock-voting-choice-ceo-larry-fink-shareholder-democracy-stakeholder-capitalism-esg/))

This project costs you maybe $10/month in API calls. Aladdin costs more per month than your rent, and the people running
it are doing all
of the above.

## Disclaimer

Not financial advice. If you YOLO your life savings because this said "HIGH signal" on some shitcoin, that's a you
problem. Hedge funds with actual Aladdin access still lose money - and then they get bailed out with your taxes.

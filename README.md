# Jafar - The villain to BlackRock's Aladdin.

BlackRock's Aladdin manages \$21 trillion dollars in assets while their CEO
has [private phone calls with the Fed Chair](https://wallstreetonparade.com/2020/08/fed-chair-powell-had-4-private-phone-calls-with-blackrocks-ceo-since-march-as-blackrock-manages-upwards-of-25-million-of-powells-personal-money-and-lands-3-no-bid-deals-with-the-fed/)
who conveniently has
[$25 million of his personal wealth invested with BlackRock](https://wallstreetonparade.com/2020/05/fed-chair-powell-has-upwards-of-11-6-million-invested-with-blackrock-the-firm-that-will-manage-a-750-billion-corporate-bond-bailout-program-for-the-fed/).
They charge god-knows-what per month for sentiment tools (they won't publish pricing because it would make defense
contractors blush). You have a mass-produced Roth IRA from Fidelity and a dream. Let's
fucking go.

Scrapes Twitter, finds what's actually trending in the consumer economy (not just fintwit), then unleashes an LLM
with tools to verify if people are lying. The LLM can check real stock prices, search the web, and dig through
historical data. When some Pepe avatar screams "SILVER MOONING!!!" and it's up 0.3%, we expose the grift.

**Key Philosophy**: Most days are boring. This system is calibrated to tell you when something *actually* matters, not
to manufacture urgency like Jim Cramer speed-running his thirteenth margin call of the week.

## What Does It Actually Look Like?

Unlike Aladdin, which requires you to sign seventeen NDAs, sacrifice a goat to their enterprise sales team, and sit
through a 90-minute "demo" that's really just a PowerPoint about their "proprietary AI" (it's a linear regression),
**we just show you the damn thing**.

**[View an actual digest](digest.pdf)**

That's a real output. Silver actually was up. The system caught it. It also caught some noise because Twitter is a
hellscape, but at least you can *see* what you're getting before you waste three weeks configuring YAML files. Yes,
I redacted my email address. No, I will not be doxxing myself to prove a point about open source transparency. The
difference is you can actually *get* a redacted sample from us. Try asking BlackRock for one. Their legal team will
get back to you sometime between "never" and "heat death of the universe."

## Features

- **Agentic Grift Detection**: LLM sees "SILVER MOONING!!!", calls `get_market_data("SI=F")`, discovers silver is up
  0.3%, and writes "EXAGGERATED" in the report. Someone needs to protect you from Crypto Twitter.
- **Deep Research**: Vague "uranium shortage" vibes? LLM searches the web for actual news. You get receipts.
- **Consumer Economy Scout**: Finds trends you didn't know to look for. RTX 5090 price hikes are economic signals
  even if CNBC isn't covering them yet.
- **Vector Memory**: Finds historical parallels. Catches fintwit grifters recycling the same thread every 6 months.
- **Skeptical by Default**: Explicitly instructed to say "nothing notable today". This would bankrupt CNBC.
- **Checkpoint System**: Got rate limited by Elon's clown show? Just run it again.
- **SOCKS5 Proxy Support**: For completely legitimate research purposes, officer.
- **Background Runner**: Start it, forget it, check logs whenever - like your Mandarin Duolingo streak.

## How It Works

1. **Scout** - Scrapes 30+ topics from Twitter. Not just fintwit bullshit. Consumer economy: "too expensive", "sold
   out", "can't afford". Because an RTX 5090 price hike tells you more about inflation than any CPI print.

2. **Investigator** - spaCy NLP extracts what people are actually talking about. Scores by engagement velocity and
   cashtag co-occurrence. Filters out the noise with a brutally honest LLM pre-filter that rejects "Christmas" and "
   Books" but keeps "$NVDA" and "shortage".

3. **Deep Dive** - Targeted scraping for the 2-4 trends that actually matter. Not 15. Not whatever has the most likes.
   The ones that pass the bullshit filter.

4. **The Agent** - Here's where it gets good. The LLM has tools:

   | Tool                                 | What It Does                                                     |
   |--------------------------------------|------------------------------------------------------------------|
   | `get_market_data(symbols)`           | Checks real prices. Exposes the "MOONING" liars.                 |
   | `search_web(query)`                  | Deep web research. Turns Twitter vibes into actual intelligence. |
   | `search_historical_parallels(query)` | "This feels familiar" → finds the receipts                       |
   | `get_trend_timeline(trend)`          | Is this new or recycled cope from last month?                    |
   | `get_weather_forecast(cities)`       | Panic buying in Houston? Checks if there's a storm coming.       |

   The agent sees a claim, gets suspicious, calls the tool. If someone says silver is crashing and it's down 0.5%,
   that's marked EXAGGERATED. If uranium is spiking and nobody knows why, it searches the web. Maximum 5 tool calls
   because we're not trying to burn $50 in API costs on some nobody's pump-and-dump.

5. **Reporter** - Emails you a digest so you can pretend you're a Bloomberg terminal owner without paying $24k/year.

6. **Memory** - Stores everything for future parallels. Next time someone tries the same grift, we remember.

## Installation

```bash
git clone https://github.com/Telesphoreo/Jafar
cd Jafar

uv sync
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

# Utilities
uv run test_email.py   # Verify SMTP settings and send test email
```

## Production Deployment (The Daemon Manifesto)

Running this on a VPS and don't want to babysit it like Larry Fink babysits his relationship with the Fed? Need it to
run
automatically without Cloudflare detecting your traffic pattern faster than BlackRock detects a new bailout opportunity?

**[Read DAEMONIZING.md](DAEMONIZING.md)** for the full systemd setup with randomized timing.

**TL;DR**: systemd timers with `RandomizedDelaySec` make your traffic look like a normal person with insomnia checking
fintwit
at random hours, instead of a cron job that screams "I'M A BOT" at 2 PM every day. Twice-daily randomized runs (
7am-12pm,
5pm-11pm windows) ensure you never wake up to a "Silver up 40%" Reuters alert like the normies while also not getting
cloudflared into oblivion. Includes automatic admin diagnostics emails so you know when your Twitter accounts get banned
before
you wonder why you haven't gotten a digest in 3 days. Because unlike Aladdin's monitoring dashboard (which probably
costs
$50k/month and requires a PhD to understand), we just email you when shit breaks.

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

**Rate limiting** - Add more accounts. Use proxies. Stop scraping during market hours like everyone else.

## Why This Exists

| Feature                                                          | This Project         | Aladdin                                                                                                                                                                                                     | Bloomberg               |
|------------------------------------------------------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| Cost                                                             | ~$10/mo in API calls | More than your rent                                                                                                                                                                                         | $24,000/year            |
| Twitter Sentiment                                                | Yes                  | Buried under 47 layers of enterprise middleware                                                                                                                                                             | Kinda, if you squint    |
| Will tell you "nothing matters today"                            | **Yes**              | No. Gotta manufacture urgency to justify the invoice.                                                                                                                                                       | Have you met Jim Cramer |
| LLM that checks if people are lying                              | **Yes**              | Probably a guy named Dave who went to Wharton                                                                                                                                                               | Still using RSS         |
| Open Source                                                      | Yes                  | Lmao                                                                                                                                                                                                        | That's adorable         |
| Actually makes you a quant                                       | No                   | Also no, but it costs more so you can pretend                                                                                                                                                               | Still no                |
| Got no-bid Fed contracts to buy its own ETFs with taxpayer money | No                   | [Yes](https://wallstreetonparade.com/2020/06/blackrock-is-bailing-out-its-etfs-with-fed-money-and-taxpayers-eating-losses-its-also-the-sole-manager-for-335-billion-of-federal-employees-retirement-funds/) | No                      |
| Lost $5B+ in pension mandates for being full of shit             | No                   | [Yes](https://www.fa-mag.com/news/blackrock-loses--5-9-billion-mandate-from-dutch-pension-pme-85220.html)                                                                                                   | No                      |
| CEO has called the Fed Chair while managing his personal money   | No                   | [Yes, "extremely carefully"](https://www.investing.com/news/economy/blackrock-conflicts-managed-extremely-carefully-feds-powell-says-2245953)                                                               | Probably not            |

*At least when we're wrong, it's free. When they're wrong, they get a bailout and a CNBC interview to explain why it was
actually your fault.*

## The Corruption Receipts

Since you made it this far, here's the full BlackRock starter pack. Print it out and tape it to your wall for when
someone tells you the system isn't rigged:

- **Fed Chair Powell has $25M personally invested with BlackRock** while giving them no-bid contracts to manage
  $750 billion in bailout
  money. ["Extremely carefully managed"](https://www.investing.com/news/economy/blackrock-conflicts-managed-extremely-carefully-feds-powell-says-2245953)
  he says, presumably with a straight face. Nothing to see here, just the guy who controls interest rates having his
  wealth managed by the company he's handing emergency contracts to. Totally
  normal. ([Source](https://wallstreetonparade.com/2020/05/fed-chair-powell-has-upwards-of-11-6-million-invested-with-blackrock-the-firm-that-will-manage-a-750-billion-corporate-bond-bailout-program-for-the-fed/))

- **BlackRock wrote the bailout plan before the crisis happened**, then got hired to implement it. In August 2019, they
  published a paper called "Going Direct" proposing central banks inject money directly into the economy. Six months
  later - oh wow what a coincidence - COVID happens and three central banks hire BlackRock to execute the exact plan
  they wrote. The universe works in mysterious ways if you're worth $10
  trillion. ([Source](https://wallstreetonparade.com/2020/06/blackrock-authored-the-bailout-plan-before-there-was-a-crisis-now-its-been-hired-by-three-central-banks-to-implement-the-plan/))

- **55.8% of their funds underperform** their benchmarks, according to Yodelar analysis. Some pension funds they manage
  returned -50.91% over 3 years while the sector average was positive. But hey, at least the fees are
  high. ([Source](https://www.yodelar.com/insights/blackrock-review))

- **Dutch pension funds pulled $5B+** from BlackRock because even the Netherlands - a country that will rent you a
  bicycle for literally anything - decided BlackRock wasn't acting in their beneficiaries' best interests on climate
  risk. When the Dutch think you're too greedy, you've achieved something
  special. ([Source](https://www.fa-mag.com/news/blackrock-loses--5-9-billion-mandate-from-dutch-pension-pme-85220.html))

- **Larry Fink said "I'm ashamed of being part of this conversation"** about ESG at Aspen 2023, then denied saying it
  moments later in the same interview. On video. That journalists were recording. After years of building his entire
  brand around stakeholder capitalism. The man contains
  multitudes. ([Source](https://www.axios.com/2023/06/26/larry-fink-ashamed-esg-weaponized-desantis))

- **Their own former Chief Investment Officer for Sustainable Investing** - the guy they literally hired to run ESG -
  quit and called the whole thing "a dangerous placebo that harms the public interest." Turns out ESG products just
  have higher fees. Who could have possibly
  predicted. ([Source](https://www.cnbc.com/2021/08/24/blackrocks-former-sustainable-investing-chief-says-esg-is-a-dangerous-placebo.html))

- **$11 billion in coal investments** while being the world's largest investor in coal-fired power stations, as of 2018.
  Meanwhile Larry's out here writing annual letters about climate responsibility. The Sierra Club literally started a
  campaign called "BlackRock's Big Problem" because sometimes you have to spell it out for
  people. ([Source](https://en.wikipedia.org/wiki/BlackRock#Investments_in_contributors_to_climate_change))

- **They voted against management 1,500+ times for "insufficient diversity"** while simultaneously dropping ESG support
  from 47% to 4% the moment Ron DeSantis made it politically inconvenient. Principles are for people who can't afford
  lobbyists. ([Source](https://fortune.com/2024/02/14/blackrock-voting-choice-ceo-larry-fink-shareholder-democracy-stakeholder-capitalism-esg/))

This project costs you maybe $10/month in API calls. Aladdin costs more per month than your rent, and the people running
it are doing all of the above.

## Disclaimer

Not financial advice. If you YOLO your life savings because this said "HIGH signal" on some shitcoin, that's a you
problem. Hedge funds with actual Aladdin access still lose money - and then they get bailed out with your taxes.

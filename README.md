# Jafar - The villain to BlackRock's Aladdin.

BlackRock's Aladdin manages \$21 trillion in assets. Their CEO
has [private phone calls with the Fed Chair](https://wallstreetonparade.com/2020/08/fed-chair-powell-had-4-private-phone-calls-with-blackrocks-ceo-since-march-as-blackrock-manages-upwards-of-25-million-of-powells-personal-money-and-lands-3-no-bid-deals-with-the-fed/)
who conveniently has
[\$25 million of his personal wealth invested with BlackRock](https://wallstreetonparade.com/2020/05/fed-chair-powell-has-upwards-of-11-6-million-invested-with-blackrock-the-firm-that-will-manage-a-750-billion-corporate-bond-bailout-program-for-the-fed/).
They won't publish pricing on their sentiment tools because the number would make Lockheed Martin blush. Their
employees post "grateful for the opportunity to drive impact in global markets" on LinkedIn from a company that
literally got no-bid Fed contracts to buy its own ETFs with your tax money. Hope the signing bonus was worth your
soul. The Morning Brew thinks slapping a rocket emoji next to "stocks go up" counts as analysis. Bloomberg charges
$24k/year for the privilege of a keyboard that looks like it was designed by someone who hates ergonomics and human
joy in equal measure. You have a mass-produced Roth IRA from Fidelity and a dream. Let's fucking go.

This thing scrapes Twitter, pulls real economic news, and hands it all to an LLM that has been specifically instructed
to assume everyone is lying until proven otherwise. The LLM has tools. It can look up real stock prices. It can search
the web. It can pull live headlines. It can dig through its own memory for historical parallels. It will use them
unprompted, like a paranoid research assistant who genuinely does not trust anyone. So when some Pepe avatar with 47
followers screams "SILVER MOONING!!!" and it's up 0.3%, we catch it, tag it EXAGGERATED, and move on with our lives.
When Reuters reports something that actually matters, you hear about it before your coworker who pays Bloomberg
\$24k/year to feel important at dinner parties. For $10/month. From your couch. In your underwear. While someone at
BlackRock is working their third consecutive 90-hour week to produce a report that says the same thing but with more
compliance disclaimers.

Most days are boring. That's the whole point. This system will tell you "nothing happened, go live your life" and
mean it, not manufacture urgency like Jim Cramer speed-running his thirteenth margin call of the week. But it won't
ghost you on quiet Twitter days either. You always get a news digest because the global economy doesn't stop
functioning just because fintwit is busy arguing about whether $PLTR is a meme stock or a defense contractor. It's
both. That's the problem.

## What does it actually look like?

Aladdin requires you to sign seventeen NDAs, sacrifice a goat to their enterprise sales team, and sit through a
90-minute "demo" that's really just a PowerPoint about their "proprietary AI" (it's a linear regression with a
marketing budget and a guy who says "machine learning" every 45 seconds because he learned it makes VPs nod).
We just show you the damn thing.

**[View an actual digest](digest.pdf)**

Real output. News roundup up top with actual economic headlines, then it descends into the Twitter analysis where
silver actually was up and the system caught it. Also caught some noise because Twitter is a hellscape, but at least
you can *see* what you're getting before you waste three weeks on YAML files. Yes, I redacted my email. No, I will not
be doxxing myself to prove a point about open source transparency. The difference between us and BlackRock is you can
actually *get* a sample. Try asking them for one. Their legal team will get back to you somewhere between "never" and
the heat death of the universe.

## Features

- **Daily economic newsletter** - Real economic news headlines delivered to your inbox even when Twitter is just people
  posting their Ls. The digest leads with actual journalism before descending into the Twitter trenches. Free version
  of The Morning Brew except angrier and not trying to sell you a fintech credit card every third paragraph.
- **Agentic grift detection** - LLM sees "SILVER MOONING!!!", calls `get_market_data("SI=F")`, discovers silver is up
  0.3%, and stamps EXAGGERATED on the report like a teacher returning a bad essay. Somebody has to protect you from
  Crypto Twitter. BlackRock charges six figures for this. We do it with an API call and a grudge.
- **Deep research** - Vague "uranium shortage" vibes on Twitter? The LLM doesn't just shrug and write "people are
  talking about uranium." It searches the web for actual news, finds the Kazakh supply disruption, and tells you why
  it matters. You get receipts, not retweets.
- **Consumer economy scout** - Finds trends you didn't know to look for. We search for "too expensive" and "sold
  out", not "\$NVDA." An RTX 5090 hitting $5,000 on secondary markets is an economic signal that tells you more about
  consumer demand and pricing power than any earnings call ever will. CNBC won't cover it for another two weeks. You'll
  know now.
- **The mentions tier** - Trends that fail the signal filter don't vanish anymore. They get a "Twitter is also
  mumbling about..." section so you know what the discourse is even when it's not worth a full writeup. Police scanner
  for financial delusion.
- **Vector memory** - Remembers every past digest and finds historical parallels. Catches fintwit grifters recycling
  the same thread every six months with a new profile picture like we wouldn't notice. We remember. They don't know
  we remember. We are the elephant in a room full of goldfish.
- **Skeptical by default** - Explicitly instructed to say "nothing happened" for Twitter signals. You still get your
  news roundup because we're not monsters. This distinction alone would put CNBC out of business. They'd rather
  manufacture a crisis about the yield curve than sit in silence for thirty seconds.
- **Checkpoint system** - Got rate limited by Elon's clown show? Run it again. It picks up where it left off.
- **SOCKS5 proxy support** - For completely legitimate research purposes, officer.
- **Background runner** - Start it, forget it, check logs whenever. Like your Mandarin Duolingo streak but this one
  actually does something useful.

## How it works

Eleven steps. Does more before your morning coffee than most hedge fund interns do all week. Every single one of
these steps would be a separate product at a fintech startup with $40M in Series B funding and a ping pong table.

1. **Scout** - Scrapes 30+ topics from Twitter. Not just fintwit. Consumer economy: "too expensive", "sold out",
   "can't afford", "shrinkflation." The stuff real people say when their grocery bill hits $200 for the third week in
   a row. An RTX 5090 price hike tells you more about inflation than any CPI print ever will, and we don't need a
   Bloomberg terminal to find it. We need a Twitter account, a VPS, and the kind of audacity that gets you
   blacklisted from career fairs.

2. **Investigator** - spaCy NLP extracts what people are actually talking about. Engagement velocity. Cashtag
   co-occurrence. Math, not vibes. Your portfolio deserves better than "I saw it trending."

3. **LLM filter** - Pre-filter that separates signal from noise with the cold efficiency of a Goldman layoff round.
   The kind of layoff where they walk you out before your coffee gets cold and your badge stops working by the time
   you reach the lobby. Rejects "Christmas" and "Books", keeps "$NVDA" and "shortage". Rejected trends get demoted to
   the mentions tier because even Twitter's rejected noise occasionally has a kernel of truth under seventeen layers
   of cope.

4. **Deep dive** - Targeted scraping for the trends that actually matter. Not fifteen of them. Not whatever got the
   most likes. The ones that survived the bullshit filter.

5. **News roundup** - Fetches real economic news headlines via DuckDuckGo because the global economy generates news
   even when Twitter is just people ratio'ing each other about whether the recession is transitory. Reuters, Bloomberg,
   AP, whatever DuckDuckGo surfaces that isn't a SEO farm in a trench coat pretending to be journalism. Runs
   independently of Twitter. If every Twitter account on earth gets banned tomorrow (honestly, give it time), you
   still get your briefing. BlackRock's Aladdin probably has a Bloomberg terminal and a Refinitiv subscription that
   cost more per month than your car is worth. We built the same thing with DuckDuckGo, a free search API, and the
   unshakeable conviction that information about the economy shouldn't cost more than a mortgage payment.

6. **Fact checker** - Verifies market claims against real-time data. Trust but verify, minus the trust. Someone
   tweets "oil is collapsing" and crude is down 0.8%? That's not collapsing. That's a Tuesday. We will tell you it's
   a Tuesday.

7. **Temporal analyzer** - Tracks trends across days. Is this Day 5 of "egg shortage" discourse or did someone just
   discover grocery stores? Flags developing stories. Catches recurring grifts. Has better pattern recognition than the
   SEC, which is a bar so low you'd need a shovel to find it.

8. **The agent** - Here's where it gets good. The LLM gets the news headlines, the Twitter data, and a toolkit:

   | Tool                                 | What it does                                                      |
   |--------------------------------------|-------------------------------------------------------------------|
   | `get_market_data(symbols)`           | Checks real prices. Exposes the "MOONING" liars.                  |
   | `search_web(query)`                  | Web research. Turns Twitter vibes into actual intelligence.       |
   | `fetch_news(query)`                  | Pulls live news. For when Twitter isn't enough (always).          |
   | `search_historical_parallels(query)` | "This feels familiar" - finds the receipts.                       |
   | `get_trend_timeline(trend)`          | New trend or recycled cope from last month?                       |
   | `get_weather_forecast(cities)`       | Panic buying in Houston? Checks if there's actually a storm.      |

   It sees a claim, gets suspicious, calls a tool. Someone says silver is crashing and it's down 0.5%? EXAGGERATED.
   Uranium spiking and nobody knows why? Searches the web. Five tool calls max because we're not burning $50 in API
   costs investigating some nobody's pump-and-dump scheme that exists solely to exit-liquidity their 47 followers.
   Writes you a news roundup AND a Twitter analysis because you deserve both without opening six apps and a Bloomberg
   terminal you definitely don't have and honestly shouldn't want.

9. **Reporter** - Emails you a formatted digest so you can pretend you have a Bloomberg terminal without paying
   $24k/year or learning what any of the 30,000 Bloomberg keyboard shortcuts do. News roundup first, Twitter analysis
   second, fact checks third. Shows exactly how many headlines and tweets went into it because we believe in
   transparency, a concept BlackRock treats like a foreign language they pretend to speak at Davos.

10. **Memory** - Stores every digest for future parallels. Next time someone tries the same grift, we pull up exactly
    when they tried it last time, what they said, and how it went (badly). The SEC has the institutional memory of a
    concussed goldfish with a $2.2 billion annual budget. We have a SQLite database. Guess which one catches more
    repeat offenders.

11. **Admin diagnostics** - Emails you when something breaks. Plain English. Not a $50k/month dashboard that requires
    a PhD to read. Just "hey, your Twitter account got banned again."

## Installation

```bash
git clone https://github.com/Telesphoreo/Jafar
cd Jafar

uv sync
```

That's it. Two commands. `uv` handles the dependencies, installs the spaCy model, the whole thing. Done before
Aladdin's enterprise sales rep finishes typing their "just circling back on this" follow-up email. Their sales cycle
is longer than most marriages and ends the same way: expensive, disappointing, and someone's getting a lawyer.

## Configuration

Copy `config.yaml.example` to `config.yaml` and `.env.example` to `.env`.

Figure it out. The examples are commented better than most codebases are documented. This is literally the easiest
part. You're about to run a market intelligence system that competes with software that costs more than a house in
most zip codes, and the barrier to entry is editing two text files. If you can't clear that bar, honestly, just buy
VOO and go live your life. There is no shame in index funds. There is shame in paying BlackRock's fees while
[their own executive calls ESG a "dangerous placebo that harms the public interest"](https://www.cnbc.com/2022/12/07/activist-investor-calls-for-blackrock-ceo-fink-to-step-down-over-esg-hypocrisy.html),
but that's a different kind of shame, and the people who should feel it never do.

## Twitter setup

Cookie auth because Elon broke the API, then charged $42,000/month for the privilege of using what's left of it:

1. Log into Twitter in your browser
2. Export cookies with the shadiest browser extension you can find
3. Save as `cookies.json`
4. Run: `uv run python add_account.py <username> cookies.json`

The `add_account.py` script handles all the proxy assignment automatically. You define your proxies in `config.yaml`,
it round-robins them across your accounts so each one gets a consistent IP. No need to wrestle with twscrape's
interface or keep a spreadsheet of which proxy goes where like some kind of deranged accountant. We automated the
annoying part because life is short and proxy management isn't how anyone should be spending it.

More accounts = more parallel workers = faster scraping. Not higher limits. More lanes on the highway, not a higher
speed limit.

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

## Production deployment (The Daemon Manifesto)

Running this on a VPS and don't want to babysit it like Larry Fink babysits his relationship with the Fed Chair?
Need it to run automatically without Cloudflare clocking your traffic pattern faster than BlackRock clocks a fresh
bailout opportunity? Cool. Same.

**[Read DAEMONIZING.md](DAEMONIZING.md)** for the full systemd setup with randomized timing.

Short version: systemd timers with `RandomizedDelaySec` make your scraping look like a normal person with insomnia
doomscrolling fintwit at weird hours, not a cron job that screams "I'M A BOT" at exactly 2:00 PM every day.
Twice-daily randomized runs (7am-12pm, 5pm-11pm windows) so you never wake up to a "Silver up 40%" Reuters alert like
a civilian. Automatic admin diagnostics emails tell you when your Twitter accounts get banned before you spend three
days wondering why your inbox is empty. Because Aladdin's monitoring dashboard probably costs $50k/month and requires
a PhD to interpret. Ours just emails you when shit breaks.

## Signal strength

| Level      | Meaning                 | Frequency                 | You still get a digest? |
|------------|-------------------------|---------------------------|-------------------------|
| **HIGH**   | Actually unusual. Rare. | 1-2x per month            | Obviously, and read it now |
| **MEDIUM** | Worth watching          | Weekly                    | Yes                     |
| **LOW**    | Normal Twitter noise    | Most days                 | Yes                     |
| **NONE**   | Twitter had nothing     | When everyone's at brunch | **Yes** - news roundup still hits |

Signal strength measures Twitter activity, not whether the digest matters. NONE days still get a full news roundup.
The economy doesn't pause because Twitter took a nap. CNBC hasn't figured this out yet, which is why they fill dead
air screaming "IS THIS THE NEXT 2008?" every time the S&P dips 0.4%. Aladdin hasn't figured this out either, but they
charge you $200k/year to not understand it, so at least it feels exclusive.

We also don't [hold $11 billion in coal investments while lecturing people about climate](https://en.wikipedia.org/wiki/BlackRock#Investments_in_contributors_to_climate_change).

## Fact check classifications

| Tag              | Meaning                                     |
|------------------|---------------------------------------------|
| **VERIFIED**     | They told the truth. Mark your calendars.   |
| **EXAGGERATED**  | Directionally correct, emotionally unhinged |
| **FALSE**        | Lying on the internet. Groundbreaking.      |
| **UNVERIFIABLE** | Made up a ticker or was too vague to check  |

## Production (pgvector)

Running on a VPS with PostgreSQL:

```yaml
# config.yaml
memory:
  store_type: pgvector
  embedding_dimensions: 1536  # pgvector maxes at 2000 dims
```

Create the extension: `CREATE EXTENSION vector;`

Dimension errors? Drop the table. Let it rebuild.

```sql
DROP TABLE IF EXISTS market_memories;
```

## Troubleshooting

**"No tweets retrieved"** - Your accounts are logged out or banned. `uv run twscrape accounts` to check. Re-add via
cookies. Welcome to the cat-and-mouse game with whatever Elon's platform is called this week.

**"Failed to initialize vector memory: 2000 dimensions"** - Add `embedding_dimensions: 1536` to config. Drop the
table. PostgreSQL has limits. So do we all.

**Rate limiting** - Add more accounts. Use proxies. Consider not scraping during market hours like an absolute maniac.

## Why this exists

| Feature                                                          | This project         | Aladdin                                                                                                                                                                                                     | Bloomberg                          | The Morning Brew                  |
|------------------------------------------------------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|-----------------------------------|
| Cost                                                             | ~$10/mo in API calls | More than your rent                                                                                                                                                                                         | $24,000/year                       | Free (you are the product)        |
| Daily economic news digest                                       | **Yes**              | Yes, for the price of a used Toyota                                                                                                                                                                         | Yes, for the price of a new Toyota | Yes, with 40 affiliate links      |
| Twitter sentiment                                                | **Yes**              | Buried under 47 layers of enterprise middleware                                                                                                                                                             | Kinda, if you squint               | They have a TikTok guy            |
| Will tell you "nothing matters today"                            | **Yes**              | No. Gotta manufacture urgency to justify the invoice.                                                                                                                                                       | Have you met Jim Cramer            | Every day is "HUGE"               |
| LLM that checks if people are lying                              | **Yes**              | Probably a guy named Dave who went to Wharton                                                                                                                                                               | Still using RSS feeds              | Their intern fact-checks by vibes |
| Mentions what Twitter chatter even when it's noise               | **Yes**              | Their sentiment feed costs extra                                                                                                                                                                            | $500/mo add-on                     | They screenshot tweets            |
| Open source                                                      | Yes                  | Lmao                                                                                                                                                                                                        | That's adorable                    | Their "tech" is Mailchimp         |
| Actually makes you a quant                                       | No                   | Also no, but it costs more so you feel like one                                                                                                                                                             | Still no                           | Makes you think you are           |
| Got no-bid Fed contracts to buy its own ETFs with taxpayer money | No                   | [Yes](https://wallstreetonparade.com/2020/06/blackrock-is-bailing-out-its-etfs-with-fed-money-and-taxpayers-eating-losses-its-also-the-sole-manager-for-335-billion-of-federal-employees-retirement-funds/) | No                                 | No                                |
| Lost $5B+ in pension mandates for being full of shit             | No                   | [Yes](https://www.fa-mag.com/news/blackrock-loses--5-9-billion-mandate-from-dutch-pension-pme-85220.html)                                                                                                   | No                                 | No                                |
| CEO has called the Fed Chair while managing his personal money   | No                   | [Yes, "extremely carefully"](https://www.investing.com/news/economy/blackrock-conflicts-managed-extremely-carefully-feds-powell-says-2245953)                                                               | Probably not                       | Their CEO calls his mom           |

When we're wrong, it's free. When they're wrong, they get a bailout, a CNBC interview to explain why it was actually
your fault, and a LinkedIn post from their junior analyst about "lessons learned in volatile markets" that gets 4,000
likes from other people who also lost your money.

## The corruption receipts

You made it this far, so either you're interested or you work at BlackRock and you're hate-reading this during your
lunch break at your standing desk that cost more than this entire project's annual operating budget. Either way,
here's the full starter pack. Every bullet is sourced. Every link works. Print it out and tape it to your monitor for
the next time someone at a cocktail party tells you the system works as intended:

- **Fed Chair Powell has $25M personally invested with BlackRock** while handing them no-bid contracts to manage \$750
  billion in bailout
  money. ["Extremely carefully managed"](https://www.investing.com/news/economy/blackrock-conflicts-managed-extremely-carefully-feds-powell-says-2245953),
  he says. The man who controls interest rates has his personal wealth managed by the company he's giving emergency
  contracts to. Just a completely normal thing that requires no further
  scrutiny. ([Source](https://wallstreetonparade.com/2020/05/fed-chair-powell-has-upwards-of-11-6-million-invested-with-blackrock-the-firm-that-will-manage-a-750-billion-corporate-bond-bailout-program-for-the-fed/))

- **BlackRock wrote the bailout playbook before the crisis existed.** August 2019, they publish a paper called
  "Going Direct" proposing that central banks inject money straight into the economy. Six months later, COVID hits
  and three central banks hire BlackRock to execute the exact plan they authored. What are the odds. What are the
  absolute
  odds. ([Source](https://wallstreetonparade.com/2020/06/blackrock-authored-the-bailout-plan-before-there-was-a-crisis-now-its-been-hired-by-three-central-banks-to-implement-the-plan/))

- **55.8% of their funds underperform their benchmarks.** Yodelar found that some pension funds BlackRock manages
  returned -50.91% over three years while the sector average was positive. Negative fifty percent. The sector was
  green. They were red. But the fees were
  collected. ([Source](https://www.yodelar.com/insights/blackrock-review))

- **Dutch pension funds pulled $5.9 billion** because even the Netherlands decided BlackRock wasn't acting in their
  beneficiaries' best interests. The Dutch will rent you a bicycle for literally anything. They are the most
  agreeable people in Europe. And they looked at BlackRock's climate record and said no. When the Dutch think you're
  too greedy, you've accomplished something
  remarkable. ([Source](https://www.fa-mag.com/news/blackrock-loses--5-9-billion-mandate-from-dutch-pension-pme-85220.html))

- **Larry Fink said "I'm ashamed of being part of this conversation"** about ESG at Aspen 2023. Then denied saying
  it. In the same interview. On camera. That journalists were recording. After building his entire personal brand
  around stakeholder capitalism for half a decade. Said it, denied it, on tape, in public. The man is
  art. ([Source](https://www.axios.com/2023/06/26/larry-fink-ashamed-esg-weaponized-desantis))

- **Their own former Chief Investment Officer for Sustainable Investing** quit and called the whole ESG operation "a
  dangerous placebo that harms the public interest." This is the guy they hired to run ESG. Their guy. He left and
  told everyone it was a fee-extraction scheme. Turns out the sustainable investment products were just regular
  products with higher fees.
  Shocking. ([Source](https://www.cnbc.com/2021/08/24/blackrocks-former-sustainable-investing-chief-says-esg-is-a-dangerous-placebo.html))

- **$11 billion in coal investments** while being the world's largest investor in coal-fired power stations. Larry
  writes annual letters about climate responsibility with one hand and signs coal investment memos with the other. The
  Sierra Club started a campaign called "BlackRock's Big Problem" because sometimes you have to use small words for
  people who manage \$10
  trillion. ([Source](https://en.wikipedia.org/wiki/BlackRock#Investments_in_contributors_to_climate_change))

- **Dropped ESG shareholder support from 47% to 4%** the exact moment Ron DeSantis made it politically inconvenient.
  Voted against management 1,500+ times for "insufficient diversity" right up until it stopped being good PR.
  Principles are for people who can't afford
  lobbyists. ([Source](https://fortune.com/2024/02/14/blackrock-voting-choice-ceo-larry-fink-shareholder-democracy-stakeholder-capitalism-esg/))

This project costs maybe $10/month in API calls. Aladdin costs more per month than your rent, and the people running
it are doing all of the above while posting "integrity is our north star" in their company Slack.

## Disclaimer

Not financial advice. Not even close. Not financial advice in the same way that a weather report is not a personal
guarantee that it won't rain on you specifically. If you YOLO your life savings because this said "HIGH signal" on
some shitcoin, that is a you problem and we will not be taking questions, interviews, or depositions. Hedge funds
with actual Aladdin access lose money all the time. The difference is they get bailed out with your taxes and then
go on CNBC to explain why it was actually your fault for not being diversified enough across their seventeen
underperforming products. Then they post a LinkedIn carousel about "resilience in uncertain times" with a headshot
where they're wearing a Patagonia vest and smiling like they didn't just vaporize a pension fund. This system pulls
publicly available news and tweets, feeds them to an LLM that is constitutionally incapable of feeling FOMO, and
emails you a summary. If that destroys your portfolio, your portfolio was already on life support and this just read
it its last rites. Our bad takes are free. BlackRock's bad takes come with a management fee and a 40-page shareholder
letter about "navigating uncertainty" that manages to be longer than most dissertations while saying less than a
fortune cookie.

In January 2021, a bunch of people on Reddit with Robinhood accounts and zero institutional backing almost
bankrupted a $13 billion hedge fund because they liked a stock. Melvin Capital needed a $2.75 billion emergency
bailout and closed permanently a year later. The entire financial establishment lost its mind. Congress held hearings.
Billionaires went on TV and cried. All because regular people had access to the same information at the same time and
acted on it before the suits could. This is that energy, but for economic intelligence. We gave you the tools. Go.

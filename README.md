# Jafar

### The villain to BlackRock's Aladdin.

BlackRock's Aladdin manages \$21 trillion in assets. Their CEO
has [private phone calls with the Fed Chair](https://wallstreetonparade.com/2020/08/fed-chair-powell-had-4-private-phone-calls-with-blackrocks-ceo-since-march-as-blackrock-manages-upwards-of-25-million-of-powells-personal-money-and-lands-3-no-bid-deals-with-the-fed/)
who conveniently has
[\$25 million of his personal wealth invested with BlackRock](https://wallstreetonparade.com/2020/05/fed-chair-powell-has-upwards-of-11-6-million-invested-with-blackrock-the-firm-that-will-manage-a-750-billion-corporate-bond-bailout-program-for-the-fed/).
They won't publish pricing on their sentiment tools because the number would make Lockheed Martin blush. Their
employees post "grateful for the opportunity to drive impact in global markets" on LinkedIn from a company that
literally got no-bid Fed contracts to buy its own ETFs with your tax money. Hope the signing bonus was worth your
soul. (It wasn't. You know it wasn't. That's why you're still scrolling LinkedIn at midnight looking for validation
from strangers who peaked in their MBA cohort.) The Morning Brew thinks slapping a rocket emoji next to "stocks go up"
counts as analysis. Bloomberg charges
\$24k/year for the privilege of a keyboard that looks like it was designed by someone who hates ergonomics and human
joy in equal measure. You have a mass-produced Roth IRA from Fidelity and a dream. Let's fucking go.

This thing scrapes Twitter, stores every tweet to PostgreSQL, and hands it to an LLM that has been specifically
instructed to assume everyone is lying until proven otherwise. The LLM has tools. It can look up real stock prices.
It can search the web. It can pull live headlines. It can dig through its own memory for historical parallels. It
will use them unprompted, like a paranoid research assistant who genuinely does not trust anyone. So when some Pepe
avatar with 47 followers screams "SILVER MOONING!!!" and it's up 0.3%, we catch it, tag it EXAGGERATED, and move on
with our lives.

Meanwhile, all those stored tweets become training data. Open the ML dashboard, run garbage detection, let
IsolationForest surface the spam bots and follower farms. Run LLM-as-Judge to classify accounts as signal or garbage.
Review the uncertain ones yourself. Block the garbage, watch the signal. Next pipeline run, those blocked accounts
never touch the LLM. Those watched accounts get scraped first. The system learns. For \$10/month. From your couch. In
your underwear. While some Aladdin engineer is working their third consecutive 90-hour week, mass-producing the same
fucking report you just got for free, wondering if "Senior Vice President" will finally fill the void where their
twenties used to be. It won't. But the signing bonus already cleared and the golden handcuffs don't come off until
2027, so here they are, 3am on a Tuesday, refactoring a risk model that exists to make Larry Fink's portfolio
decisions look like math instead of vibes.

Most days are boring. That's the whole point. This system will tell you "nothing happened, go live your life" and
mean it, not manufacture urgency like Jim Cramer speed-running his thirteenth margin call of the week.

## What does it actually look like?

Aladdin requires you to sign seventeen NDAs, sacrifice a goat to their enterprise sales team, and sit through a
90-minute "demo" that's really just a PowerPoint about their "proprietary AI" (it's a linear regression with a
marketing budget and a guy who says "machine learning" every 45 seconds because he learned it makes VPs nod).
We just show you the damn thing.

**[View an actual digest](digest.pdf)**

Real output. Twitter analysis where silver actually was up and the system caught it. Also caught some noise because
Twitter is a hellscape, but at least you can *see* what you're getting before you waste three weeks on YAML files.
Yes, I redacted my email. No, I will not be doxxing myself to prove a point about open source transparency. The
difference between us and BlackRock is you can actually *get* a sample. Try asking them for one. Their legal team
will get back to you somewhere between "never" and the heat death of the universe.

## Features

- **Consumer economy scout** - Finds trends you didn't know to look for. We search for "too expensive" and "sold
  out", not "\$NVDA." An RTX 5090 hitting \$5,000 on secondary markets is an economic signal that tells you more about
  consumer demand and pricing power than any earnings call ever will. CNBC won't cover it for another two weeks. You'll
  know now.
- **Agentic grift detection** - LLM sees "SILVER MOONING!!!", calls `get_market_data("SI=F")`, discovers silver is up
  0.3%, and stamps EXAGGERATED on the report like a teacher returning a bad essay. Somebody has to protect you from
  Crypto Twitter. BlackRock charges six figures for this. We do it with an API call and a grudge.
- **ML garbage detection** - Every tweet gets stored. Run IsolationForest from the dashboard whenever you want. 16
  behavioral features: posting frequency, engagement ratios, duplicate content, suspicious timing patterns. If an
  account tweets like it was born in a server rack, we notice. We don't say "bot detection" because @DeItaone is
  automated and invaluable. We say "garbage detection" because what we're filtering is worthless spam, not automation.
  Block the garbage accounts, and future pipeline runs filter them out automatically.
- **LLM-as-Judge** - Run it from the dashboard. Gemini reviews actual tweets and classifies accounts. Not "is this a
  bot?" but "would this account's tweets add signal or noise to economic analysis?" Grifters with hot takes? Signal.
  Soulless crypto pump spam nobody reads? Garbage. 80%+ confidence auto-applies. Uncertain accounts go to your HITL
  review queue.
- **Deep research** - Vague "uranium shortage" vibes on Twitter? The LLM doesn't just shrug and write "people are
  talking about uranium." It searches the web for actual news, finds the Kazakh supply disruption, and tells you why
  it matters. You get receipts, not retweets.
- **The mentions tier** - Trends that fail the signal filter don't vanish anymore. They get a "Twitter is also
  mumbling about..." section so you know what the discourse is even when it's not worth a full writeup. Police scanner
  for financial delusion.
- **Vector memory** - Remembers every past digest and finds historical parallels. Catches fintwit grifters recycling
  the same thread every six months with a new profile picture like we wouldn't notice. We remember. They don't know
  we remember. We are the elephant in a room full of goldfish.
- **Skeptical by default** - Explicitly instructed to say "nothing happened" when nothing happened. This distinction
  alone would put CNBC out of business. They'd rather manufacture a crisis about the yield curve than sit in silence
  for thirty seconds.
- **ML dashboard** - A full control center with a Windows Aero aesthetic and a green glass progress bar. Run it with
  `uv run dashboard`. Bloomberg's terminal hasn't been redesigned since the Cold War. Ours has a shimmer animation.
- **Checkpoint system** - Got rate limited by Elon's clown show? Run it again. It picks up where it left off.
- **SOCKS5 proxy support** - For completely legitimate research purposes, officer.
- **Background runner** - Start it, forget it, check logs whenever. Like your Mandarin Duolingo streak but this one
  actually does something useful.

## How it works

Ten steps. Does more before you've finished your Red Bull than most hedge fund interns do all week. They're still
waiting for their \$4,000 La Marzocco to heat up, arranging their notebooks just so, and "getting settled" for three
hours before opening a terminal. You chugged a Monster at 6am because you have a life outside of performative
productivity theater. Every single one of these steps would be a separate product at a fintech startup with \$40M in
Series B funding and a ping pong table.

1. **Scout** - Scrapes 30+ topics from Twitter. Not just fintwit. Consumer economy: "too expensive", "sold out",
   "can't afford", "shrinkflation." The stuff real people say when their grocery bill hits \$200 for the third week in
   a row. An RTX 5090 price hike tells you more about inflation than any CPI print ever will, and we don't need a
   Bloomberg terminal to find it. We need a Twitter account, a VPS, and the kind of audacity that gets you blacklisted
   from career fairs. Also fetches trending topics from the platform itself and scrapes your watched accounts from the
   ML dashboard.

2. **Garbage filter** - Tweets from blocked accounts never touch the LLM. If the ML dashboard flagged
   @CryptoSpamBot9000 as garbage, their tweets are filtered out before analysis. Saves tokens. Saves dignity.

3. **Investigator** - spaCy NLP extracts what people are actually talking about. Engagement velocity. Entity
   co-occurrence. Math, not vibes. Your portfolio deserves better than "I saw it trending."

4. **LLM filter** - Pre-filter that separates signal from noise with the cold efficiency of a Goldman layoff round.
   The kind of layoff where they walk you out before your coffee gets cold and your badge stops working by the time
   you reach the lobby. Rejects "Christmas" and "Books", keeps "\$NVDA" and "shortage". Rejected trends get demoted to
   the mentions tier because even Twitter's rejected noise occasionally has a kernel of truth under seventeen layers
   of cope.

5. **Deep dive** - Targeted scraping for the trends that actually matter. Not fifteen of them. Not whatever got the
   most likes. The ones that survived the bullshit filter.

6. **Fact checker** - Verifies market claims against real-time yfinance data. Trust but verify, minus the trust.
   Someone tweets "oil is collapsing" and crude is down 0.8%? That's not collapsing. That's a Tuesday. We will tell
   you it's a Tuesday.

7. **Temporal analyzer** - Tracks trends across days. Is this Day 5 of "egg shortage" discourse or did someone just
   discover grocery stores? Flags developing stories. Catches recurring grifts. Has better pattern recognition than the
   SEC, which is a bar so low you'd need a shovel to find it.

8. **The agent** - Here's where it gets good. The LLM gets the Twitter data and a toolkit:

   | Tool                                 | What it does                                                 |
   |--------------------------------------|--------------------------------------------------------------|
   | `get_market_data(symbols)`           | Checks real prices. Exposes the "MOONING" liars.             |
   | `search_web(query)`                  | Web research. Turns Twitter vibes into actual intelligence.  |
   | `fetch_news(query)`                  | Pulls live news headlines. Deep research on stories.         |
   | `search_historical_parallels(query)` | "This feels familiar" - finds the receipts.                  |
   | `get_trend_timeline(trend)`          | New trend or recycled cope from last month?                  |
   | `get_weather_forecast(cities)`       | Panic buying in Houston? Checks if there's actually a storm. |

   It sees a claim, gets suspicious, calls a tool. Someone says silver is crashing and it's down 0.5%? EXAGGERATED.
   Uranium spiking and nobody knows why? Searches the web. Five tool calls max because we're not burning \$50 in API
   costs investigating some nobody's pump-and-dump scheme that exists solely to exit-liquidity their 47 followers.

9. **Reporter** - Emails you a formatted digest so you can pretend you have a Bloomberg terminal without paying
   \$24k/year or learning what any of the 30,000 Bloomberg keyboard shortcuts do. Signal strength rating, fact checks,
   historical parallels. Shows exactly how many tweets went into it because we believe in transparency, a concept
   BlackRock treats like a foreign language they pretend to speak at Davos.

10. **Memory** - Stores every digest for future parallels. Next time someone tries the same grift, we pull up exactly
    when they tried it last time, what they said, and how it went (badly). The SEC has the institutional memory of a
    concussed goldfish with a \$2.2 billion annual budget. We have a PostgreSQL database. Guess which one catches more
    repeat offenders.

## ML Dashboard

```bash
uv run dashboard
```

That's it. You now have a control center that Bloomberg charges \$25,000/yr to approximate poorly.

The pipeline stores tweets. The dashboard is where you do things with them. Run ML scoring whenever you want. Review
accounts. Build your watch and block lists. The pipeline reads those lists on the next run. Two processes, one
database, zero coordination required. Aladdin needs a deployment team. We need `uv run`.

The dashboard is a Flask app with a UI aesthetic that would make Windows Vista shed a single tear of pride. Dark chrome
navbar with an amber glow line. Brushed metal card headers. A progress bar that is a 1:1 recreation of the Windows 7
progress bar, green glass gradient and all, shimmer included, because if you're going to build a dashboard you should
build one with a soul. No Tailwind. No shadcn. No Lucide icons. No "modern minimal" that looks like every other
ChatGPT-generated SaaS landing page that will pivot to AI agents next quarter. We went full Aero.

**ML Operations** (run these from the dashboard whenever you want):

- **Garbage detection.** IsolationForest across 16 behavioral features. Posting intervals, engagement ratios, duplicate
  content, night posting patterns. Click "Run Scoring" and it trains on all stored tweets. If an account tweets like
  it came off an assembly line, we notice.
- **Signal scoring.** This is the good shit. Percentile-ranked composite with KMeans clustering. Finds the retail
  workers posting about empty shelves, the truck drivers mentioning freight slowdowns, the semiconductor fabs tweeting
  about supply issues. Real people in real industries posting real observations BEFORE Reuters picks it up. Not the
  blue checkmarks posting engagement bait. Not the finance bros with 50k followers recycling the same thread. The
  actual sources.
- **LLM-as-Judge.** Click "Run LLM Judge" and Gemini reads actual tweets to classify accounts. Not "is this a bot?"
  but "does this account produce signal or garbage?" @DeItaone is automated and it's one of the most valuable feeds
  on Twitter. We want hot takes, even bad ones. We don't want soulless spam nobody reads. 80%+ confidence auto-applies
  labels. Uncertain accounts go to your review queue.

**Account Management** (these feed back into the pipeline):

- **HITL review.** Not HTML. HITL. Human in the loop. Three buttons: signal, garbage, unsure. Simple enough that
  even an MBA wouldn't need a walkthrough. "Unsure" comes back when new tweets arrive. And if you don't want to be in
  the loop? Remove the confidence threshold and let the LLMs run the whole thing. Nobody's stopping you.
- **Watch list.** High-signal accounts get scraped every pipeline run. Alternative data without the terabytes of
  satellite imagery and the CNN that thinks tree shadows are Camrys. Ask me how I fucking know. I've trained those
  models. I've stared at parking lot segmentation masks at 3am wondering if that blob is a truck or a dumpster. This
  is better.
- **Block list.** Crypto spam, follower farms, "Bullish" reply bots. Blocked forever. Their tweets get filtered out
  before analysis. They don't deserve the compute.
- **Search.** Type a username, hit enter. Bloomberg requires ticker, EQUITY, GO. Three keystrokes. Ours is one.

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

Copy `config.example.yaml` to `config.yaml` and `.env.example` to `.env`.

Figure it out. The examples are commented better than most codebases are documented. This is literally the easiest
part. You're about to run a market intelligence system that competes with software that costs more than a house in
most zip codes, and the barrier to entry is editing two text files. If you can't clear that bar, honestly, just buy
VOO and go live your life. There is no shame in index funds. There is shame in paying BlackRock's fees while
[their own executive calls ESG a "dangerous placebo that harms the public interest"](https://www.cnbc.com/2022/12/07/activist-investor-calls-for-blackrock-ceo-fink-to-step-down-over-esg-hypocrisy.html),
but that's a different kind of shame, and the people who should feel it never do.

## Twitter setup

Cookie auth because Elon broke the API, then charged \$42,000/month for the privilege of using what's left of it:

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
# Run the pipeline (scrape -> analyze -> email digest)
uv run jafar

# Open the ML dashboard (scoring, LLM judge, account review)
uv run dashboard

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
days wondering why your inbox is empty. Because Aladdin's monitoring dashboard probably costs \$50k/month and requires
a PhD to interpret. Ours just emails you when shit breaks.

## Signal strength

| Level      | Meaning                 | Frequency                 | You still get a digest?  |
|------------|-------------------------|---------------------------|--------------------------|
| **HIGH**   | Actually unusual. Rare. | 1-2x per month            | Obviously, and read it now |
| **MEDIUM** | Worth watching          | Weekly                    | Yes                      |
| **LOW**    | Normal Twitter noise    | Most days                 | Yes                      |
| **NONE**   | Twitter had nothing     | When everyone's at brunch | Yes                      |

Signal strength measures Twitter activity. CNBC hasn't figured out that most days are boring, which is why they fill
dead air screaming "IS THIS THE NEXT 2008?" every time the S&P dips 0.4%. Aladdin hasn't figured this out either, but
they charge you \$200k/year to not understand it, so at least it feels exclusive.

We also don't [hold \$11 billion in coal investments while lecturing people about climate](https://en.wikipedia.org/wiki/BlackRock#Investments_in_contributors_to_climate_change).

## Fact check classifications

| Tag              | Meaning                                     |
|------------------|---------------------------------------------|
| **VERIFIED**     | They told the truth. Mark your calendars.   |
| **EXAGGERATED**  | Directionally correct, emotionally unhinged |
| **FALSE**        | Lying on the internet. Groundbreaking.      |
| **UNVERIFIABLE** | Made up a ticker or was too vague to check  |

## Production (VectorChord)

Running on a VPS with PostgreSQL:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION vchord CASCADE;
```

One database. Tweets, embeddings, ML scores, signal judgments, account watch/block lists, pipeline checkpoints,
historical digests. All PostgreSQL. Advisory locks for pipeline concurrency. 3072-dimension Gemini embeddings. No
Oracle DBA explaining why the license renewal costs more than the GDP of Micronesia.

## Troubleshooting

**"No tweets retrieved"** - Your accounts are logged out or banned. `uv run twscrape accounts` to check. Re-add via
cookies. Welcome to the cat-and-mouse game with whatever Elon's platform is called this week.

**Rate limiting** - Add more accounts. Use proxies. Consider not scraping during market hours like an absolute maniac.

## Why this exists

| Feature                                                          | This project                 | Aladdin                                      | Bloomberg             | The Morning Brew        |
|------------------------------------------------------------------|------------------------------|----------------------------------------------|-----------------------|-------------------------|
| Cost                                                             | ~\$10/mo in API calls         | More than your rent                          | \$24,000/year          | Free (you are the product) |
| ML garbage detection                                             | IsolationForest + LLM judge  | Probably a guy named Dave who went to Wharton | N/A                   | N/A                     |
| Twitter sentiment                                                | **Yes**                      | Buried under 47 layers of enterprise middleware | Kinda, if you squint | They have a TikTok guy  |
| Will tell you "nothing matters today"                            | **Yes**                      | No. Gotta manufacture urgency to justify the invoice. | Have you met Jim Cramer | Every day is "HUGE"  |
| LLM that checks if people are lying                              | **Yes**                      | Their compliance team, eventually            | Still using RSS feeds | Their intern fact-checks by vibes |
| Dashboard aesthetic                                              | Windows Aero, shimmer included | SAP in a 400x300 box surrounded by whitespace | 1987                  | A WordPress template    |
| Open source                                                      | Yes                          | Lmao                                         | That's adorable       | Their "tech" is Mailchimp |
| Actually makes you a quant                                       | No                           | Also no, but it costs more so you feel like one | Still no           | Makes you think you are |
| Got no-bid Fed contracts to buy its own ETFs with taxpayer money | No                           | [Yes](https://wallstreetonparade.com/2020/06/blackrock-is-bailing-out-its-etfs-with-fed-money-and-taxpayers-eating-losses-its-also-the-sole-manager-for-335-billion-of-federal-employees-retirement-funds/) | No | No |
| Lost \$5B+ in pension mandates for being full of shit             | No                           | [Yes](https://www.fa-mag.com/news/blackrock-loses--5-9-billion-mandate-from-dutch-pension-pme-85220.html) | No | No |
| CEO has called the Fed Chair while managing his personal money   | No                           | [Yes, "extremely carefully"](https://www.investing.com/news/economy/blackrock-conflicts-managed-extremely-carefully-feds-powell-says-2245953) | Probably not | Their CEO calls his mom |

When we're wrong, it's free. When they're wrong, they get a bailout, a CNBC interview to explain why it was actually
your fault for not being diversified enough, and a LinkedIn post from their junior analyst about "lessons learned in
volatile markets" that gets 4,000 likes from other miserable fucks who also sacrificed their youth for a title that
impresses exactly no one outside of finance. Congrats on the VP promotion though. Your parents must be so proud. Do
they know what you actually do, or do you just say "risk management" and change the subject?

## The corruption receipts

You made it this far, so either you're interested or you work at BlackRock and you're hate-reading this during your
lunch break at your standing desk that cost more than this entire project's annual operating budget. If it's the
latter: how's the equity vesting going? Was it worth missing your sister's wedding for that Q3 deliverable? Do you
still remember what you wanted to be before "impact at scale" became your entire personality? The README isn't going
anywhere. Take your time. Either way,
here's the full starter pack. Every bullet is sourced. Every link works. Print it out and tape it to your monitor for
the next time someone at a cocktail party tells you the system works as intended:

- **Fed Chair Powell has \$25M personally invested with BlackRock** while handing them no-bid contracts to manage \$750
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

- **Dutch pension funds pulled \$5.9 billion** because even the Netherlands decided BlackRock wasn't acting in their
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

- **\$11 billion in coal investments** while being the world's largest investor in coal-fired power stations. Larry
  writes annual letters about climate responsibility with one hand and signs coal investment memos with the other. The
  Sierra Club started a campaign called "BlackRock's Big Problem" because sometimes you have to use small words for
  people who manage \$10
  trillion. ([Source](https://en.wikipedia.org/wiki/BlackRock#Investments_in_contributors_to_climate_change))

- **Dropped ESG shareholder support from 47% to 4%** the exact moment Ron DeSantis made it politically inconvenient.
  Voted against management 1,500+ times for "insufficient diversity" right up until it stopped being good PR.
  Principles are for people who can't afford
  lobbyists. ([Source](https://fortune.com/2024/02/14/blackrock-voting-choice-ceo-larry-fink-shareholder-democracy-stakeholder-capitalism-esg/))

This project costs maybe \$10/month in API calls. Aladdin costs more per month than your rent, and the people running
it are doing all of the above while posting "integrity is our north star" in their company Slack, then crying in the
bathroom because they haven't seen their friends in eight months and their therapist just raised their rates again.

## Disclaimer

Not financial advice. Not even close. Not financial advice in the same way that a weather report is not a personal
guarantee that it won't rain on you specifically. If you YOLO your life savings because this said "HIGH signal" on
some shitcoin, that is a you problem and we will not be taking questions, interviews, or depositions. Hedge funds
with actual Aladdin access lose money all the time. The difference is they get bailed out with your taxes and then
go on CNBC to explain why it was actually your fault for not being diversified enough across their seventeen
underperforming products. Then they post a LinkedIn carousel about "resilience in uncertain times" with a headshot
where they're wearing a Patagonia vest and smiling like they didn't just vaporize a pension fund. This system pulls
publicly available tweets, runs them through ML garbage detection, feeds them to an LLM that is constitutionally
incapable of feeling FOMO, and emails you a summary. If that destroys your portfolio, your portfolio was already on
life support and this just read it its last rites. Our bad takes are free. BlackRock's bad takes come with a
management fee and a 40-page shareholder letter about "navigating uncertainty" that manages to be longer than most
dissertations while saying less than a fortune cookie.

In January 2021, a bunch of people on Reddit with Robinhood accounts and zero institutional backing almost
bankrupted a \$13 billion hedge fund because they liked a stock. Melvin Capital needed a \$2.75 billion emergency
bailout and closed permanently a year later. The entire financial establishment lost its mind. Congress held hearings.
Billionaires went on TV and cried actual tears about fairness and market integrity, which is fucking hilarious coming
from people who've been front-running retail orders since before most of us were born. All because regular people had
access to the same information at the same time and acted on it before the suits could. This is that energy, but for
economic intelligence. We gave you the tools. Now go do something with your life that doesn't require sacrificing it
to a company that will lay you off via Zoom the moment Q4 numbers look soft.

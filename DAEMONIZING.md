# Daemonizing Jafar (Or: How I Learned to Stop Worrying and Love systemd)

So you want to run this on a VPS without babysitting it like it's a toddler with a hedge fund account. Smart. Unlike BlackRock's
[55.8% of underperforming funds](https://www.yodelar.com/insights/blackrock-review), we actually want this thing to run
*correctly* without human intervention.

## The Problem: Cloudflare Isn't Stupid (Unlike Most SaaS Companies)

If you run this at the same time every day, Cloudflare will notice. They're not idiots - they literally built a business on
detecting patterns. Running at 2:00 PM every day screams "I'm a bot" louder than Elon's latest unhinged tweet at 3 AM.

**Bad**: `cron: 0 14 * * *` → Every day at 2 PM sharp
**Result**: Cloudflare starts giving you the "prove you're human" treatment, which is ironic because most humans can't pass
those tests either

**Good**: Run at random times within windows, twice daily
**Result**: Cloudflare shrugs and lets you through because your traffic looks like a normal person who has insomnia and checks
fintwit at weird hours

## The Solution: systemd with Randomization

We're going to use systemd timers with `RandomizedDelaySec`, which is fancy Linux speak for "add some random-ass delay so you
don't look like a cron job written by someone who learned Linux from a 2009 blog post."

### Two Strategies (Pick Your Poison)

#### Option A: Twice-Daily Randomized (Recommended)

Run twice a day in randomized windows. This prevents overnight gaps where Silver could literally 3x and you'd wake up to a
Reuters alert like everyone else.

- **Morning**: Sometime between 7 AM - 12 PM (different time each day)
- **Evening**: Sometime between 5 PM - 11 PM (different time each day)
- **Max gap**: ~14 hours (11 PM → 1 PM next day worst case)
- **Min gap**: ~5 hours (12 PM → 5 PM same day worst case)

**Why this works**: You catch pre-market chatter AND after-hours panic. No 4 AM runs to wake you up. Different times defeat
pattern detection. If something goes nuclear at midnight, you'll catch it in the morning run.

#### Option B: Once-Daily (For the Minimalists)

Run once per day somewhere between 8 AM - 8 PM.

- **Window**: 8 AM - 8 PM (randomized daily)
- **Max gap**: Up to 36 hours (8 PM Monday → 8 AM Wednesday)
- **Warning**: You can literally miss an entire market news cycle

**When to use this**: You're okay with potentially missing overnight developments, or you just want to keep it simple because
you have a day job and don't need to catch every fintwit panic at 2 AM.

## Installation

### Step 1: Edit the Service File

```bash
cd systemd/
nano jafar.service  # or vim if you want to feel superior
```

Find these lines and replace `yourusername` with your actual username (appears 3 times):

```ini
User=yourusername
Group=yourusername
WorkingDirectory=/home/yourusername/twitter_sentiment_analysis
ExecStart=/home/yourusername/.local/bin/uv run main.py
ReadWritePaths=/home/yourusername/twitter_sentiment_analysis
```

**Pro tip**: Run `which uv` to confirm the path to your uv binary. If it's not in `~/.local/bin/uv`, update `ExecStart` accordingly.

### Step 2: Install systemd Files

```bash
# Copy the service file (required for both strategies)
sudo cp systemd/jafar.service /etc/systemd/system/

# OPTION A: Twice-daily (recommended)
sudo cp systemd/jafar-morning.timer /etc/systemd/system/
sudo cp systemd/jafar-evening.timer /etc/systemd/system/

# OPTION B: Once-daily (alternative)
sudo cp systemd/jafar.timer /etc/systemd/system/
```

### Step 3: Enable and Start Timers

```bash
# Tell systemd to reload because it's 2025 and we still need to do this manually
sudo systemctl daemon-reload

# OPTION A: Twice-daily
sudo systemctl enable jafar-morning.timer
sudo systemctl enable jafar-evening.timer
sudo systemctl start jafar-morning.timer
sudo systemctl start jafar-evening.timer

# OPTION B: Once-daily
sudo systemctl enable jafar.timer
sudo systemctl start jafar.timer
```

### Step 4: Verify It Actually Works

```bash
# Check timer status
systemctl list-timers jafar-*
```

You should see something like:

```
NEXT                         LEFT     LAST PASSED UNIT                  ACTIVATES
Mon 2025-12-30 09:34:22 EST  2h left  -    -      jafar-morning.timer   jafar.service
Mon 2025-12-30 19:12:45 EST  12h left -    -      jafar-evening.timer   jafar.service
```

**Important**: The `NEXT` times will be **different every day** due to randomization. If you see the exact same time
tomorrow, you fucked something up. Go back to Step 3.

### Step 5: Test It Before You Trust It

```bash
# Run it immediately (for testing)
sudo systemctl start jafar.service

# Watch the logs in real-time (like a hawk watching BlackRock's quarterly earnings)
sudo journalctl -u jafar.service -f

# Or check the file log
tail -f ~/twitter_sentiment_analysis/pipeline.log
```

If it works, congrats. If it doesn't, read the error messages. They're in English. Usually.

## How Randomization Works

Each timer has a base time + random delay:

### Morning Timer
```ini
OnCalendar=*-*-* 07:00:00      # Base: 7:00 AM
RandomizedDelaySec=18000        # Random: 0-5 hours (18000 seconds)
# Result: Runs somewhere between 7:00 AM - 12:00 PM
```

### Evening Timer
```ini
OnCalendar=*-*-* 17:00:00      # Base: 5:00 PM
RandomizedDelaySec=21600        # Random: 0-6 hours (21600 seconds)
# Result: Runs somewhere between 5:00 PM - 11:00 PM
```

**Example week** (completely randomized each day):

```
Monday:     9:23 AM, 8:47 PM  (morning run, evening run)
Tuesday:   11:12 AM, 6:34 PM
Wednesday:  7:45 AM, 10:22 PM
Thursday:   8:15 AM, 7:09 PM
Friday:    10:58 AM, 5:31 PM
```

Cloudflare looks at this and thinks "this is just a regular degenerate who can't maintain a consistent sleep schedule" and
lets you through. Which is exactly what we want.

## Monitoring & Maintenance

### Check if it's actually running

```bash
# Status of the timers themselves
sudo systemctl status jafar-morning.timer
sudo systemctl status jafar-evening.timer

# Status of the last service run
sudo systemctl status jafar.service
```

### View logs from today

```bash
sudo journalctl -u jafar.service --since today
```

### View the last 100 log entries

```bash
sudo journalctl -u jafar.service -n 100 --no-pager
```

### See upcoming schedule

```bash
systemctl list-timers
```

This shows when the next run is scheduled. Remember: **the time will be different every day**.

### Manually trigger a run (for testing)

```bash
sudo systemctl start jafar.service
```

This bypasses the timer and runs immediately. Useful for:
- Testing changes to config.yaml
- Showing off to your friends that you have a VPS
- Panicking at 2 AM because you think you missed something

### Disable timers temporarily

```bash
sudo systemctl stop jafar-morning.timer jafar-evening.timer
```

**Warning**: This stops future scheduled runs but doesn't kill an in-progress run. If you want to murder a running process:

```bash
sudo systemctl stop jafar.service
```

### Re-enable after disabling

```bash
sudo systemctl start jafar-morning.timer jafar-evening.timer
```

## Admin Diagnostics (The Good Stuff)

The system automatically sends you an email if shit goes sideways. Configure this in `config.yaml`:

```yaml
email:
  admin:
    enabled: true              # Enable admin diagnostics emails
    send_on_success: false     # Set to true if you want emails even when everything works
    recipients:                # Leave empty to use main email recipients
      - your-admin@email.com
    log_retention_count: 10    # Keep 10 most recent log files
```

**What triggers an admin alert:**

- **Zero tweets scraped** → Your Twitter accounts are dead/banned/rate-limited
- **All accounts unavailable** → Time to add more accounts or wait for rate limits to reset
- **Critical errors** → Something broke and the email has details
- **Very low tweet count** (< 50) → Scraper is struggling, might need more accounts
- **No trends discovered** → Either genuinely quiet day or something's fucked

**What's in the diagnostic email:**

- Run duration and timestamp
- Tweet counts (broad + deep dive)
- Twitter account health (X active out of Y total)
- Trends discovered (before and after LLM filter)
- LLM token usage (so you know when OpenAI is bleeding you dry)
- Performance breakdown by pipeline step
- All errors and warnings with timestamps

**Log rotation**: Logs are automatically rotated after each run. Old logs are renamed with timestamps (`pipeline_20251230_143022.log`)
and kept according to `log_retention_count`. Older logs are deleted automatically because disk space isn't infinite, despite
what AWS wants you to believe.

## Troubleshooting

### "The timer isn't running"

```bash
# Did you enable it?
sudo systemctl enable jafar-morning.timer

# Did you start it?
sudo systemctl start jafar-morning.timer

# Check status
sudo systemctl status jafar-morning.timer
```

If status shows "inactive (dead)", you didn't enable or start it. Go back to Step 3 and actually read the commands this time.

### "I changed config.yaml and nothing happened"

systemd doesn't auto-reload config files because it's not psychic. The changes will apply on the next scheduled run. If you
can't wait:

```bash
sudo systemctl start jafar.service  # Run it now
```

### "I get CRITICAL emails saying zero tweets scraped"

Your Twitter accounts are fucked. Check them:

```bash
cd ~/twitter_sentiment_analysis
uv run twscrape accounts
```

If they show as logged out or rate-limited, re-add them via cookies:

```bash
uv run python add_account.py <username> cookies.json
```

Consider adding more accounts or configuring SOCKS5 proxies if you're consistently hitting rate limits.

### "I want to change the timing windows"

Edit the timer files:

```bash
sudo nano /etc/systemd/system/jafar-morning.timer
```

Change `OnCalendar` (base time) or `RandomizedDelaySec` (random delay in seconds):

- Want 6am-11am instead of 7am-12pm? Change `OnCalendar` to `06:00:00`
- Want more randomization? Increase `RandomizedDelaySec` (18000 = 5 hours)
- Want less? Decrease it

Then reload:

```bash
sudo systemctl daemon-reload
sudo systemctl restart jafar-morning.timer
```

### "This is too complicated, I just want cron"

No.

If you use cron at a fixed time, you're defeating the entire purpose of randomization. Cloudflare will pattern-match your
ass faster than BlackRock can spin up a new underperforming ESG fund.

But if you insist on being wrong:

```bash
# DO NOT DO THIS
0 14 * * * cd ~/twitter_sentiment_analysis && uv run main.py
```

Congratulations, you've recreated the problem we're trying to solve. When Cloudflare starts blocking you, don't come crying.

## systemd File Reference

**jafar.service** - The main service definition
- Runs the pipeline with proper permissions and resource limits
- Logs to `pipeline.log` (both stdout and stderr)
- Security hardening: PrivateTmp, NoNewPrivileges, ProtectSystem
- Resource limits: 2GB RAM max, 80% CPU quota
- 10 minute timeout (adjust if your internet is dialup)

**jafar-morning.timer** - Morning randomized runs
- Base: 7:00 AM
- Random: +0-5 hours
- Window: 7:00 AM - 12:00 PM
- `Persistent=true` means if your VPS was off when it should have run, it'll catch up on next boot

**jafar-evening.timer** - Evening randomized runs
- Base: 5:00 PM (17:00)
- Random: +0-6 hours
- Window: 5:00 PM - 11:00 PM
- Same persistence behavior

**jafar.timer** - Alternative single daily run
- Base: 2:00 PM (14:00)
- Random: ±6 hours
- Window: 8:00 AM - 8:00 PM
- Warning: Can create gaps up to 36 hours. Use twice-daily instead.

## Final Notes

- **Randomization defeats pattern detection**. That's the entire point. If you want predictable timing, you're missing the plot.
- **Twice-daily is recommended** because you don't want to wake up to a "BREAKING: Silver up 40%" Reuters alert like a normie
- **Admin diagnostics emails** mean you'll know when shit breaks instead of wondering why you haven't gotten a digest in 3 days
- **Log rotation** keeps your disk from filling up with gigabytes of "Successfully scraped tweet #47392"

You now have a system that:
1. Runs automatically at unpredictable times
2. Emails you when something breaks
3. Catches both pre-market and after-hours fintwit panic
4. Doesn't wake you up at 4 AM
5. Makes Cloudflare think you're just another degenerate trader with insomnia

Unlike Aladdin, which requires a blood sacrifice to their enterprise sales team, three weeks of configuration, and a mortgage
to afford, this just... works. And it's free. Well, $10/month in API calls. BlackRock charges more for their vending machine
coffee.

Now go touch grass. Or don't. I'm a markdown file, not your therapist.

"""
Market Data Fact Checker for Twitter Sentiment Analysis.

Fetches real market data to verify claims made in tweets.
When tweets say "SILVER IS SUPER HIGH!!!", this module checks
actual silver prices to ground the analysis in facts.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf

logger = logging.getLogger("jafar.fact_checker")


# Symbol mappings for common terms
COMMODITY_MAP = {
    "gold": "GC=F",
    "silver": "SI=F",
    "platinum": "PL=F",
    "palladium": "PA=F",
    "oil": "CL=F",
    "crude": "CL=F",
    "crude oil": "CL=F",
    "wti": "CL=F",
    "brent": "BZ=F",
    "natural gas": "NG=F",
    "nat gas": "NG=F",
    "natgas": "NG=F",
    "copper": "HG=F",
    "wheat": "ZW=F",
    "corn": "ZC=F",
    "soybeans": "ZS=F",
    "soybean": "ZS=F",
    "coffee": "KC=F",
    "sugar": "SB=F",
    "cocoa": "CC=F",
    "cotton": "CT=F",
    "lumber": "LBS=F",
    "uranium": "URA",  # ETF proxy
}

CRYPTO_MAP = {
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "solana": "SOL-USD",
    "sol": "SOL-USD",
    "xrp": "XRP-USD",
    "ripple": "XRP-USD",
    "cardano": "ADA-USD",
    "ada": "ADA-USD",
    "dogecoin": "DOGE-USD",
    "doge": "DOGE-USD",
    "litecoin": "LTC-USD",
    "ltc": "LTC-USD",
}

INDEX_MAP = {
    "s&p": "^GSPC",
    "s&p 500": "^GSPC",
    "sp500": "^GSPC",
    "spx": "^GSPC",
    "spy": "SPY",
    "nasdaq": "^IXIC",
    "qqq": "QQQ",
    "dow": "^DJI",
    "djia": "^DJI",
    "dow jones": "^DJI",
    "russell": "^RUT",
    "russell 2000": "^RUT",
    "iwm": "IWM",
    "vix": "^VIX",
    "volatility": "^VIX",
}

SECTOR_ETF_MAP = {
    "energy": "XLE",
    "financials": "XLF",
    "technology": "XLK",
    "tech": "XLK",
    "healthcare": "XLV",
    "consumer discretionary": "XLY",
    "consumer staples": "XLP",
    "industrials": "XLI",
    "materials": "XLB",
    "utilities": "XLU",
    "real estate": "XLRE",
    "communication": "XLC",
}


@dataclass
class MarketDataPoint:
    """Real market data for a single symbol."""
    symbol: str
    name: str
    current_price: float
    price_change_1d: float
    price_change_1d_pct: float
    price_change_5d_pct: float | None
    volume: int
    avg_volume: int
    high_52w: float
    low_52w: float
    market_cap: float | None
    fetched_at: datetime
    category: str  # "commodity", "crypto", "stock", "index", "etf"

    @property
    def is_near_52w_high(self) -> bool:
        """Check if price is within 5% of 52-week high."""
        if self.high_52w <= 0:
            return False
        return self.current_price >= self.high_52w * 0.95

    @property
    def is_near_52w_low(self) -> bool:
        """Check if price is within 5% of 52-week low."""
        if self.low_52w <= 0:
            return False
        return self.current_price <= self.low_52w * 1.05

    @property
    def is_unusual_volume(self) -> bool:
        """Check if volume is 2x above average."""
        if self.avg_volume <= 0:
            return False
        return self.volume > self.avg_volume * 2

    @property
    def volume_ratio(self) -> float:
        """Get volume as multiple of average."""
        if self.avg_volume <= 0:
            return 0.0
        return self.volume / self.avg_volume


class MarketFactChecker:
    """
    Fetches and caches real market data to fact-check tweet claims.
    """

    def __init__(
        self,
        cache_ttl_minutes: int = 5,
        price_tolerance_pct: float = 2.0,
    ):
        """
        Initialize the fact checker.

        Args:
            cache_ttl_minutes: How long to cache market data.
            price_tolerance_pct: Allowed variance for "price at X" claims.
        """
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.price_tolerance_pct = price_tolerance_pct
        self._cache: dict[str, tuple[MarketDataPoint, datetime]] = {}
        logger.info(f"MarketFactChecker initialized (cache TTL: {cache_ttl_minutes}m)")

    def extract_symbols_from_trends(self, trends: list[str]) -> set[str]:
        """
        Extract market symbols from trend names.

        Args:
            trends: List of discovered trend terms.

        Returns:
            Set of Yahoo Finance symbols to fetch.
        """
        symbols = set()

        for trend in trends:
            trend_lower = trend.lower().strip()

            # Check for cashtags ($AAPL)
            cashtag_match = re.match(r'^\$([A-Za-z]{1,5})$', trend)
            if cashtag_match:
                symbols.add(cashtag_match.group(1).upper())
                continue

            # Check commodity map
            for keyword, symbol in COMMODITY_MAP.items():
                if keyword in trend_lower:
                    symbols.add(symbol)
                    break

            # Check crypto map
            for keyword, symbol in CRYPTO_MAP.items():
                if keyword == trend_lower or keyword in trend_lower.split():
                    symbols.add(symbol)
                    break

            # Check index map
            for keyword, symbol in INDEX_MAP.items():
                if keyword == trend_lower or keyword in trend_lower:
                    symbols.add(symbol)
                    break

            # Check sector ETFs
            for keyword, symbol in SECTOR_ETF_MAP.items():
                if keyword in trend_lower:
                    symbols.add(symbol)
                    break

        logger.info(f"Extracted {len(symbols)} symbols from {len(trends)} trends: {symbols}")
        return symbols

    def _get_category(self, symbol: str) -> str:
        """Determine the category of a symbol."""
        if symbol.endswith("=F"):
            return "commodity"
        if symbol.endswith("-USD"):
            return "crypto"
        if symbol.startswith("^"):
            return "index"
        if symbol in SECTOR_ETF_MAP.values():
            return "etf"
        return "stock"

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid."""
        if symbol not in self._cache:
            return False
        _, cached_at = self._cache[symbol]
        return datetime.now() - cached_at < self.cache_ttl

    async def fetch_market_data(
        self,
        symbols: set[str],
        include_common: bool = True,
    ) -> dict[str, MarketDataPoint]:
        """
        Fetch market data for symbols.

        Args:
            symbols: Set of symbols to fetch.
            include_common: Whether to include common symbols (gold, silver, SPY, BTC).

        Returns:
            Dictionary mapping symbols to their market data.
        """
        # Add common symbols for context
        if include_common:
            common = {"GC=F", "SI=F", "CL=F", "SPY", "^VIX", "BTC-USD"}
            symbols = symbols | common

        # Filter out cached symbols
        to_fetch = {s for s in symbols if not self._is_cache_valid(s)}

        if to_fetch:
            logger.info(f"Fetching market data for {len(to_fetch)} symbols...")
            # Run yfinance in thread pool (it's synchronous)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._fetch_batch, to_fetch)

        # Return all requested data from cache
        result = {}
        for symbol in symbols:
            if symbol in self._cache:
                data, _ = self._cache[symbol]
                result[symbol] = data

        logger.info(f"Returning market data for {len(result)} symbols")
        return result

    def _fetch_batch(self, symbols: set[str]) -> None:
        """
        Fetch a batch of symbols (synchronous, called from executor).
        Uses history(period="5d") to get context on recent moves.
        """
        try:
            # Join symbols for batch fetching if possible, but history() is per-ticker usually
            # efficient enough to just loop if we use the Ticker object correctly
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # Fetch 5 days of history for context
                    # period="1mo" is safer to ensure we get 5 trading days even with holidays
                    hist = ticker.history(period="1mo")
                    
                    if hist.empty:
                        logger.warning(f"No history found for {symbol}")
                        continue

                    # Get latest data
                    current_row = hist.iloc[-1]
                    current_price = float(current_row["Close"])
                    
                    # Calculate 1-day change
                    if len(hist) >= 2:
                        prev_close = float(hist.iloc[-2]["Close"])
                        price_change_1d = current_price - prev_close
                        price_change_1d_pct = (price_change_1d / prev_close * 100)
                    else:
                        price_change_1d = 0.0
                        price_change_1d_pct = 0.0

                    # Calculate 5-day change (1 week)
                    price_change_5d_pct = None
                    if len(hist) >= 6:
                        # 5 trading days ago is index -6 (current is -1)
                        # e.g., if we have [Mon, Tue, Wed, Thu, Fri, Mon(Today)]
                        # -1 is Mon(Today), -6 is Mon(Last week)
                        five_day_close = float(hist.iloc[-6]["Close"])
                        price_change_5d_pct = ((current_price - five_day_close) / five_day_close * 100)
                    elif len(hist) > 1:
                        # Fallback to start of available data if < 6 days
                        start_close = float(hist.iloc[0]["Close"])
                        price_change_5d_pct = ((current_price - start_close) / start_close * 100)

                    # Get auxiliary info (volume, ranges) from fast_info or info
                    try:
                        fast = ticker.fast_info
                        vol = int(current_row["Volume"]) if "Volume" in current_row else 0
                        avg_vol = fast.get("averageVolume", 0) or 0
                        high_52 = fast.get("fiftyTwoWeekHigh", 0) or 0
                        low_52 = fast.get("fiftyTwoWeekLow", 0) or 0
                        mkt_cap = fast.get("marketCap")
                        name = symbol # Default
                    except Exception:
                        # Fallback to slower .info if fast_info fails
                        info = ticker.info
                        vol = info.get("regularMarketVolume", 0) or 0
                        avg_vol = info.get("averageDailyVolume10Day", 0) or 0
                        high_52 = info.get("fiftyTwoWeekHigh", 0) or 0
                        low_52 = info.get("fiftyTwoWeekLow", 0) or 0
                        mkt_cap = info.get("marketCap")
                        name = info.get("shortName", symbol)

                    # Create data point
                    data = MarketDataPoint(
                        symbol=symbol,
                        name=name,
                        current_price=current_price,
                        price_change_1d=price_change_1d,
                        price_change_1d_pct=price_change_1d_pct,
                        price_change_5d_pct=price_change_5d_pct,
                        volume=vol,
                        avg_volume=avg_vol,
                        high_52w=high_52,
                        low_52w=low_52,
                        market_cap=mkt_cap,
                        fetched_at=datetime.now(),
                        category=self._get_category(symbol),
                    )

                    self._cache[symbol] = (data, datetime.now())
                    
                    log_msg = f"Fetched {symbol}: ${data.current_price:.2f} ({data.price_change_1d_pct:+.2f}%)"
                    if price_change_5d_pct is not None:
                        log_msg += f" [5d: {price_change_5d_pct:+.2f}%]"
                    logger.debug(log_msg)

                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol}: {e}")

        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")

    def format_for_llm(
        self,
        market_data: dict[str, MarketDataPoint],
        trends: list[str],
    ) -> str:
        """
        Format market data as context for the LLM.

        Args:
            market_data: Dictionary of fetched market data.
            trends: List of trends being analyzed.

        Returns:
            Markdown-formatted string for LLM context.
        """
        if not market_data:
            return ""

        now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "## VERIFIED MARKET DATA",
            f"*Real-time data as of {now}. Use this to verify or refute tweet claims.*",
            "",
        ]

        # Helper to format 5d change
        def fmt_5d(val: float | None) -> str:
            if val is None:
                return "-"
            return f"{val:+.1f}%"

        # Group by category
        commodities = {k: v for k, v in market_data.items() if v.category == "commodity"}
        crypto = {k: v for k, v in market_data.items() if v.category == "crypto"}
        indices = {k: v for k, v in market_data.items() if v.category in ("index", "etf")}
        stocks = {k: v for k, v in market_data.items() if v.category == "stock"}

        if commodities:
            lines.append("### Commodities")
            lines.append("| Asset | Price | 24h Change | 5d Trend | Volume | Notes |")
            lines.append("|-------|-------|------------|----------|--------|-------|")
            for symbol, data in sorted(commodities.items(), key=lambda x: x[1].name):
                notes = self._get_notes(data)
                vol_str = f"{data.volume_ratio:.1f}x avg" if data.avg_volume > 0 else "N/A"
                lines.append(
                    f"| {data.name} | ${data.current_price:,.2f} | "
                    f"{data.price_change_1d_pct:+.1f}% | {fmt_5d(data.price_change_5d_pct)} | "
                    f"{vol_str} | {notes} |"
                )
            lines.append("")

        if indices:
            lines.append("### Indices & ETFs")
            lines.append("| Index | Level | 24h Change | 5d Trend | Notes |")
            lines.append("|-------|-------|------------|----------|-------|")
            for symbol, data in sorted(indices.items(), key=lambda x: x[1].name):
                notes = self._get_notes(data)
                lines.append(
                    f"| {data.name} ({symbol}) | {data.current_price:,.2f} | "
                    f"{data.price_change_1d_pct:+.1f}% | {fmt_5d(data.price_change_5d_pct)} | {notes} |"
                )
            lines.append("")

        if crypto:
            lines.append("### Crypto")
            lines.append("| Asset | Price | 24h Change | 5d Trend | Notes |")
            lines.append("|-------|-------|------------|----------|-------|")
            for symbol, data in sorted(crypto.items(), key=lambda x: x[1].name):
                notes = self._get_notes(data)
                lines.append(
                    f"| {data.name} | ${data.current_price:,.2f} | "
                    f"{data.price_change_1d_pct:+.1f}% | {fmt_5d(data.price_change_5d_pct)} | {notes} |"
                )
            lines.append("")

        if stocks:
            lines.append("### Stocks Mentioned")
            lines.append("| Ticker | Price | 24h Change | 5d Trend | Volume | Notes |")
            lines.append("|--------|-------|------------|----------|--------|-------|")
            for symbol, data in sorted(stocks.items()):
                notes = self._get_notes(data)
                vol_str = f"{data.volume_ratio:.1f}x avg" if data.avg_volume > 0 else "N/A"
                lines.append(
                    f"| {symbol} | ${data.current_price:,.2f} | "
                    f"{data.price_change_1d_pct:+.1f}% | {fmt_5d(data.price_change_5d_pct)} | "
                    f"{vol_str} | {notes} |"
                )
            lines.append("")

        # Add instructions
        lines.extend([
            "---",
            "**FACT-CHECK INSTRUCTIONS:**",
            "- Compare tweet claims against this verified data",
            "- **CHECK THE 5d TREND**: A large daily drop might just be a pullback if the 5d trend is still positive.",
            "- Flag claims that contradict the actual numbers",
            "- Note when sentiment aligns with real price action",
            "- \"Massive volume\" should show >2x avg; otherwise it's exaggerated",
            "- \"All-time high\" / \"52w high\" claims should match the Notes column",
            "",
        ])

        return "\n".join(lines)

    def _get_notes(self, data: MarketDataPoint) -> str:
        """Generate notes for a data point."""
        notes = []
        if data.is_near_52w_high:
            notes.append("Near 52w High")
        if data.is_near_52w_low:
            notes.append("Near 52w Low")
        if data.is_unusual_volume:
            notes.append("High Volume")
        return ", ".join(notes) if notes else "-"

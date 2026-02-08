"""
Tool definitions for Jafar's LLM agent.

This module provides the tools that the LLM can call to fetch market data,
historical parallels, and other context on demand.
"""

import logging
from typing import Any

import aiohttp

from .fact_checker import MarketFactChecker
from .memory import MemoryManager
from .temporal_analyzer import TemporalTrendAnalyzer, TrendTimeline

logger = logging.getLogger("jafar.tools")


class ToolRegistry:
    """
    Registry of tools available to the LLM.
    """

    def __init__(
        self,
        fact_checker: MarketFactChecker | None = None,
        memory: MemoryManager | None = None,
        temporal_analyzer: TemporalTrendAnalyzer | None = None,
        trend_timelines: dict[str, TrendTimeline] | None = None,
        enable_web_search: bool = True,
    ):
        """
        Initialize the tool registry with necessary dependencies.
        """
        self.fact_checker = fact_checker
        self.memory = memory
        self.temporal_analyzer = temporal_analyzer
        self.trend_timelines = trend_timelines or {}
        self.enable_web_search = enable_web_search
        
        # Initialize DuckDuckGo search if enabled
        if self.enable_web_search:
            try:
                from ddgs import DDGS
                self.ddgs = DDGS()
            except ImportError:
                logger.warning("ddgs not installed. Web search tool disabled.")
                self.enable_web_search = False
            except Exception as e:
                logger.warning(f"Failed to initialize DuckDuckGo search: {e}")
                self.enable_web_search = False

    def get_definitions(self) -> list[dict[str, Any]]:
        """
        Get the JSON schema definitions for all available tools.
        """
        tools = []
        
        if self.enable_web_search:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for real-time information, news, or verification of claims. Use this for deep research.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (e.g. 'RTX 5090 release date', 'uranium shortage causes')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })

        if self.enable_web_search:
            tools.append({
                "type": "function",
                "function": {
                    "name": "fetch_news",
                    "description": "Fetch latest news headlines on a specific topic. Use this to drill deeper into a news story or find additional context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "News search query (e.g. 'Federal Reserve rate decision', 'NVIDIA earnings results')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })

        if self.fact_checker:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_market_data",
                    "description": "Fetch real-time market data for specific stocks, commodities, crypto, or indices. Use this to verify claims about prices, volume, or trends.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of symbols or names to fetch (e.g. ['NVDA', 'Silver', 'Bitcoin'])"
                            }
                        },
                        "required": ["symbols"]
                    }
                }
            })

        if self.memory:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_historical_parallels",
                    "description": "Search for historical events similar to the current situation. Use this to find precedents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Description of the current situation to match against history."
                            },
                        },
                        "required": ["query"]
                    }
                }
            })

        if self.temporal_analyzer and self.trend_timelines:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_trend_timeline",
                    "description": "Get the timeline of a specific trend (first seen, consecutive days, gaps).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "trend": {
                                "type": "string",
                                "description": "The trend name to check."
                            }
                        },
                        "required": ["trend"]
                    }
                }
            })

        # Weather tool is always available (uses free Open-Meteo API)
        tools.append({
            "type": "function",
            "function": {
                "name": "get_weather_forecast",
                "description": "Get current weather and 7-day forecast for cities. Use this to understand weather-driven consumer behavior (panic buying, supply disruptions, travel impacts).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of city names (e.g. ['Houston', 'Dallas', 'New York City'])"
                        }
                    },
                    "required": ["cities"]
                }
            }
        })

        # Submit report tool - always available, used to finalize the analysis
        tools.append({
            "type": "function",
            "function": {
                "name": "submit_report",
                "description": "Submit your final analysis report. You MUST call this tool when you are done analyzing to deliver your findings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject_line": {
                            "type": "string",
                            "description": "A punchy email subject (5-10 words). Reference the top trend. Can be witty but informative."
                        },
                        "signal_strength": {
                            "type": "string",
                            "enum": ["high", "medium", "low", "none"],
                            "description": "HIGH=rare actionable signal, MEDIUM=worth monitoring, LOW=noise, NONE=nothing happening"
                        },
                        "assessment": {
                            "type": "string",
                            "description": "2-3 sentences with dry wit. If LOW/NONE signal, say 'Another day of fintwit being fintwit.'"
                        },
                        "trends_observed": {
                            "type": "string",
                            "description": "Bullet points of what's being discussed - factual, not hyped. Use '•' character for each bullet point (e.g., '• Gold up 4.6%\\n• Silver breaking out')."
                        },
                        "fact_check": {
                            "type": "string",
                            "description": "Categorized fact-check results using '•' bullets. Group by category with each on its own line, using bold Title Case labels:\\n• **Verified:** [claims that match market data]\\n• **Exaggerated:** [directionally correct but overstated]\\n• **False:** [claims contradicting data]\\n• **Unverified:** [claims without data to check]\\nLeave empty if no fact checking was done."
                        },
                        "actionability": {
                            "type": "string",
                            "enum": ["not actionable", "monitor only", "worth researching", "warrants attention"],
                            "description": "How actionable is this information?"
                        },
                        "actionability_reason": {
                            "type": "string",
                            "description": "1 sentence explaining the actionability rating."
                        },
                        "historical_parallel": {
                            "type": "string",
                            "description": "If meaningful: 'History rhymes: [parallel]'. Otherwise: 'No meaningful historical parallels.'"
                        },
                        "bottom_line": {
                            "type": "string",
                            "description": "1 sentence. Be direct, be memorable. 'Save your attention for another day' is valid."
                        },
                        "news_roundup": {
                            "type": "string",
                            "description": "Bullet-pointed summary of today's economic news headlines with brief commentary. ALWAYS populate this when news headlines are provided in the prompt. Use '•' bullets. Example: '• Fed holds rates steady - no surprise, markets shrug\\n• NVIDIA earnings beat estimates by 15%'"
                        }
                    },
                    "required": ["subject_line", "signal_strength", "assessment", "trends_observed", "actionability", "actionability_reason", "bottom_line"]
                }
            }
        })

        return tools

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool by name with arguments.
        """
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        try:
            if tool_name == "search_web":
                return await self._search_web(arguments.get("query", ""))

            elif tool_name == "get_market_data" and self.fact_checker:
                symbols = arguments.get("symbols", [])
                if not symbols:
                    return "No symbols provided."
                
                # Convert names to symbols (the fact checker helper does this)
                # But fact_checker.extract_symbols_from_trends expects a list of trends
                # We can reuse it or adding a simple mapper
                # For robust usage, let's treat inputs as potential trends/names
                
                # Important: Include common symbols ONLY if explicitly requested or if list is empty? 
                # No, we want to solve the pollution. So strictly fetch what is asked.
                
                extracted_symbols = self.fact_checker.extract_symbols_from_trends(symbols)
                if not extracted_symbols:
                   # Try to interpret the input strings directly as symbols if extraction failed
                   # e.g. user passed "NVDA", which might not be in the dictionary but is a valid ticker
                   for s in symbols:
                       if s.isupper() and len(s) <= 5: 
                           extracted_symbols.add(s)
                           
                market_data = await self.fact_checker.fetch_market_data(extracted_symbols, include_common=False)
                return self.fact_checker.format_for_llm(market_data, symbols)

            elif tool_name == "search_historical_parallels" and self.memory:
                query = arguments.get("query", "")
                parallels = await self.memory.find_parallels(
                    trends=[query], # simplified usage
                    themes=[],
                    sentiment="unknown",
                    signal_strength="unknown",
                    limit=3,
                    min_similarity=0.7 # stricter threshold for search
                )
                if not parallels:
                    return "No significant historical parallels found."
                return await self.memory.format_parallels_for_llm(parallels)

            elif tool_name == "get_trend_timeline" and self.temporal_analyzer:
                trend = arguments.get("trend")
                if trend in self.trend_timelines:
                    timeline = self.trend_timelines[trend]
                    # Format a single timeline
                    timelines_dict = {trend: timeline}
                    return self.temporal_analyzer.format_context_for_llm(timelines_dict)
                return f"No timeline data found for trend: {trend}"

            elif tool_name == "fetch_news":
                return await self._fetch_news(arguments.get("query", ""))

            elif tool_name == "get_weather_forecast":
                cities = arguments.get("cities", [])
                if not cities:
                    return "No cities provided."
                return await self._get_weather_forecast(cities)

            else:
                return f"Tool {tool_name} not found or dependency missing."

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing tool: {e}"

    async def _search_web(self, query: str) -> str:
        """
        Execute a DuckDuckGo web search.
        """
        if not self.enable_web_search or not hasattr(self, 'ddgs'):
             return "Web search is not enabled or available."

        try:
            # Run in executor since ddgs is synchronous
            import asyncio
            loop = asyncio.get_running_loop()
            
            def run_search():
                # text() returns an iterator/generator in recent versions, or list in older
                # We want max_results=4
                return self.ddgs.text(query, max_results=4)
                
            results = await loop.run_in_executor(None, run_search)
            
            if not results:
                return f"No results found for query: {query}"
            
            # Format results
            formatted = f"Web Search Results for '{query}':\n\n"
            for r in results:
                formatted += f"- **{r.get('title', 'No Title')}**\n"
                formatted += f"  {r.get('body', 'No snippet')}\n"
                formatted += f"  Source: {r.get('href', 'No URL')}\n\n"
            
            return formatted

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Error performing web search: {e}"

    async def _fetch_news(self, query: str) -> str:
        """
        Fetch news headlines via DuckDuckGo news search.
        """
        if not self.enable_web_search or not hasattr(self, 'ddgs'):
            return "News fetching is not enabled or available."

        try:
            import asyncio
            loop = asyncio.get_running_loop()

            def run_news_search():
                return self.ddgs.news(query, max_results=5)

            results = await loop.run_in_executor(None, run_news_search)

            if not results:
                return f"No news found for query: {query}"

            formatted = f"News Results for '{query}':\n\n"
            for r in results:
                formatted += f"- **{r.get('title', 'No Title')}**\n"
                formatted += f"  {r.get('body', 'No snippet')}\n"
                formatted += f"  Source: {r.get('source', 'Unknown')} | {r.get('date', '')}\n\n"

            return formatted

        except Exception as e:
            logger.error(f"News fetch failed: {e}")
            return f"Error fetching news: {e}"

    async def _get_weather_forecast(self, cities: list[str]) -> str:
        """
        Get weather forecast for multiple cities using Open-Meteo API (free, no key needed).
        """
        # Weather code descriptions for interpretation
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            66: "Light freezing rain", 67: "Heavy freezing rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
        }

        results = []

        async with aiohttp.ClientSession() as session:
            for city in cities[:5]:  # Limit to 5 cities
                try:
                    # Step 1: Geocode city name to coordinates
                    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
                    async with session.get(geo_url, params={"name": city, "count": 1}) as resp:
                        if resp.status != 200:
                            results.append(f"**{city}**: Could not geocode city")
                            continue
                        geo_data = await resp.json()

                    if not geo_data.get("results"):
                        results.append(f"**{city}**: City not found")
                        continue

                    location = geo_data["results"][0]
                    lat, lon = location["latitude"], location["longitude"]
                    display_name = f"{location.get('name', city)}, {location.get('admin1', '')}, {location.get('country', '')}"

                    # Step 2: Get weather forecast
                    weather_url = "https://api.open-meteo.com/v1/forecast"
                    weather_params = {
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_gusts_10m",
                        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
                        "temperature_unit": "fahrenheit",
                        "wind_speed_unit": "mph",
                        "precipitation_unit": "inch",
                        "timezone": "auto",
                        "forecast_days": 7,
                    }

                    async with session.get(weather_url, params=weather_params) as resp:
                        if resp.status != 200:
                            results.append(f"**{display_name}**: Could not fetch weather")
                            continue
                        weather = await resp.json()

                    # Format current conditions
                    current = weather.get("current", {})
                    current_code = current.get("weather_code", 0)
                    current_desc = weather_codes.get(current_code, "Unknown")

                    city_result = f"**{display_name}**\n"
                    city_result += f"  Current: {current_desc}, {current.get('temperature_2m', 'N/A')}°F "
                    city_result += f"(feels like {current.get('apparent_temperature', 'N/A')}°F)\n"
                    city_result += f"  Wind: {current.get('wind_speed_10m', 'N/A')} mph, "
                    city_result += f"Gusts: {current.get('wind_gusts_10m', 'N/A')} mph\n"
                    city_result += f"  Humidity: {current.get('relative_humidity_2m', 'N/A')}%\n"

                    # Format 7-day forecast
                    daily = weather.get("daily", {})
                    dates = daily.get("time", [])
                    city_result += "  7-Day Forecast:\n"

                    for i, date in enumerate(dates):
                        code = daily.get("weather_code", [])[i] if i < len(daily.get("weather_code", [])) else 0
                        desc = weather_codes.get(code, "Unknown")
                        high = daily.get("temperature_2m_max", [])[i] if i < len(daily.get("temperature_2m_max", [])) else "N/A"
                        low = daily.get("temperature_2m_min", [])[i] if i < len(daily.get("temperature_2m_min", [])) else "N/A"
                        precip = daily.get("precipitation_sum", [])[i] if i < len(daily.get("precipitation_sum", [])) else 0
                        precip_prob = daily.get("precipitation_probability_max", [])[i] if i < len(daily.get("precipitation_probability_max", [])) else 0

                        city_result += f"    {date}: {desc}, {low}°F-{high}°F"
                        if precip > 0 or precip_prob > 30:
                            city_result += f", Precip: {precip}\" ({precip_prob}% chance)"
                        city_result += "\n"

                    results.append(city_result)

                except Exception as e:
                    logger.error(f"Weather fetch failed for {city}: {e}")
                    results.append(f"**{city}**: Error fetching weather: {e}")

        if not results:
            return "Could not fetch weather for any cities."

        return "Weather Forecast:\n\n" + "\n".join(results)

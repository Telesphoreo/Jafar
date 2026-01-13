"""
Tool definitions for Jafar's LLM agent.

This module provides the tools that the LLM can call to fetch market data,
historical parallels, and other context on demand.
"""

import logging
from typing import Any, Callable

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
                from duckduckgo_search import DDGS
                self.ddgs = DDGS()
            except ImportError:
                logger.warning("duckduckgo-search not installed. Web search tool disabled.")
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

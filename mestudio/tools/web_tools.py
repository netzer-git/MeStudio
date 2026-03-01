"""Web search and page reading tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from mestudio.context.token_counter import get_token_counter
from mestudio.core.config import get_settings
from mestudio.tools.registry import tool

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str

    def format(self) -> str:
        """Format for display."""
        return f"**{self.title}**\n{self.url}\n{self.snippet}"


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Execute a search query.
        
        Args:
            query: Search query string.
            num_results: Number of results to return.
        
        Returns:
            List of search results.
        """
        pass


class DDGSProvider(SearchProvider):
    """DuckDuckGo search provider using the ddgs library.
    
    This is the default and most reliable option - no API key required,
    handles rate limiting and bot detection automatically.
    """

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Search using DuckDuckGo via ddgs library.
        
        Args:
            query: Search query string.
            num_results: Number of results to return.
        
        Returns:
            List of search results.
        """
        import asyncio
        from ddgs import DDGS
        
        # Run sync DDGS in thread pool to not block
        def _search():
            try:
                return list(DDGS().text(query, max_results=num_results))
            except Exception as e:
                logger.error(f"DDGS search failed: {e}")
                return []
        
        loop = asyncio.get_event_loop()
        results_raw = await loop.run_in_executor(None, _search)
        
        results = []
        for item in results_raw:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("href", ""),
                snippet=item.get("body", "")[:300],
            ))
        
        return results


class BraveSearchProvider(SearchProvider):
    """Brave Search API provider (requires API key).
    
    Optional alternative that requires a Brave Search API key.
    Free tier available at https://brave.com/search/api/
    """

    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Search using Brave Search API.
        
        Args:
            query: Search query string.
            num_results: Number of results to return.
        
        Returns:
            List of search results.
        """
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }

        params = {
            "q": query,
            "count": num_results,
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.API_URL,
                headers=headers,
                params=params,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

        results = []
        web_results = data.get("web", {}).get("results", [])

        for item in web_results[:num_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", "")[:300],
            ))

        return results


class WebToolManager:
    """Manages browser lifecycle and search providers."""

    def __init__(self) -> None:
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._search_provider: SearchProvider | None = None

    async def start(self) -> None:
        """Start the browser (needed for read_webpage)."""
        if self._browser is not None:
            return

        from playwright.async_api import async_playwright

        settings = get_settings()

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=settings.browser_headless,
        )
        logger.info("Browser started")

    async def stop(self) -> None:
        """Stop the browser."""
        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if hasattr(self, "_playwright"):
            await self._playwright.stop()

        logger.info("Browser stopped")

    async def get_context(self) -> BrowserContext:
        """Get or create a browser context (incognito)."""
        if self._browser is None:
            await self.start()

        # Create fresh context for each request (incognito)
        if self._context:
            await self._context.close()

        self._context = await self._browser.new_context()
        return self._context

    def get_search_provider(self) -> SearchProvider:
        """Get the search provider.
        
        Returns DDGSProvider by default (most reliable, no API key).
        """
        if self._search_provider is None:
            self._search_provider = DDGSProvider()
        return self._search_provider

    def set_search_provider(self, provider: SearchProvider) -> None:
        """Set a custom search provider (e.g., BraveSearchProvider)."""
        self._search_provider = provider


# Global manager instance
_manager: WebToolManager | None = None


def get_web_manager() -> WebToolManager:
    """Get the global web tool manager."""
    global _manager
    if _manager is None:
        _manager = WebToolManager()
    return _manager


@tool(
    name="web_search",
    description="Search the web for information. Returns titles, URLs, and snippets.",
    timeout=30.0,
)
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web.
    
    Args:
        query: Search query string.
        num_results: Number of results to return (default: 5, max: 10).
    """
    manager = get_web_manager()
    provider = manager.get_search_provider()

    num_results = min(max(1, num_results), 10)

    try:
        results = await provider.search(query, num_results)

        if not results:
            return f"No results found for '{query}'"

        formatted = [f"Search results for '{query}':\n"]
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. {result.format()}\n")

        return "\n".join(formatted)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Error: Search failed for '{query}': {e}"


@tool(
    name="read_webpage",
    description="Read and extract content from a webpage. Returns clean markdown text.",
    max_result_tokens=8000,
    timeout=30.0,
)
async def read_webpage(url: str, max_tokens: int = 4000) -> str:
    """Read a webpage and extract its content.
    
    Args:
        url: URL of the webpage to read.
        max_tokens: Maximum tokens to return (default: 4000).
    """
    manager = get_web_manager()
    token_counter = get_token_counter()

    try:
        context = await manager.get_context()
        page = await context.new_page()

        try:
            # Navigate to page
            await page.goto(url, timeout=15000, wait_until="domcontentloaded")

            # Try to wait for network idle (but don't fail if it times out)
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass

            # Get HTML content
            html = await page.content()

        finally:
            await page.close()

        # Extract content using trafilatura
        try:
            import trafilatura
            content = trafilatura.extract(
                html,
                include_links=True,
                include_images=False,
                include_tables=True,
            )
        except Exception:
            content = None

        # Fall back to BeautifulSoup if trafilatura fails
        if not content:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")

                # Remove script and style elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()

                content = soup.get_text(separator="\n", strip=True)
            except Exception as e:
                return f"Error extracting content from '{url}': {e}"

        if not content:
            return f"Error: Could not extract content from '{url}'"

        # Convert to markdown if possible
        try:
            import markdownify
            content = markdownify.markdownify(content, heading_style="ATX")
        except Exception:
            pass

        # Truncate to token limit
        content = token_counter.truncate_to_tokens(content, max_tokens)

        return f"Content from {url}:\n\n{content}"

    except Exception as e:
        error_msg = str(e).lower()

        if "timeout" in error_msg:
            return f"Error: Page timed out: {url}"
        elif "403" in error_msg or "forbidden" in error_msg:
            return f"Error: Access denied: {url}"
        elif "404" in error_msg or "not found" in error_msg:
            return f"Error: Page not found: {url}"
        else:
            return f"Error reading '{url}': {e}"


async def cleanup_web_tools() -> None:
    """Clean up web tool resources."""
    global _manager
    if _manager:
        await _manager.stop()
        _manager = None


__all__ = [
    "SearchResult",
    "SearchProvider",
    "DDGSProvider",
    "BraveSearchProvider",
    "WebToolManager",
    "get_web_manager",
    "web_search",
    "read_webpage",
    "cleanup_web_tools",
]

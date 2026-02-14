"""web_search tool - search the internet for external information.

Direct tool (not a subagent). Uses DuckDuckGo HTML for search.
Returns summarised results: titles, URLs, and snippets.
"""

import html
import random
import re
import urllib.request
from urllib.parse import quote_plus

from langchain_core.tools import tool

from ..logging_config import get_logger

logger = get_logger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]


def _perform_web_search(query: str, max_results: int = 5, timeout: int = 10) -> list[dict]:
    """Perform DuckDuckGo HTML search and return parsed results.

    Args:
        query: Search query
        max_results: Maximum number of results to return
        timeout: Request timeout in seconds

    Returns:
        List of dicts with keys: title, url, snippet
    """
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        req = urllib.request.Request(
            search_url,
            headers={"User-Agent": random.choice(USER_AGENTS)},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_text = response.read().decode("utf-8")

        # Extract result blocks - DuckDuckGo wraps each result in result class
        # Titles: links with result__a class
        titles = re.findall(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>',
            response_text,
            re.DOTALL,
        )
        # URLs: result__url has the display URL (domain format)
        urls = re.findall(
            r'<a[^>]*class="[^"]*result__url[^"]*"[^>]*>([^<]*)</a>',
            response_text,
            re.DOTALL,
        )

        # Snippets: result__snippet
        snippets = re.findall(
            r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
            response_text,
            re.DOTALL,
        )

        results = []
        for i in range(min(len(titles), max_results)):
            title = ""
            if i < len(titles):
                title = re.sub(r"<[^>]+>", "", titles[i]).strip()
                title = html.unescape(title)

            url = ""
            if i < len(urls):
                url = urls[i].strip()
                if url and not url.startswith("http"):
                    url = f"https://{url}"

            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
                snippet = html.unescape(snippet)

            if title or url:
                results.append({"title": title, "url": url, "snippet": snippet})

        return results

    except Exception as e:
        logger.warning("web_search failed: %s", e)
        return []


def _format_results(results: list[dict], max_chars: int = 4000) -> str:
    """Format search results for agent consumption."""
    if not results:
        return "No search results found."

    lines = [f"Found {len(results)} results:\n"]
    total = 0
    for i, r in enumerate(results, 1):
        block = f"\n{i}. {r['title']}\n   URL: {r['url']}\n"
        if r.get("snippet"):
            block += f"   {r['snippet']}\n"
        if total + len(block) > max_chars:
            lines.append("\n... (results truncated)")
            break
        lines.append(block)
        total += len(block)

    return "".join(lines)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the internet for external information.

    Use this for API docs, library patterns, language references, or any
    information not in the codebase.

    Args:
        query: Search query (e.g., "Godot 4 GDScript signals documentation",
               "Python asyncio best practices", "LangChain tool calling")
        max_results: Maximum number of results to return (default 5)

    Returns:
        Summarised search results with titles, URLs, and snippets
    """
    try:
        logger.debug("web_search invoked: %s", query[:80])
        results = _perform_web_search(query, max_results=max_results)
        formatted = _format_results(results)
        logger.debug("web_search completed: %d results", len(results))
        return formatted
    except Exception as e:
        logger.warning("web_search failed: %s", e)
        return f"web_search error: {e}"

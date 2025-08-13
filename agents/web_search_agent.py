import os
import time
from typing import List

_TAVILY_OK = bool(os.getenv("TAVILY_API_KEY"))


def _tavily_search(query: str, max_results: int = 5) -> List[str]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    res = client.search(query=query, max_results=max_results)
    lines = []
    for r in res.get("results", []):
        lines.append(f"- {r.get('title')}: {r.get('url')}")
    return lines


def _ddg_search(query: str, max_results: int = 5) -> List[str]:
    from duckduckgo_search import DDGS
    delays = [0.0, 0.5, 1.0, 2.0]
    for d in delays:
        try:
            if d:
                time.sleep(d)
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            lines = []
            for r in results:
                title = r.get('title')
                href = r.get('href')
                lines.append(f"- {title}: {href}")
            return lines
        except Exception:
            continue
    return ["(web search rate-limited"]


def search_web(query: str, max_results: int = 5) -> str:
    try:
        lines = _tavily_search(query, max_results) if _TAVILY_OK else _ddg_search(query, max_results)
        return "".join(lines) if lines else "No web results found."
    except Exception as e:
        return f"(web search failed: {e.__class__.__name__}: {e})"

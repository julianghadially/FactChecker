"""Serper API service for Google Search."""

import requests
from dataclasses import dataclass
from typing import Optional
from src.context_.context import serper_key
import time
from typing import Literal


@dataclass
class SearchResult:
    """A single search result from Serper."""
    title: str
    link: str
    snippet: str
    position: int


class SerperService:
    """Service for Google Search via Serper API.

    Attributes:
        api_key: Serper API key for authentication.
    """

    BASE_URL = "https://google.serper.dev/search"
    NEWS_URL = "https://google.serper.dev/news"

    def __init__(self):
        """Initialize the Serper service.

        Args:
            api_key: Serper API key.
        """
        self.api_key = serper_key

    def search(
        self,
        query: str,
        num_results: int = 10,
        country: str = "us"
    ) -> list[SearchResult]:
        """Execute a Google search and return structured results.

        Args:
            query: Search query string.
            num_results: Number of results to return (max 100).
            country: Country code for localized results.

        Returns:
            List of SearchResult objects.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": num_results,
            "gl": country
        }

        start_time = time.time()
        response = requests.post(self.BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        results = []

        for i, item in enumerate(data.get("organic", [])):
            results.append(SearchResult(
                title=item.get("title", ""),
                link=item.get("link", ""),
                snippet=item.get("snippet", ""),
                position=i + 1
            ))

        print(f"Serper search time. Query: {query}. \nTime: {time.time() - start_time:.2f} seconds")
        return results

    def search_news(
        self,
        query: str,
        recency: Literal["m", "w", "d", ""] = "m"
    ) -> list[dict]:
        """Execute a Google News search and return news articles.

        Args:
            query: Search query string.
            recency: Time filter - "m" (month), "w" (week), "d" (day), or "" (all time).

        Returns:
            List of news article dictionaries with keys: title, link, snippet, source, date, etc.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        if recency:
            payload = {
                "q": query,
                "tbm": recency
            }
        else:
            payload = {
                "q": query
            }

        start_time = time.time()
        response = requests.post(self.NEWS_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Extract news articles from response
        articles = data.get("news", [])
        
        print(f"Serper news search time. Query: {query}. Found {len(articles)} articles. \nTime: {time.time() - start_time:.2f} seconds")
        return articles

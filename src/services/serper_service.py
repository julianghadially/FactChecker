"""Serper API service for Google Search."""

import requests
from dataclasses import dataclass
from typing import Optional
from src.context_.context import serper_key
import time


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

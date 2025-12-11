from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService
from src.context_.context import newsapi_key
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Literal
import requests
import time


def fetch_articles_for_company(company: str, from_date: str, max_articles: int = 5) -> List[Dict]:
    """Fetch articles for a specific company from NewsAPI.
    
    Args:
        company: Company name to search for
        from_date: Start date in YYYY-MM-DD format
        api_key: NewsAPI key
        max_articles: Maximum articles to fetch per company
        
    Returns:
        List of article dictionaries
    """
    url = "https://newsapi.org/v2/everything"
    articles = []
    page = 1
    
    while len(articles) < max_articles:
        params = {
            'q': company,
            'from': from_date,
            'sortBy': 'popularity',
            'apiKey': newsapi_key,
            'page': page,
            'pageSize': min(100, max_articles - len(articles))  # NewsAPI max is 100 per page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'ok':
                print(f"Warning: API returned status {data.get('status')} for {company}")
                break
            
            articles_batch = data.get('articles', [])
            if not articles_batch:
                break  # No more articles
            
            articles.extend(articles_batch)
            
            # Check if we've reached the total available
            total_results = data.get('totalResults', 0)
            if len(articles) >= total_results or len(articles) >= max_articles:
                break
            
            page += 1
            time.sleep(1)  # Rate limiting - NewsAPI allows 100 requests per day on free tier
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching articles for {company}: {e}")
            break
    
    return articles[:max_articles]

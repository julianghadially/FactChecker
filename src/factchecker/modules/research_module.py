"""Research module for web-based evidence retrieval."""

import dspy
from typing import Optional
from src.services import SerperService, FirecrawlService
from src.factchecker.signatures.research import SearchQueryGenerator, EvidenceSummarizer
from src.factchecker.models.data_types import ResearchResult


class ResearchModule(dspy.Module):
    """Module that retrieves web evidence for fact-checking.

    This module orchestrates web search and scraping to gather evidence for
    verifying statements. It generates optimized search queries, retrieves
    search results, scrapes the top sources, and summarizes the evidence.
    """

    def __init__(self, num_queries: int = 2, num_sources: int = 5):
        """Initialize the research module.

        Args:
            num_queries: Number of search queries to generate and use (default: 2).
            num_sources: Maximum number of unique sources to scrape (default: 5).
        """
        super().__init__()
        self.query_generator = dspy.Predict(SearchQueryGenerator)
        self.evidence_summarizer = dspy.ChainOfThought(EvidenceSummarizer)
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()
        self.num_queries = num_queries
        self.num_sources = num_sources

    def forward(self, statement: str) -> ResearchResult:
        """Research a statement using web search and scraping.

        Args:
            statement: The statement to research and verify.

        Returns:
            ResearchResult containing search queries, sources, evidence summary,
            and success status.

        Process:
            1. Generate search queries using LLM
            2. Search for sources using SerperService
            3. Scrape top results using FirecrawlService
            4. Summarize evidence using LLM
        """
        try:
            # Step 1: Generate search queries
            query_result = self.query_generator(statement=statement)
            queries = [query_result.query1, query_result.query2][:self.num_queries]

            # Step 2: Search for sources
            all_results = []
            seen_urls = set()

            for query in queries:
                search_results = self.serper.search(query=query, num_results=10)
                for result in search_results:
                    if result.link not in seen_urls and len(all_results) < self.num_sources:
                        seen_urls.add(result.link)
                        all_results.append({
                            'url': result.link,
                            'title': result.title,
                            'snippet': result.snippet,
                            'position': result.position
                        })

            # Step 3: Scrape top sources
            sources = []
            for result in all_results[:self.num_sources]:
                scraped = self.firecrawl.scrape(
                    url=result['url'],
                    max_length=5000,  # Limit to avoid token overflow
                    skip_pdfs=True
                )

                if scraped.success:
                    sources.append({
                        'url': result['url'],
                        'title': result['title'],
                        'snippet': result['snippet'],
                        'content': scraped.markdown
                    })

            # Step 4: Combine and summarize evidence
            if sources:
                combined_content = "\n\n---\n\n".join([
                    f"Source: {s['title']}\nURL: {s['url']}\n\n{s['content']}"
                    for s in sources
                ])

                summary_result = self.evidence_summarizer(
                    statement=statement,
                    raw_content=combined_content
                )
                evidence_summary = summary_result.summary
            else:
                evidence_summary = "No evidence found from web sources."

            return ResearchResult(
                statement=statement,
                search_queries=queries,
                sources=sources,
                evidence_summary=evidence_summary,
                success=True
            )

        except Exception as e:
            return ResearchResult(
                statement=statement,
                search_queries=[],
                sources=[],
                evidence_summary="",
                success=False,
                error=str(e)
            )

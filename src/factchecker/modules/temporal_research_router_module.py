"""Temporal-aware research router that intelligently routes to news search for time-sensitive claims."""

import re
from datetime import datetime
from typing import Literal, Optional, Tuple
import dspy
from .research_agent_module import ResearchAgentModule


class TemporalResearchRouterModule(dspy.Module):
    """Research router with intelligent temporal detection and news search routing.

    Analyzes claims for temporal signals (recent dates, temporal phrases, company/market news)
    and automatically routes to Google News search with appropriate recency filters instead
    of regular web search when temporal signals are detected.

    Attributes:
        research_agent: The underlying ResearchAgentModule to delegate to.
        current_year: Current year for date context.
        current_month: Current month for date context.
    """

    # Temporal signal patterns
    TEMPORAL_PHRASES = [
        # Recent actions/events
        r'\b(has|have)\s+(opened|launched|announced|released|upgraded|introduced|unveiled|rolled out)\b',
        r'\b(recently|just|newly)\s+(opened|launched|announced|released|upgraded|introduced)\b',
        r'\bjust\s+(opened|launched|released|upgraded)\b',

        # Legal/governance actions
        r'\b(ruled|declared|ordered|mandated|approved|rejected|signed|passed)\b',
        r'\bcourt\s+(ruled|declared|ordered|decided)\b',
        r'\bgovernment\s+(announced|declared|approved|mandated)\b',

        # Market/company news indicators
        r'\b(IPO|acquisition|merger|bankruptcy|layoffs|hiring|rebranding)\b',
        r'\b(stock|shares|market|trading|quarterly|earnings)\b',
        r'\b(CEO|executive|founder|board)\s+(announced|resigned|appointed|stepped down)\b',

        # Recent temporal markers
        r'\b(this|last)\s+(week|month|quarter|year)\b',
        r'\b(today|yesterday|now)\b',
        r'\bin\s+\d{4}\b',  # "in 2024"
    ]

    # Date patterns for extracting specific dates
    DATE_PATTERNS = [
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b',
    ]

    # Very recent indicators (use daily recency)
    VERY_RECENT_PHRASES = [
        r'\b(today|yesterday|this week|just|breaking)\b',
        r'\bhas\s+(just|recently)\b',
    ]

    # Recent indicators (use weekly recency)
    RECENT_PHRASES = [
        r'\b(this month|last week|recently|newly|latest)\b',
        r'\b(announced|released|launched)\s+(recently|this)\b',
    ]

    def __init__(self, max_page_visits: int = 3):
        """Initialize the temporal research router.

        Args:
            max_page_visits: Maximum pages to visit per search query (passed to ResearchAgentModule).
        """
        super().__init__()
        self.research_agent = ResearchAgentModule(max_page_visits=max_page_visits)

        # Get current date context
        now = datetime.now()
        self.current_year = now.year
        self.current_month = now.strftime("%B")
        self.current_month_num = now.month

    def _detect_temporal_signals(self, claim: str) -> Tuple[bool, str]:
        """Detect if claim contains temporal signals requiring news search.

        Args:
            claim: The claim text to analyze.

        Returns:
            Tuple of (has_temporal_signals, recency_filter) where recency_filter
            is "d" (day), "w" (week), "m" (month), or "" (not temporal).
        """
        claim_lower = claim.lower()

        # Check for very recent indicators (daily recency)
        for pattern in self.VERY_RECENT_PHRASES:
            if re.search(pattern, claim_lower, re.IGNORECASE):
                return True, "d"

        # Check for recent indicators (weekly recency)
        for pattern in self.RECENT_PHRASES:
            if re.search(pattern, claim_lower, re.IGNORECASE):
                return True, "w"

        # Check for general temporal phrases (monthly recency)
        for pattern in self.TEMPORAL_PHRASES:
            if re.search(pattern, claim_lower, re.IGNORECASE):
                return True, "m"

        # Check for specific dates
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, claim, re.IGNORECASE)
            if match:
                # Extract year from the date
                year_match = re.search(r'\d{4}', match.group())
                if year_match:
                    year = int(year_match.group())
                    # If it's current year or last year, it's temporal
                    if year >= self.current_year - 1:
                        return True, "m"

        # Check for current year mentions
        if str(self.current_year) in claim or str(self.current_year - 1) in claim:
            return True, "m"

        return False, ""

    def _enrich_query_with_temporal_context(
        self,
        query: str,
        claim: str,
        recency: str
    ) -> str:
        """Enrich search query with temporal context.

        Args:
            query: Original search query.
            claim: Original claim (for extracting dates).
            recency: Recency filter being used ("d", "w", "m").

        Returns:
            Enhanced query with temporal context.
        """
        # Don't modify if query already has year
        if str(self.current_year) in query or str(self.current_year - 1) in query:
            return query

        # Extract specific dates from claim
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, claim, re.IGNORECASE)
            if match:
                date_str = match.group()
                # If query doesn't have this date, consider adding year context
                if date_str not in query:
                    year_match = re.search(r'\d{4}', date_str)
                    if year_match:
                        year = year_match.group()
                        # Only add year if it's recent
                        if int(year) >= self.current_year - 1:
                            return f"{query} {year}"

        # For very recent claims, add current month and year
        if recency == "d":
            return f"{query} {self.current_month} {self.current_year}"

        # For recent claims, add current year
        if recency in ["w", "m"]:
            return f"{query} {self.current_year}"

        return query

    def forward(self, claim: str, query: str) -> dspy.Prediction:
        """Route research request based on temporal signal detection.

        Analyzes the claim for temporal signals and automatically routes to
        news search with appropriate recency filters when temporal indicators
        are detected. Falls back to regular web search for non-temporal claims.

        Args:
            claim: The claim being fact-checked.
            query: Search query to execute.

        Returns:
            Evidence from research (same format as ResearchAgentModule).
        """
        # Detect temporal signals in the claim
        is_temporal, recency = self._detect_temporal_signals(claim)

        if is_temporal and recency:
            # Enrich query with temporal context
            enriched_query = self._enrich_query_with_temporal_context(query, claim, recency)

            print(f"[TemporalRouter] Detected temporal claim with recency '{recency}'")
            print(f"[TemporalRouter] Original query: {query}")
            print(f"[TemporalRouter] Enriched query: {enriched_query}")
            print(f"[TemporalRouter] Routing to NEWS search")

            # Route to news search
            return self.research_agent(
                claim=claim,
                query=enriched_query,
                use_news_search=True,
                news_recency=recency
            )
        else:
            print(f"[TemporalRouter] No temporal signals detected")
            print(f"[TemporalRouter] Routing to REGULAR web search")

            # Route to regular web search
            return self.research_agent(
                claim=claim,
                query=query,
                use_news_search=False
            )

"""Temporal routing module that intelligently routes between JudgeModule and FactCheckerPipeline.

This module analyzes input statements for temporal references and provided URLs to determine
whether to use the fast JudgeModule (for statements within LLM knowledge) or the full
FactCheckerPipeline with web research (for statements requiring current information).
"""

import re
from datetime import datetime
from typing import Optional
import dspy

from src.factchecker.simple.modules.judge_module import JudgeModule
from src.factchecker.modules.fact_checker_pipeline import FactCheckerPipeline


class TemporalRouterModule(dspy.Module):
    """Routes fact-checking requests based on temporal references and provided URLs.

    The router analyzes the input to determine if web research is needed:
    1. Extracts dates and temporal references from the statement
    2. Compares dates against the LLM knowledge cutoff (June 2024)
    3. Checks for provided URLs in the input
    4. Routes to FactCheckerPipeline if dates are recent/future OR URLs are provided
    5. Routes to simple JudgeModule otherwise for fast evaluation

    Attributes:
        knowledge_cutoff: Date representing the LLM's knowledge cutoff (default: June 2024)
        judge: Simple judge module for fast evaluation without web research
        pipeline: Full fact-checking pipeline with web research capabilities
    """

    # LLM knowledge cutoff date (can be adjusted based on model)
    KNOWLEDGE_CUTOFF = datetime(2024, 6, 1)

    def __init__(
        self,
        max_judge_iterations: int = 3,
        max_page_visits: int = 3,
        knowledge_cutoff: Optional[datetime] = None
    ):
        """Initialize the temporal router.

        Args:
            max_judge_iterations: Max search iterations per claim in pipeline.
            max_page_visits: Max pages to visit per search query in pipeline.
            knowledge_cutoff: Custom knowledge cutoff date (defaults to June 2024).
        """
        super().__init__()

        self.knowledge_cutoff = knowledge_cutoff or self.KNOWLEDGE_CUTOFF

        # Initialize both modules
        self.judge = JudgeModule()
        self.pipeline = FactCheckerPipeline(
            max_judge_iterations=max_judge_iterations,
            max_page_visits=max_page_visits
        )

        # Date pattern regex - matches various date formats
        self.date_patterns = [
            # YYYY-MM-DD, YYYY/MM/DD
            r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b',
            # Month DD, YYYY (e.g., "January 15, 2025")
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
            # Mon DD, YYYY (e.g., "Jan 15, 2025")
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})\b',
            # DD Month YYYY (e.g., "15 January 2025")
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
            # Just year (e.g., "in 2025")
            r'\bin\s+(\d{4})\b',
            r'\bof\s+(\d{4})\b',
            r'\byear\s+(\d{4})\b',
        ]

        # Temporal keywords that suggest recent/current events
        self.temporal_keywords = [
            r'\btoday\b',
            r'\byesterday\b',
            r'\btomorrow\b',
            r'\bthis\s+(week|month|year)\b',
            r'\blast\s+(week|month|year)\b',
            r'\bnext\s+(week|month|year)\b',
            r'\bcurrent\b',
            r'\brecent(ly)?\b',
            r'\blatest\b',
            r'\bnow\b',
            r'\bpresent\b',
            r'\bupcoming\b',
            r'\b2024\b',  # Post-cutoff year
            r'\b2025\b',
            r'\b2026\b',
        ]

        # Month name to number mapping
        self.month_map = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12,
        }

    def _extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text.

        Args:
            text: Input text to search for URLs.

        Returns:
            List of extracted URLs.
        """
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        return urls

    def _extract_dates(self, text: str) -> list[datetime]:
        """Extract and parse dates from text.

        Args:
            text: Input text to search for dates.

        Returns:
            List of parsed datetime objects.
        """
        dates = []
        text_lower = text.lower()

        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    groups = match.groups()

                    # Handle different date format patterns
                    if len(groups) == 3 and groups[0].isdigit() and len(groups[0]) == 4:
                        # YYYY-MM-DD format
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        dates.append(datetime(year, month, day))

                    elif len(groups) == 3 and groups[0].lower() in self.month_map:
                        # Month DD, YYYY format
                        month = self.month_map[groups[0].lower()]
                        day = int(groups[1])
                        year = int(groups[2])
                        dates.append(datetime(year, month, day))

                    elif len(groups) == 3 and groups[1].lower() in self.month_map:
                        # DD Month YYYY format
                        day = int(groups[0])
                        month = self.month_map[groups[1].lower()]
                        year = int(groups[2])
                        dates.append(datetime(year, month, day))

                    elif len(groups) == 1 and groups[0].isdigit():
                        # Just year
                        year = int(groups[0])
                        dates.append(datetime(year, 1, 1))

                except (ValueError, IndexError):
                    # Skip invalid dates
                    continue

        return dates

    def _has_temporal_keywords(self, text: str) -> bool:
        """Check if text contains temporal keywords suggesting current events.

        Args:
            text: Input text to analyze.

        Returns:
            True if temporal keywords are found, False otherwise.
        """
        text_lower = text.lower()
        for pattern in self.temporal_keywords:
            if re.search(pattern, text_lower):
                return True
        return False

    def _should_use_web_research(
        self,
        statement: str,
        urls: list[str],
        dates: list[datetime]
    ) -> tuple[bool, str]:
        """Determine if web research is needed based on temporal analysis and URLs.

        Args:
            statement: The input statement.
            urls: Extracted URLs from input.
            dates: Extracted dates from statement.

        Returns:
            Tuple of (should_use_web_research: bool, reason: str)
        """
        # Rule 1: URLs provided - use web research to leverage them
        if urls:
            return True, f"URLs provided ({len(urls)} URLs found)"

        # Rule 2: Check for dates beyond knowledge cutoff
        for date in dates:
            if date >= self.knowledge_cutoff:
                return True, f"Date beyond knowledge cutoff: {date.strftime('%Y-%m-%d')} >= {self.knowledge_cutoff.strftime('%Y-%m-%d')}"

        # Rule 3: Check for temporal keywords suggesting recent events
        if self._has_temporal_keywords(statement):
            return True, "Temporal keywords suggest recent/current events"

        # Default: use fast judge module
        return False, "No temporal references or URLs requiring web research"

    def forward(self, statement: str, urls: Optional[list[str]] = None) -> dspy.Prediction:
        """Route the fact-checking request to appropriate module.

        Args:
            statement: The statement to fact-check.
            urls: Optional list of URLs to use as priority evidence sources.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
                - route_decision: Which module was used (judge or pipeline)
                - route_reason: Why that route was chosen

            For pipeline route, also includes:
                - claims: List of extracted claims
                - claim_results: Detailed results for each claim
        """
        # Extract URLs from statement if not provided
        if urls is None:
            urls = self._extract_urls(statement)

        # Extract dates from statement
        dates = self._extract_dates(statement)

        # Determine routing
        use_web_research, reason = self._should_use_web_research(statement, urls, dates)

        # Log routing decision
        print(f"\n{'='*60}")
        print("TEMPORAL ROUTING DECISION")
        print(f"{'='*60}")
        print(f"Statement: {statement[:100]}...")
        print(f"URLs found: {len(urls)}")
        if urls:
            for url in urls[:3]:  # Show first 3 URLs
                print(f"  - {url}")
        print(f"Dates found: {len(dates)}")
        if dates:
            for date in dates[:3]:  # Show first 3 dates
                print(f"  - {date.strftime('%Y-%m-%d')}")
        print(f"Route: {'FactCheckerPipeline (with web research)' if use_web_research else 'JudgeModule (fast evaluation)'}")
        print(f"Reason: {reason}")
        print(f"{'='*60}\n")

        # Route to appropriate module
        if use_web_research:
            # Use full pipeline with web research
            result = self.pipeline(statement=statement, priority_urls=urls if urls else None)

            # Add routing metadata
            result.route_decision = "pipeline"
            result.route_reason = reason

            return result
        else:
            # Use simple judge for fast evaluation
            result = self.judge(statement=statement)

            # Add routing metadata
            result.route_decision = "judge"
            result.route_reason = reason

            return result

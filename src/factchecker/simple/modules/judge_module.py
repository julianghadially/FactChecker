"""Simple judge module - barebones fact checker without research."""

import dspy
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.factchecker.simple.signatures.judge import Judge
from src.factchecker.simple.signatures.judge_with_context import JudgeWithContext
from src.services.serper_service import SerperService


class JudgeModule(dspy.Module):
    """Barebones fact checker that judges statements without research.

    Takes a statement as input and outputs a verdict directly using LLM knowledge.
    If the LLM indicates knowledge cutoff limitations or uncertainty, performs
    a web search and re-evaluates with the additional context.

    This serves as a simpler/faster alternative to the full FactCheckerPipeline
    for cases where external research is not needed or desired, but can handle
    recent events when needed.
    """

    def __init__(self, enable_web_search: bool = True):
        """Initialize the simple judge module.

        Args:
            enable_web_search: If True, performs web search when knowledge limitations
                             are detected. If False, behaves as original judge-only module.
        """
        super().__init__()
        self.judge = dspy.ChainOfThought(Judge)
        self.judge_with_context = dspy.ChainOfThought(JudgeWithContext)
        self.enable_web_search = enable_web_search
        if enable_web_search:
            self.serper = SerperService()

    def forward(self, statement: str) -> dspy.Prediction:
        """Evaluate a statement for factual correctness.

        First attempts to judge using LLM knowledge. If the reasoning indicates
        knowledge cutoff limitations, uncertainty, or inability to verify, performs
        a web search and re-evaluates with the additional context.

        Args:
            statement: The statement to evaluate.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
                - web_search_performed: Boolean indicating if web search was used
        """
        # Step 1: Initial judgment using LLM knowledge
        result = self.judge(statement=statement)

        # Step 2: Check if web search is needed
        needs_search = self.enable_web_search and self._detect_knowledge_limitations(
            result.reasoning, result.verdict, statement
        )

        if needs_search:
            # Step 3: Perform web search for recent information
            search_results = self._perform_web_search(statement)

            if search_results:
                # Step 4: Re-evaluate with search context
                enhanced_result = self.judge_with_context(
                    statement=statement,
                    search_results=search_results,
                    initial_reasoning=result.reasoning,
                )

                return dspy.Prediction(
                    statement=statement,
                    overall_verdict=enhanced_result.verdict,
                    confidence=enhanced_result.confidence,
                    reasoning=enhanced_result.reasoning,
                    web_search_performed=True,
                )

        # Return original result if no search needed or search failed
        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
            web_search_performed=False,
        )

    def _detect_knowledge_limitations(self, reasoning: str, verdict: str, statement: str = "") -> bool:
        """Detect if the reasoning indicates knowledge cutoff or uncertainty.

        Args:
            reasoning: The LLM's reasoning for its verdict.
            verdict: The verdict assigned.
            statement: The original statement being evaluated (used for temporal detection).

        Returns:
            True if knowledge limitations are detected, False otherwise.
        """
        # Check verdict first - CONTAINS_UNSUPPORTED_CLAIMS suggests uncertainty
        if verdict == "CONTAINS_UNSUPPORTED_CLAIMS":
            return True

        # Check for explicit knowledge limitation patterns in reasoning
        limitation_patterns = [
            r"knowledge cutoff",
            r"training data",
            r"cannot verify",
            r"unable to verify",
            r"don't have.*information",
            r"do not have.*information",
            r"lack.*information",
            r"beyond my knowledge",
            r"after.*202[0-9]",  # References to dates after 2020s
            r"recent.*event",
            r"current.*information",
            r"up-to-date.*information",
            r"latest.*information",
            r"as of.*202[0-9]",
            r"no.*access.*current",
            r"uncertain",
            r"unclear",
            r"may have changed",
            r"could have changed",
            r"might.*changed",
        ]

        reasoning_lower = reasoning.lower()
        for pattern in limitation_patterns:
            if re.search(pattern, reasoning_lower):
                return True

        # Check for temporal references in the statement itself
        temporal_refs = self._extract_temporal_references(statement)
        if temporal_refs['needs_verification']:
            return True

        return False

    def _extract_temporal_references(self, statement: str) -> dict:
        """Extract temporal references from the statement.

        Detects dates and time-sensitive keywords to determine if the statement
        requires current web verification.

        Args:
            statement: The statement to analyze for temporal references.

        Returns:
            Dictionary with:
                - dates: List of detected date objects
                - temporal_keywords: List of detected temporal keywords
                - needs_verification: Boolean indicating if web search is needed
        """
        dates = []
        temporal_keywords = []
        today = datetime.now()
        cutoff_date = today - relativedelta(months=24)  # 24 months ago

        # Pattern 1: YYYY-MM-DD format
        yyyy_mm_dd_pattern = r'\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b'
        for match in re.finditer(yyyy_mm_dd_pattern, statement):
            try:
                date_obj = datetime.strptime(match.group(0), '%Y-%m-%d')
                dates.append(date_obj)
            except ValueError:
                pass

        # Pattern 2: Month YYYY format (e.g., "January 2024", "Jan 2024")
        month_yyyy_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(20\d{2})\b'
        for match in re.finditer(month_yyyy_pattern, statement, re.IGNORECASE):
            try:
                date_obj = datetime.strptime(match.group(0), '%B %Y')
            except ValueError:
                try:
                    date_obj = datetime.strptime(match.group(0), '%b %Y')
                except ValueError:
                    continue
            dates.append(date_obj)

        # Pattern 3: "in 20XX" format
        in_year_pattern = r'\bin\s+(20\d{2})\b'
        for match in re.finditer(in_year_pattern, statement):
            try:
                year = int(match.group(1))
                # For year-only dates, use December 31st to be more inclusive
                # (if any part of the year is recent, we should check)
                date_obj = datetime(year, 12, 31)
                dates.append(date_obj)
            except ValueError:
                pass

        # Pattern 4: Just a year in 2000s (e.g., "2024")
        year_pattern = r'\b(202[0-9]|20[3-9][0-9])\b'
        for match in re.finditer(year_pattern, statement):
            try:
                year = int(match.group(1))
                # For year-only dates, use December 31st to be more inclusive
                # (if any part of the year is recent, we should check)
                date_obj = datetime(year, 12, 31)
                dates.append(date_obj)
            except ValueError:
                pass

        # Temporal keywords that indicate time-sensitive information
        temporal_keyword_patterns = [
            r'\brecent\b',
            r'\brecently\b',
            r'\blatest\b',
            r'\bcurrent\b',
            r'\bcurrently\b',
            r'\bthis year\b',
            r'\blast year\b',
            r'\blast month\b',
            r'\bthis month\b',
            r'\btoday\b',
            r'\bnow\b',
            r'\bpresent\b',
            r'\bup-to-date\b',
            r'\bup to date\b',
            r'\bmodern\b',
            r'\bongoing\b',
            r'\bas of\b',
        ]

        statement_lower = statement.lower()
        for pattern in temporal_keyword_patterns:
            if re.search(pattern, statement_lower):
                match = re.search(pattern, statement_lower)
                if match:
                    temporal_keywords.append(match.group(0))

        # Determine if verification is needed
        needs_verification = False

        # Check if any date is within the last 24 months
        for date in dates:
            if date >= cutoff_date:
                needs_verification = True
                break

        # Check if temporal keywords are present
        if temporal_keywords:
            needs_verification = True

        return {
            'dates': dates,
            'temporal_keywords': temporal_keywords,
            'needs_verification': needs_verification,
        }

    def _perform_web_search(self, statement: str) -> str:
        """Perform web search and format results for the judge.

        Args:
            statement: The statement to search for.

        Returns:
            Formatted search results string, or empty string if search fails.
        """
        try:
            # Extract key entities/terms from the statement for search
            # For now, use the statement directly as the search query
            # Future enhancement: use LLM to extract key search terms
            search_query = statement

            # Perform search (get top 5 results for efficiency)
            results = self.serper.search(query=search_query, num_results=5)

            if not results:
                return ""

            # Format results for LLM consumption
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. {result.title}\n"
                    f"   URL: {result.link}\n"
                    f"   Snippet: {result.snippet}\n"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            print(f"Web search failed: {e}")
            return ""

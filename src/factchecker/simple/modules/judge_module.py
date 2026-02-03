"""Simple judge module - barebones fact checker without research."""

import dspy
import re
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
            result.reasoning, result.verdict
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

    def _detect_knowledge_limitations(self, reasoning: str, verdict: str) -> bool:
        """Detect if the reasoning indicates knowledge cutoff or uncertainty.

        Args:
            reasoning: The LLM's reasoning for its verdict.
            verdict: The verdict assigned.

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

        return False

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

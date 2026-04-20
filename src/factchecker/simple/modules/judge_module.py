"""Simple judge module - barebones fact checker without research."""

import dspy
from src.factchecker.simple.signatures.judge import Judge

class JudgeModule(dspy.Module):
    """Barebones fact checker that judges statements without research.

    Takes a statement as input and outputs a verdict directly using LLM knowledge.
    No claim extraction, no web search, no evidence gathering.

    This serves as a simpler/faster alternative to the full FactCheckerPipeline
    for cases where external research is not needed or desired.
    """

    def __init__(self):
        """Initialize the simple judge module."""
        super().__init__()
        self.judge = dspy.ChainOfThought(Judge)
        # Few-shot demonstrations covering all three verdict classes
        self.judge.demos = [
            {
                "statement": "Coffee is generally grown in tropical and subtropical regions.",
                "reasoning": (
                    "Coffee cultivation is concentrated in the 'Bean Belt' between the Tropics of Cancer and Capricorn, "
                    "spanning major producers like Brazil, Colombia, Ethiopia, Vietnam, and Indonesia. The qualifier "
                    "'generally' accurately hedges the claim: while the vast majority of global coffee production occurs "
                    "in tropical/subtropical zones, some coffee is grown in peripheral regions (e.g., Hawaii, parts of "
                    "southern China). This hedging language does not undermine the core claim — it reflects the actual "
                    "geographic distribution accurately. The statement is broadly true and reflects well-established "
                    "agricultural knowledge. Verdict: SUPPORTED."
                ),
                "verdict": "SUPPORTED",
                "confidence": 0.95,
            },
            {
                "statement": "The Great Wall of China is visible from the Moon with the naked eye.",
                "reasoning": (
                    "This is one of the most widely repeated myths in popular science, but it has been definitively "
                    "refuted. The Great Wall is only about 4–9 meters wide — far too narrow to resolve with the human "
                    "eye from the Moon (~384,400 km away) or even from low Earth orbit (~400 km). Astronauts including "
                    "China's own Yang Liwei confirmed they could not see the Wall from space. NASA has also addressed "
                    "this: the Wall's width is comparable to a human hair when viewed from the Moon. Multiple independent "
                    "scientific and astronaut sources confirm this claim is false. The statement contains a demonstrably "
                    "incorrect factual claim. Verdict: CONTAINS_REFUTED_CLAIMS."
                ),
                "verdict": "CONTAINS_REFUTED_CLAIMS",
                "confidence": 0.98,
            },
            {
                "statement": "A recently patented pharmaceutical compound designated XB-7749 produces fewer adverse effects than existing TNF inhibitors in treating rheumatoid arthritis.",
                "reasoning": (
                    "This claim refers to a specific proprietary compound (XB-7749) and makes a precise comparative "
                    "efficacy claim. I have no reliable information about this particular compound in my world knowledge — "
                    "it does not correspond to any publicly well-documented drug I can identify. Without being able to "
                    "verify the existence of this compound or access its clinical trial data, I cannot confirm or refute "
                    "the specific comparative efficacy claim. This is a case where I genuinely cannot determine truth from "
                    "available world knowledge, not merely a case of vague language. Verdict: CONTAINS_UNSUPPORTED_CLAIMS."
                ),
                "verdict": "CONTAINS_UNSUPPORTED_CLAIMS",
                "confidence": 0.55,
            },
        ]

    def forward(self, statement: str) -> dspy.Prediction:
        """Evaluate a statement for factual correctness.

        Args:
            statement: The statement to evaluate.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
        """
        result = self.judge(statement=statement)

        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )

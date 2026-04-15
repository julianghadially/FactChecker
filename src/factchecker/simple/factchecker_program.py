import dspy
from src.factchecker.simple.modules.judge_module import JudgeModule


class FactCheckerProgram:
    """CodeEvolver-compatible wrapper around the DSPy fact-checking pipeline.

    Instantiated once per sandbox start. All setup (LM config, pipeline assembly)
    happens in __init__ with defaulted kwargs. Called per-row via __call__.
    """

    def __init__(
        self,
        model: str = "openai/gpt-5-mini",
        max_tokens: int = 4000,
    ):
        self.lm = dspy.LM(model, max_tokens=max_tokens)
        dspy.configure(lm=self.lm)

        self.judge = JudgeModule()

    def __call__(self, statement: str) -> dict:
        prediction = self.judge(statement=statement)
        return {
            "statement": prediction.statement,
            "overall_verdict": prediction.overall_verdict,
            "confidence": prediction.confidence,
            "reasoning": prediction.reasoning,
        }


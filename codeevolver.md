PARENT_MODULE_PATH:"src.factchecker.simple.factchecker_program.FactCheckerProgram"

## ARCHITECTURE TITLE: "Direct Judge with Anti-Hedging Signature"

## ARCHITECTURE SUMMARY:
FactCheckerProgram is a single-step fact-checking pipeline that evaluates statements directly using LLM world knowledge. It wraps a JudgeModule (dspy.ChainOfThought over the Judge signature) and exposes a simple __call__ interface returning verdict, confidence, and reasoning. The Judge signature enforces decisive verdicts by strongly preferring SUPPORTED or CONTAINS_REFUTED_CLAIMS over the hedge verdict CONTAINS_UNSUPPORTED_CLAIMS.

## ARCHITECTURE DESCRIPTION:
FactCheckerProgram instantiates a JudgeModule backed by dspy.ChainOfThought(Judge). On each call, the statement is passed directly to the judge, which uses the Judge DSPy Signature to produce chain-of-thought reasoning, a verdict (SUPPORTED / CONTAINS_REFUTED_CLAIMS / CONTAINS_UNSUPPORTED_CLAIMS), and a confidence score. The Judge signature's docstring has been carefully crafted to reduce over-hedging: it instructs the model to mark broadly-true statements as SUPPORTED even when they use vague qualifiers ("typically", "generally", "often", "tend to"), and to reserve CONTAINS_UNSUPPORTED_CLAIMS only for claims where the model genuinely cannot determine truth. No external research, web search, or claim extraction is performed. The pipeline is minimal and fast, trading recall depth for low latency and reduced false UNSUPPORTED verdicts on everyday factual statements.

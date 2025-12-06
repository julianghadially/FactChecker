from dspy import Signature, InputField, OutputField

class ClaimExtractor(Signature):
    """Extract the claims from the statement(s)."""
    statement: str = InputField(desc="the statements to extract claims from")
    claims: list[str] = OutputField(desc="A list of distinct claims")
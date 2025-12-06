from dspy import Signature, InputField, OutputField
from pydantic import BaseModel
from typing import Literal, Optional

class FireJudge(Signature):
    """You are a judge responsible for evaluating the factual correctness of a claim"""
    claims: list[str] = InputField(desc="the statements to extract claims from")
    final_output: Optional[Literal["supported", "not_supported", "refuted"]] = OutputField(desc="The final judgement on veracity of the claim - supported, not supported, or refuted")
    next_search: Optional[str] = OutputField(desc="a single query for gathering evidence from the web")
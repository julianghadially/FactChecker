"""Source deep dive signature for generating site-specific follow-up queries."""

from dspy import Signature, InputField, OutputField


class SourceDeepDive(Signature):
    """Analyze initial evidence and generate site-specific queries for deeper investigation.

    This signature enables multi-hop reasoning by identifying which authoritative sources
    from the initial evidence warrant deeper investigation, then generating targeted
    site-specific queries using the "site:" operator.

    The goal is to drill into promising sources (e.g., foundation websites, university pages,
    official organizations, government sites) to discover additional context that may not
    have appeared in initial broad searches.

    Use cases:
    - Initial evidence mentions "PSEG Foundation gave $100K to TESU" but doesn't explain
      what TESU programs exist → Generate "site:tesu.edu programs scholarships"
    - Evidence cites a research paper but doesn't have full details → Generate
      "site:university.edu research paper title author"
    - Evidence mentions a company partnership but lacks specifics → Generate
      "site:company.com partnership agreement details"

    Source selection criteria:
    - Prioritize authoritative domains (.edu, .gov, .org, official company sites)
    - Focus on sources directly related to the statement's claims
    - Avoid news aggregators or secondary sources
    - Target 1-3 most promising sources (don't overdo it)

    Query generation guidelines:
    - Use "site:domain.com" operator to restrict search to that domain
    - Include specific keywords related to gaps in understanding
    - Keep queries focused on discovering program details, context, or relationships
    - Avoid duplicate information already in evidence
    """

    statement: str = InputField(desc="The statement being fact-checked")
    evidence: str = InputField(desc="Initial evidence gathered from web sources")

    reasoning: str = OutputField(
        desc="Explanation of which sources warrant deeper investigation and why, "
        "including what specific information gaps could be filled by drilling into these sources"
    )
    targeted_site_queries: list[str] = OutputField(
        desc="1-3 site-specific search queries using 'site:' operator to drill into promising "
        "authoritative sources (e.g., 'site:tesu.edu scholarship programs', "
        "'site:pseg.com foundation grants education'). Return empty list if no promising sources found."
    )

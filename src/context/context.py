import os
openai_key = os.getenv("OPENAI_AGENTJUDGEJG_KEY")
print(f"OpenAI key: {openai_key[:5]}...")
serper_key = os.getenv("SERPER_KEY")
firecrawl_key = os.getenv("FIRECRAWL_KEY")
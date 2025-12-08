import json

def clean_llm_outputted_url(url: str) -> str:
    """Clean a URL for LLM use."""
    clean_url = url.strip().strip('"').strip("'")
    
    # Handle JSON object case (e.g., {"anyOf": ["url", null]})
    if clean_url.startswith("{"):
        try:
            parsed = json.loads(clean_url)
            if isinstance(parsed, dict) and "anyOf" in parsed:
                # Extract first non-null URL from anyOf
                for item in parsed["anyOf"]:
                    if item and isinstance(item, str):
                        clean_url = item.strip('"').strip("'")
                        break
        except json.JSONDecodeError:
            pass
    return clean_url
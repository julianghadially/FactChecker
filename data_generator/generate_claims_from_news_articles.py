"""Generate fact-checking claims from news articles using GPT-5.1."""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import openai
from src.context_.context import openai_key
import time
from datetime import datetime


def load_articles(input_file: str) -> List[Dict]:
    """Load articles from JSON file.
    
    Args:
        input_file: Path to JSON file with articles
        
    Returns:
        List of article dictionaries
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_articles_by_topic(articles: List[Dict]) -> Dict[str, List[Dict]]:
    """Group articles by search_topic.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Dictionary mapping search_topic to list of articles
    """
    grouped = defaultdict(list)
    for article in articles:
        topic = article.get('search_topic', 'unknown')
        grouped[topic].append(article)
    return dict(grouped)


def generate_claims_for_topic(articles: List[Dict], model: str = "gpt-5.1") -> Dict:
    """Generate supported and refuted claims from a group of articles using GPT.
    
    Args:
        articles: List of articles for the same search_topic (1-5 articles)
        model: OpenAI model to use
        
    Returns:
        Dictionary with 'supported_claims' and 'refuted_claims' lists
    """
    # Prepare article summaries for the prompt
    article_summaries = []
    for i, article in enumerate(articles, 1):
        title = article.get('title', 'N/A')
        content = article.get('content', '')
        url = article.get('url', '')
        # Truncate content if too long (keep first 2000 chars)
        content_preview = content[:2000] + "..." if len(content) > 2000 else content
        
        article_summaries.append(
            f"Article {i}:\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content_preview}\n"
        )
    
    articles_text = "\n\n".join(article_summaries)
    
    prompt = f"""You are analyzing news articles to generate fact-checking claims for a fact lookup dataset.

Here are the articles for a single search topic:

{articles_text}

Please generate:
1. 1 objective claims that are SUPPORTED by the articles (facts that can be verified as true). Make sure the claims are supported by the articles.
2. 1 objective claims that would be REFUTED by the articles (facts that contradict what's stated in the articles)

We want facts that are:
- Objective truths about a person, place, or a thing.
- Make it specific and verifiable
- Base it directly on information in the articles
- Use clear, factual language

Discernment:
- Please ignore all facts about events. 
- Some articles contain opinions or subjective statements that are not facts. These claims should be ignored.
- Some articles are untrustworthy and related to gossip. They should be ignored.
- Some facts are extremely temporary in nature. They should be ignored. Some person did something yesterday... That's not a helpful fact for our fact checking database.
- Focus on claims that are likely to still be true in one month's time.

Example 1:
- content: Wild new cockpit audio reveals moment Alaska Airlines pilot tried to crash plane mid-flight: 'Just tried to shut our engines off'. Newly released cockpit or flight deck audio captures Joseph Emerson repeatedly saying 'I'm not OK' before a struggle is heard and the flight crew declares an emergency to air traffic control. He was piloting an Alaska Airlines flight with 84 people on board when he shut down its engines in October, 2023.\n\n“I’m not OK,” a distressed-sounding Emerson says repeatedly in the audio obtained by KGW Portland.
- claim: none. 
- comment: All facts about events should be ignored.

Example 2:
- content: United MileagePlus will keep its current Premier elite status qualification requirements for 2026, with no increases or changes to thresholds.\n- Starting February 2026, all Premier members can receive upgrades on award tickets, aligning upgrade eligibility for both cash and mileage bookings.\n- PlusPoints, the upgrade currency for high-level elites, will shift to dynamic pricing in February 2027, making upgrade costs less predictable and dependent on route, demand and timing.\n- Expanded saver award availability will provide more opportunities for Premier members and select cardholders to access lower-priced premium cabin redemptions.\n\n#### What to consider\n\n- Flights on eligible partners count toward Premier status, but at least four segments must be flown with United itself.\n- Dynamic pricing for PlusPoints upgrades may result in higher costs during peak travel periods or on popular routes.\n- Limits remain on how many Premier qualifying points and PlusPoints can be earned through card spending.\n\n### What you'll miss from the article\n\n- A detailed breakdown of how dynamic PlusPoints pricing could affect upgrade strategies and tips for maximizing new upgrade opportunities.\n\nGenerated by AI with support from our editorial team.\n\nShow summary\n\nUnited Airlines started the week with some big news about its [United MileagePlus program]
- claim: Beginning in February 2026, United Airlines MileagePlus Premier members will be eligible for complimentary upgrades on award tickets in the same way as on cash tickets.
- comment: This claim is about a company. It is objectively verifiable. 

If there are not enough facts to generate the requested number of claims, generate fewer. If no claims exist, return empty claims.

Output your response as a JSON object with this exact structure:
{{
    "1": {{"claim":"","label":"true"}},
    "2": {{"claim":"","label":"false"}}
}}

Only output the JSON object, no additional text."""

    try:
        client = openai.OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a fact-checking assistant that generates objective, verifiable claims from news articles. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        # Parse new format: {"1": {"claim":"","label":"true"}, "6": {"claim":"","label":"false"}}
        supported = []
        refuted = []
        
        for key, claim_obj in result.items():
            if isinstance(claim_obj, dict):
                claim_text = claim_obj.get('claim', '').strip()
                label = claim_obj.get('label', '').lower()
                
                if claim_text:  # Only add non-empty claims
                    if label in ['true', '1', 'supported']:
                        supported.append(claim_text)
                    elif label in ['false', '0', 'refuted']:
                        refuted.append(claim_text)
        
        return {
            'supported_claims': supported,
            'refuted_claims': refuted
        }
    except Exception as e:
        print(f"  Error generating claims: {e}")
        return {
            'supported_claims': [],
            'refuted_claims': []
        }


def process_articles_to_claims(
    input_file: str,
    output_file: str,
    model: str = "gpt-5.1",
    topic_limit: int = 2000
) -> None:
    """Process articles and generate claims.
    
    Args:
        input_file: Path to input JSON file with articles
        output_file: Path to output JSON file with claims
        model: OpenAI model to use
    """
    print(f"Loading articles from {input_file}...")
    try:
        articles = load_articles(input_file)
        print(f"Loaded {len(articles)} articles")
    except Exception as e:
        print(f"Error loading articles: {e}")
        return
    
    # Group articles by search_topic
    grouped = group_articles_by_topic(articles)
    print(f"Grouped into {len(grouped)} topics")
    
    # Limit topics to process
    topics_to_process = list(grouped.items())[:topic_limit]
    total_topics = len(topics_to_process)
    print(f"Processing {total_topics} topics (limited from {len(grouped)})")
    
    all_claims = []
    skipped_topics = []
    
    # Process each topic
    for topic_idx, (topic, topic_articles) in enumerate(topics_to_process, 1):
        print(f"\n[{topic_idx}/{total_topics}] Processing topic: {topic} ({len(topic_articles)} articles)")
        
        try:
            # Generate claims for this topic
            claims_result = generate_claims_for_topic(topic_articles, model=model)
            
            supported = claims_result.get('supported_claims', [])
            refuted = claims_result.get('refuted_claims', [])
            
            # Get URLs from articles (use first article's URL for each claim, or combine if multiple)
            urls = [article.get('url', '') for article in topic_articles if article.get('url')]
            # Use first URL, or combine if multiple articles
            url = urls[0] if urls else ''
            if len(urls) > 1:
                url = ', '.join(urls[:3])  # Limit to 3 URLs max
            
            # Create claim entries for supported claims
            for claim in supported:
                if claim and claim.strip():  # Skip empty claims
                    all_claims.append({
                        'topic': topic,
                        'claim': claim.strip(),
                        'label': True,
                        'url': url
                    })
            
            # Create claim entries for refuted claims
            for claim in refuted:
                if claim and claim.strip():  # Skip empty claims
                    all_claims.append({
                        'topic': topic,
                        'claim': claim.strip(),
                        'label': False,
                        'url': url
                    })
            
            print(f"  Generated {len(supported)} supported and {len(refuted)} refuted claims")
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            error_msg = f"Error processing topic '{topic}': {e}"
            print(f"  {error_msg}")
            skipped_topics.append({
                'topic': topic,
                'error': str(e)
            })
            continue
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    #print(f"\nSaving {len(all_claims)} claims to {output_path}...")
    #with open(output_path, 'w', encoding='utf-8') as f:
    #    json.dump(all_claims, f, indent=2, ensure_ascii=False)
    #print(f"Successfully saved {len(all_claims)} claims to JSON")
    
    # save as CSV
    csv_path = output_path.with_suffix('.csv')
    print(f"Saving {len(all_claims)} claims to {csv_path}...")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['topic', 'claim', 'label', 'url']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for claim in all_claims:
            # Convert boolean label to lowercase string
            label_str = 'true' if claim.get('label') else 'false'
            writer.writerow({
                'topic': claim.get('topic', ''),
                'claim': claim.get('claim', ''),
                'label': label_str,
                'url': claim.get('url', '')
            })
    
    print(f"Successfully saved {len(all_claims)} claims to CSV")
    
    # Log skipped topics if any
    if skipped_topics:
        skipped_file = str(output_path).replace('.json', '_skipped.json')
        print(f"\nLogging {len(skipped_topics)} skipped topics to {skipped_file}...")
        with open(skipped_file, 'w', encoding='utf-8') as f:
            json.dump(skipped_topics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate fact-checking claims from news articles"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/fortune500_news_articles_20251210.json",
        help="Input JSON file with articles"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"data/news_claims_{datetime.now().strftime('%Y%m%d')}.json",
        help="Output JSON file for claims"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="OpenAI model to use (default: gpt-5.1)"
    )
    parser.add_argument(
        "--topiclimit",
        type=int,
        default=200,
        help="Number of topics to process"
    )

    args = parser.parse_args()
    
    process_articles_to_claims(
        input_file=args.input,
        output_file=args.output,
        model=args.model,
        topic_limit = args.topiclimit
    )

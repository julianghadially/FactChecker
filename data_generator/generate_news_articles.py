"""Generate current event claims from Fortune 500 companies using Serper News API."""

from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Literal
import time

fortune_500_companies = ['Walmart', 'Amazon', 'UnitedHealth', 'Apple', 'CVS Health', 'Berkshire Hathaway', 'Alphabet', 'Exxon Mobil', 'McKesson', 'Cencora', 'JPMorgan Chase', 'Costco', 'Cigna', 'Microsoft', 'Cardinal Health', 'Chevron', 'Bank of America', 'General Motors', 'Ford Motor', 'Elevance Health', 'Citi', 'Meta', 'Centene', 'Home Depot', 'Fannie Mae', 'Walgreens', 'Kroger', 'Phillips 66', 'Marathon Petroleum', 'Verizon', 'Nvidia', 'Goldman Sachs', 'Wells Fargo', 'Valero Energy', 'Comcast', 'State Farm', 'AT&T', 'Freddie Mac', 'Humana', 'Morgan Stanley', 'Target', 'StoneX', 'Tesla', 'Dell Technologies', 'PepsiCo', 'Walt Disney', 'UPS', 'Johnson & Johnson', 'FedEx', 'Archer Daniels Midland', 'Procter & Gamble', 'Lowes', 'Energy Transfer', 'RTX', 'Albertsons', 'Sysco', 'Progressive', 'American Express', 'Lockheed Martin', 'MetLife', 'HCA Healthcare', 'Prudential Financial', 'Boeing', 'Caterpillar', 'Merck', 'Allstate', 'Pfizer', 'IBM', 'New York Life Insurance', 'Delta Airlines', 'Publix Super Markets', 'Nationwide', 'TD Synnex', 'United Airlines', 'ConocoPhillips', 'TJX', 'AbbVie', 'Enterprise Products Partners', 'Charter Communications', 'Performance Food', 'American Airlines', 'Capital One', 'Cisco Systems', 'HP', 'Tyson Foods', 'Intel', 'Oracle', 'Broadcom', 'Deere', 'Nike', 'Liberty Mutual Insurance', 'Plains GP', 'USAA', 'Bristol-Myers Squibb', 'Ingram Micro', 'General Dynamics', 'Coca-Cola', 'TIAA', 'Travelers', 'Eli Lilly', 'Uber', 'Mass Mutual', 'Dow', 'Thermo Fisher Scientific', 'U.S. Bancorp', 'World Kinect', 'Abbott Laboratories', 'Best Buy', 'Northwestern Mutual', 'Northrop Grumman', 'Molina Healthcare', 'Dollar General', 'Bank of New York', 'Warner Bros. Discovery', 'CHS', 'Netflix', 'Qualcomm', 'General Electric', 'Honeywell', 'Salesforce', 'Philip Morris', 'US Foods', 'D.R. Horton', 'Lithia Motors', 'Mondelez', 'Starbucks', 'Visa', 'CBRE', 'Lennar', 'GE Vernova', 'PNC', 'Cummins', 'Paccar', 'Amgen', 'PBF Energy', 'GuideWell Mutual', 'PayPal', 'United Natural Foods', 'Dollar Tree', 'Nucor', 'Penske Automotive', 'Coupang', 'Hewlett Packard', 'Duke Energy', 'KKR', 'Ferguson', 'Paramount', 'Jabil', 'Gilead Sciences', 'HF Sinclair', 'CarMax', 'Mastercard', 'NRG Energy', 'Arrow Electronics', 'Baker Hughes', 'Southwest Airlines', 'American International', 'Applied Materials', 'Occidental Petroleum', 'AutoNation', 'Southern', 'Hartford Insurance', 'Apollo Global Management', 'Charles Schwab', 'McDonalds', 'Kraft Heinz', 'Advanced Micro Devices', 'Truist Financial', 'Freeport-McMoRan', 'Micron Technology', 'Marriott International', 'Carrier Global', 'NextEra Energy', '3M', 'Marsh & McLennan', 'PG&E', 'Union Pacific', 'Synchrony Financial', 'Block', 'Danaher', 'Avnet', 'Booking s', 'EOG Resources', 'Quanta Services', 'Discover Financial', 'Constellation Energy', 'Genuine Parts', 'Jones Lang LaSalle', 'Lear', 'Live Nation Entertainment', 'Sherwin-Williams', 'Exelon', 'Macys', 'Halliburton', 'Stryker', 'Reinsurance of America', 'Waste Management', 'State Street', 'WESCO International', 'Oneok', 'Adobe', 'American Family Insurance', 'L3Harris Technologies', 'Ross Stores', 'CDW', 'Tenet Healthcare', 'BJs Wholesale Club', 'Fiserv', 'Altria', 'BlackRock', 'Becton Dickinson', 'Colgate-Palmolive', 'Kimberly-Clark', '1 Automotive', 'Parker-Hannifin', 'General Mills', 'Cognizant Technology', 'American Electric Power', 'GE HealthCare Technologies', 'Automatic Data Processing', 'Cleveland-Cliffs', 'Aflac', 'Goodyear Tire & Rubber', 'Corebridge Financial', 'Newmont', 'International Paper', 'AutoZone', 'Lincoln National', 'Pulte', 'Ameriprise Financial', 'Murphy USA', 'Manpower', 'C.H. Robinson Worldwide', 'PPG Industries', 'Edison International', 'Steel Dynamics', 'Loews', 'Emerson Electric', 'Aramark', 'MGM Resorts International', 'Vistra', 'Asbury Automotive', 'W.W. Grainger', 'Global Partners', 'Jacobs Solutions', 'Corteva', 'Peter Kiewit Sons', 'Boston Scientific', 'OReilly Automotive', 'Leidos s', 'Markel', 'Whirlpool', 'Guardian Life', 'Builders FirstSource', 'Ally Financial', 'Targa Resources', 'Fluor', 'Intuit', 'AECOM', 'Edward Jones', 'Kohls', 'Land OLakes', 'Principal Financial', 'Dominion Energy', 'Kyndryl s', 'Republic Services', 'Devon Energy', 'Illinois Tool Works', 'Northern Trust', 'Auto-Owners Insurance', 'Universal Health Services', 'Pacific Life', 'EchoStar', 'Ecolab', 'Cheniere Energy', 'Omnicom', 'Texas Instruments', 'United States Steel', 'Estée Lauder', 'Farmers Insurance Exchange', 'Kenvue', 'IQVIA s', 'Stanley Black & Decker', 'Keurig Dr Pepper', 'United Rentals', 'Consolidated Edison', 'Amphenol', 'Baxter International', 'Kinder Morgan', 'Gap', 'Nordstrom', 'Super Micro Computer', 'First Citizens BancShares', 'Raymond James Financial', 'Lam Research', 'Tractor Supply', 'Caseys General Stores', 'Viatris', 'Mutual of Omaha Insurance', 'EMCOR', 'CSX', 'LKQ', 'Otis Worldwide', 'Sonic Automotive', 'S&P Global', 'Regeneron Pharmaceuticals', 'BorgWarner', 'Fox', 'Reliance', 'Western & Southern Financial', 'Textron', 'Expedia', 'Fidelity National Financial', 'Carvana', 'DXC Technology', 'W.R. Berkley', 'M&T Bank', 'Dicks Sporting Goods', 'Xcel Energy', 'Fifth Third Bancorp', 'Blackstone', 'Sempra', 'Erie Insurance', 'Corning', 'Lumen Technologies', 'FirstEnergy', 'Hess', 'Labcorp s', 'Western Digital', 'Unum', 'DaVita', 'Kellanova', 'Henry Schein', 'Ryder System', 'Community Health Systems', 'Delek US', 'DTE Energy', 'Equitable', 'DuPont', 'LPL Financial s', 'Citizens Financial', 'MasTec', 'AES', 'Berry Global', 'Westlake', 'Norfolk Southern', 'Air Products & Chemicals', 'J.B. Hunt Transport Services', 'Ball', 'Conagra Brands', 'Huntington Bancshares', 'Hormel Foods', 'Eversource Energy', 'Alcoa', 'Entergy', 'Assurant', 'Chewy', 'Wayfair', 'Crown s', 'Avis Budget', 'Intercontinental Exchange', 'Alaska Air', 'GXO Logistics', 'AGCO', 'Graybar Electric', 'Molson Coors Beverage', 'Arthur J. Gallagher', 'Huntington Ingalls Industries', 'International Flavors & Fragrances', 'Darden Restaurants', 'Cincinnati Financial', 'Chipotle Mexican Grill', 'Yum China s', 'Las Vegas Sands', 'Ulta Beauty', 'Caesars Entertainment', 'BrightSpring Health Services', 'Hershey', 'Hilton', 'Mosaic', 'Airbnb', 'Diamondback Energy', 'American Tower', 'Vertex Pharmaceuticals', 'ServiceNow', 'Owens Corning', 'Thrivent Financial for Lutherans', 'Advance Auto Parts', 'Toll Brothers', 'Mohawk Industries', 'Motorola Solutions', 'Oshkosh', 'DoorDash', 'Owens & Minor', 'NVR', 'Interpublic', 'Booz Allen Hamilton', 'Burlington Stores', 'Expeditors International of Washington', 'Lululemon athletica', 'Fidelity National Information', 'Jefferies Financial', 'Williams', 'VF', 'FM', 'Autoliv', 'Westinghouse Air Brake Technologies', 'Public Service Enterprise', 'Dana', 'Ebay', 'Celanese', 'Global Payments', 'News Corp.', 'THOR Industries', 'QVC', 'Icahn Enterprises', 'Constellation Brands', 'Quest Diagnostics', 'KLA', 'QXO Building Products', 'APA', 'A-Mark Precious Metals', 'Biogen', 'Campbells', 'Concentrix', 'Cintas', 'SpartanNash', 'Ace Hardware', 'Analog Devices', 'Eastman Chemical', 'Interactive Brokers', 'Regions Financial', 'JetBlue Airways', 'Zoetis', 'KeyCorp', 'Oscar Health', 'Ovintiv', 'Seaboard', 'Hertz', 'Skechers U.S.A.', 'Altice USA', 'NOV', 'Graphic Packaging', 'Avery Dennison', 'Equinix', 'Insight Enterprises', 'Sirius XM', 'PVH', 'CenterPoint Energy', 'WEC Energy', 'Xylem', 'Franklin Resources', 'PPL', 'Workday', 'Dover', 'Packaging Corp. of America', 'ABM Industries', 'Intuitive Surgical', 'American Financial', 'Rockwell Automation', 'Solventum', 'Old Republic International', 'Securian Financial', 'Prologis', 'J.M. Smucker', 'Taylor Morrison Home', 'XPO', 'Voya Financial', 'Palo Alto Networks', 'Vertiv s', 'Welltower', 'Foot Locker', 'Par Pacific', 'TransDigm', 'Commercial Metals', 'Post s', 'Masco', 'Rush Enterprises', 'KBR', 'Sprouts Farmers Market', 'Williams-Sonoma', 'Zimmer Biomet s', 'CACI International', 'Microchip Technology', 'Watsco', 'Newell Brands', 'ARKO', 'Sanmina', 'Electronic Arts', 'Yum Brands', 'Fastenal', 'CMS Energy', 'Monster Beverage', 'Endeavor', 'Science Applications International', 'Core & Main', 'Howmet Aerospace', 'Ingredion', 'Vulcan Materials'] #'Andersons'


serper_service = SerperService()

def fetch_articles_for_company(company: str, recency: Literal["m", "w", "d"] = "m", max_articles: int = 5) -> List[Dict]:
    """Fetch articles for a specific company from Serper News API.
    
    Args:
        company: Company name to search for
        recency: Time filter - "m" (month), "w" (week), "d" (day), or "" (all time)
        max_articles: Maximum articles to fetch per company
        
    Returns:
        List of article dictionaries in NewsAPI-compatible format
    """
    query = company
    
    try:
        # Fetch news articles from Serper
        serper_articles = serper_service.search_news(
            query=query,
            recency=recency
        )
        
        # Convert Serper format to NewsAPI-compatible format
        articles = []
        for item in serper_articles[:max_articles]:  # Limit to max_articles
            # Map Serper news format to NewsAPI format
            article = {
                'title': item.get('title', ''),
                'description': item.get('snippet', ''),
                'url': item.get('link', ''),
                'publishedAt': item.get('date', ''),
                'source': {
                    'name': item.get('source', '')
                },
                'author': None,  # Serper doesn't provide author
                'content': item.get('snippet', '')  # Will be replaced with full content if fetch_full_content=True
            }
            articles.append(article)
        
        return articles
        
    except Exception as e:
        print(f"Error fetching articles for {company}: {e}")
        return []


def extract_full_content(url: str, firecrawl_service: Optional[FirecrawlService] = None) -> str:
    """Extract full article content from URL using Firecrawl.
    
    Args:
        url: Article URL
        firecrawl_service: Optional FirecrawlService instance (creates new one if None)
        
    Returns:
        Full article content as markdown, or empty string if extraction fails
    """
    if not url:
        return ""
    
    try:
        if firecrawl_service is None:
            firecrawl_service = FirecrawlService()
        
        # Scrape with a larger max_length to get full content
        scraped = firecrawl_service.scrape(url, max_length=50000, skip_pdfs=True)
        
        if scraped.success and scraped.markdown:
            return scraped.markdown
        else:
            print(f"  Warning: Failed to scrape {url}: {scraped.error or 'Unknown error'}")
            return ""
    except Exception as e:
        print(f"  Error extracting content from {url}: {e}")
        return ""



def format_article_for_json(article: Dict, full_content: Optional[str] = None, search_topic: str = '') -> Dict:
    """Format article data for JSON export.
    
    Args:
        article: Raw article dictionary from NewsAPI
        full_content: Optional full content extracted via Firecrawl
        
    Returns:
        Formatted dictionary with required fields
    """
    source_name = article.get('source', {}).get('name', '') if isinstance(article.get('source'), dict) else str(article.get('source', ''))
    
    # Use full content if provided, otherwise fall back to NewsAPI's truncated content
    content = full_content if full_content else article.get('content', '')
    
    return {
        'search_topic': search_topic,
        'source': source_name,
        'author': article.get('author', ''),
        'title': article.get('title', ''),
        'description': article.get('description', ''),
        'url': article.get('url', ''),
        'publishedAt': article.get('publishedAt', ''),
        'content': content
    }


def generate_news_articles(
    output_file: str = "data/fortune500_news_articles.json", 
    num_searches: int = 1,
    num_articles: int = 5,
    fetch_full_content: bool = True
):
    """Generate news articles from Fortune 500 companies using Serper News API.
    
    Args:
        output_file: Path to output JSON file
        num_searches: Number of companies to search
        num_articles: Target number of articles per company
        fetch_full_content: If True, use Firecrawl to extract full article content (slower, costs more)
    """
    
    # Calculate date from one month ago
    one_month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    print(f"Fetching articles from {one_month_ago} onwards...")
    
    # Initialize Serper service
    serper_service = SerperService()
    
    # Randomly select companies (we'll try multiple companies to get enough articles)
    np.random.seed(42)
    selected_companies = np.random.choice(fortune_500_companies, size=num_searches, replace=False)
    
    all_articles = []
    seen_urls = set()  # Avoid duplicates
    
    print(f"Searching for {len(selected_companies)} randomly selected companies...")
    
    for i, company in enumerate(selected_companies, 1):
        current_article_ct = 0
        
        print(f"[{i}/{len(selected_companies)}] Fetching articles for {company}...")
        articles = fetch_articles_for_company(company, recency="m", max_articles=num_articles)
        
        # Filter out duplicates and add to collection
        for article in articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                article['_search_topic'] = company
                all_articles.append(article)
                current_article_ct += 1
                if current_article_ct >= num_articles:
                    break
               
        print(f"  Found {current_article_ct} new articles (total: {len(all_articles)})")
        time.sleep(0.5)  # Rate limiting for Serper 
    
    
    # Extract full content if requested
    firecrawl_service = None
    if fetch_full_content:
        print(f"\nExtracting full content for {len(all_articles)} articles using Firecrawl...")
        firecrawl_service = FirecrawlService()
        for i, article in enumerate(all_articles, 1):
            url = article.get('url', '')
            print(f"[{i}/{len(all_articles)}] Extracting content from {url[:80]}...")
            full_content = extract_full_content(url, firecrawl_service)
            # Store full content in article dict for later use
            article['_full_content'] = full_content
            time.sleep(0.2)  # Rate limiting for Firecrawl
    
    # Export to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting {len(all_articles)} articles to {output_path}...")
    
    # Format all articles
    formatted_articles = []
    for article in all_articles:
        full_content = article.get('_full_content') if fetch_full_content else None
        search_topic = article.get('_search_topic', '')
        formatted = format_article_for_json(article, full_content=full_content, search_topic=search_topic)
        formatted_articles.append(formatted)
    
    # Write JSON file with pretty formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_articles, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully exported {len(all_articles)} articles to {output_path}")


if __name__ == "__main__":
    generate_news_articles(num_searches=40, num_articles=5)

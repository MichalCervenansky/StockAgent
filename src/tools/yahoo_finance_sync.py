import requests
from bs4 import BeautifulSoup
import json
import os
import re
import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dateutil import parser


class ArticleDetails(BaseModel):
    """Model representing a news article's details."""
    title: str = Field(..., description="Title of the article")
    content: str = Field(..., description="Main content of the article")
    url: str = Field(..., description="URL of the article")  # Changed from HttpUrl to str
    timestamp: Optional[str] = Field(None, description="Publication timestamp of the article")
    
    @field_validator('content')
    @classmethod
    def clean_content(cls, v: str) -> str:
        # Clean content if needed
        return v
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )


class ScraperConfig(BaseModel):
    """Configuration model for the scraper."""
    stock_symbol: str = Field(..., description="Stock symbol to scrape news for")
    max_articles: int = Field(20, description="Maximum number of articles to scrape")
    max_age: int = Field(3, description="Maximum age of articles in days")
    user_agent: str = Field(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        description="User agent string for HTTP requests"
    )
    
    @field_validator('stock_symbol')
    @classmethod
    def validate_stock_symbol(cls, v: str) -> str:
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("Stock symbol cannot be empty")
        return v.strip().upper()


class YahooFinanceScraper:
    """Class for scraping news articles from Yahoo Finance."""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.headers = {"User-Agent": config.user_agent}
    
    def extract_article_details(self, url: str) -> Optional[ArticleDetails]:
        """Extract details from a single article page."""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Title extraction
            title_element = soup.find(class_=re.compile(r'cover-title yf-\w+'))
            title_text = title_element.get_text().strip() if title_element else "No title found"
            
            # Content extraction
            content_text = soup.find('div', class_='atoms-wrapper')
            paragraphs = []
            if content_text:
                for p in content_text.find_all('p'):
                    # Filter out empty paragraphs or paragraphs with specific content
                    text = p.get_text().strip()
                    if text and not text.startswith(('Read more:', 'Related:', 'Follow us on')):
                        # Clean text of special characters and extra spaces
                        text = re.sub(r'[\xa0\u200b]+', ' ', text)
                        text = re.sub(r'\s+', ' ', text)
                        paragraphs.append(text)
            
            content_text = '\n\n'.join(paragraphs)

            # Time extraction
            time_element = soup.find('time', class_='byline-attr-meta-time')
            article_time = time_element['datetime'] if time_element and time_element.has_attr('datetime') else None
            
            # Create and validate article model
            article = ArticleDetails(
                title=title_text,
                content=content_text,
                url=url,
                timestamp=article_time
            )
            
            return article
            
        except Exception as e:
            print(f"Error extracting details from {url}: {e}")
            return None

    def is_article_within_age_limit(self, timestamp: Optional[str]) -> bool:
        """Check if article is within the configured max_age in days."""
        if not timestamp:
            # If we don't know the timestamp, we'll include it to be safe
            return True
            
        try:
            article_date = parser.parse(timestamp)
            current_date = datetime.datetime.now(datetime.timezone.utc)
            age_days = (current_date - article_date).days
            
            return age_days <= self.config.max_age
        except Exception as e:
            print(f"Error parsing article date: {e}")
            # Include articles with unparseable dates to be safe
            return True

    def scrape_news(self) -> List[ArticleDetails]:
        """Scrape news articles for the configured stock symbol."""
        print(f"Starting to scrape Yahoo Finance news for {self.config.stock_symbol}")
        
        url = f"https://finance.yahoo.com/quote/{self.config.stock_symbol}/news/"
        
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract article URLs with deduplication
            links = soup.select('div[class*="holder yf-"] a[href^="https://finance.yahoo.com/"]')
            seen_urls = set()
            unique_article_urls = []
            
            for a in links:
                url = a['href']
                # Normalize URL by removing query parameters and fragments
                clean_url = url.split('?')[0].split('#')[0]
                if clean_url not in seen_urls:
                    seen_urls.add(clean_url)
                    unique_article_urls.append(url)
            
            articles = []
            processed_count = 0
            max_urls_to_check = min(self.config.max_articles * 2, len(unique_article_urls))  # Process more to account for filtering
            print(f"Found {len(unique_article_urls)} unique articles, checking up to {max_urls_to_check} for age filtering")
            
            for article_url in unique_article_urls[:max_urls_to_check]:
                if len(articles) >= self.config.max_articles:
                    break
                    
                article_details = self.extract_article_details(article_url)
                processed_count += 1
                
                if article_details:
                    # Check if article is within age limit
                    if self.is_article_within_age_limit(article_details.timestamp):
                        articles.append(article_details)
                    else:
                        print(f"Skipping article older than {self.config.max_age} days: {article_details.title}")
            
            print(f"Processed {processed_count} articles, found {len(articles)} within {self.config.max_age} day(s) age limit")
            return articles
            
        except Exception as e:
            print(f"Error scraping news for {self.config.stock_symbol}: {e}")
            return []
    
    def save_articles_to_json(self, articles: List[ArticleDetails]) -> str:
        """Save scraped articles to a JSON file."""
        if not articles:
            print("No articles to save")
            return ""
            
        # Prepare directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Create filename
        filename = f"{self.config.stock_symbol}_news_articles.json"
        filepath = os.path.join(data_dir, filename)
        
        # Convert articles to dict and save using model_dump instead of dict
        articles_dict = [article.model_dump() for article in articles]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles_dict, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(articles)} articles to {filepath}")
        return filepath


def scrape_yahoo_finance_news(stock_symbol: str, max_articles: int = 10) -> List[ArticleDetails]:
    """Main function to run the scraper."""
    # Create config with validation
    config = ScraperConfig(stock_symbol=stock_symbol, max_articles=max_articles)
    
    # Initialize and run scraper
    scraper = YahooFinanceScraper(config)
    articles = scraper.scrape_news()
    
    # Save results
    if articles:
        scraper.save_articles_to_json(articles)
    
    return articles


if __name__ == "__main__":
    print("Running the script directly")
    stock = "NVDA"
    print(f"Scraping news for stock: {stock}")
    articles = scrape_yahoo_finance_news(stock)
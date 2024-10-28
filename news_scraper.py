import requests
import pandas as pd
import re
import logging
import time
import os
import numpy as np
import json
import logging.handlers
from bs4 import BeautifulSoup, Tag
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass 
from urllib.parse import urljoin
from pathlib import Path

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(console_formatter)
    
    file_handler = logging.handlers.RotatingFileHandler(
        'scraper_debug.log', 
        maxBytes=1024*1024,  
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console)
    logger.addHandler(file_handler)

    # Silence noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return logger

class DateTimeHandler:    
    @staticmethod
    def standardize_timestamp(timestamp) -> str:
        if isinstance(timestamp, str):
            try:
                timestamp = pd.to_datetime(timestamp)
            except:
                timestamp = datetime.now()
        return timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')

class ContentCleaner:    
    @staticmethod
    def is_valid_title(title: str) -> bool:
        """Check if the title is valid."""
        if not title or not isinstance(title, str):
            return False
            
        # Remove titles that are likely author names or metadata
        invalid_patterns = [
            r'^by\s+',
            r'^\d+\s+(mins?|hours?|days?)\s+ago',
            r'^[A-Za-z]+\s+[A-Za-z]+\s*-\s*Reporter$',
            r'^CNN\s+',
            r'^\s*$',
            r'^Video:',
            r'^Photo:'
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, title, re.IGNORECASE):
                return False
        
        if len(title.strip()) < 10 or len(title.strip()) > 300:
            return False
            
        return True

class ArticleAnalytics:    
    def analyze_topics_by_age(self, df: pd.DataFrame) -> Dict:
        try:
            df = df.copy()
            df['age_category'] = pd.cut(
                df['age'],
                bins=[0, 1, 6, 24, float('inf')],
                labels=['last_hour', 'last_6_hours', 'last_day', 'older']
            )
            
            # Split the topics string into a list
            df['topic_list'] = df['topics'].fillna('').str.split(',')
            
            topics_by_age = {}
            for age_cat in df['age_category'].unique():
                age_df = df[df['age_category'] == age_cat]
                
                # Count topics in this age category
                topic_counts = {}
                for topics in age_df['topic_list']:
                    if isinstance(topics, list):
                        for topic in topics:
                            if topic:  # Skip empty topics
                                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                topics_by_age[str(age_cat)] = topic_counts
            
            return topics_by_age
            
        except Exception as e:
            logging.error(f"Error in analyze_topics_by_age: {str(e)}")
            return {}
    
    def analyze_freshness(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {
                'total_articles': 0,
                'age_distribution': {},
                'average_age': 0,
                'median_age': 0,
                'sources_distribution': {},
                'topics_by_age': {}
            }
        
        try:
            df = df.copy()
            now = pd.Timestamp.now(tz='UTC')
            df['age'] = (now - pd.to_datetime(df['timestamp'])).dt.total_seconds() / 3600
            
            analytics = {
                'total_articles': len(df),
                'age_distribution': {
                    'last_hour': len(df[df['age'] <= 1]),
                    'last_6_hours': len(df[df['age'] <= 6]),
                    'last_24_hours': len(df[df['age'] <= 24]),
                    'last_week': len(df[df['age'] <= 168])
                },
                'average_age': df['age'].mean(),
                'median_age': df['age'].median(),
                'sources_distribution': df['source'].value_counts().to_dict(),
                'topics_by_age': self.analyze_topics_by_age(df)
            }
            
            return analytics
            
        except Exception as e:
            logging.error(f"Error in analyze_freshness: {str(e)}")
            return {
                'error': str(e),
                'total_articles': len(df) if df is not None else 0
            }

class WebsiteStatus:    
    def __init__(self):
        self.status_log = {}
        self.logger = logging.getLogger(__name__)
    
    def update_status(self, source: str, success: bool, error_msg: str = None):
        """Update status for a news source."""
        self.status_log[source] = {
            'last_check': datetime.now(),
            'success': success,
            'error': error_msg
        }
    
    def get_source_status(self) -> Dict:
        return {
            source: {
                'status': 'active' if info['success'] else 'error',
                'last_successful': info['last_check'] if info['success'] else None,
                'error': info['error']
            }
            for source, info in self.status_log.items()
        }

@dataclass
class NewsArticle:
    title: str
    source: str
    url: Optional[str]
    timestamp: datetime
    category: Optional[str] = None
    raw_text: Optional[str] = None
    cleaned_text: Optional[str] = None
    topics: List[str] = None

class TextCleaner:    
    def __init__(self):
        # Custom list of words to remove
        self.custom_stopwords = {
            'says', 'said', 'reuters', 'nyt', 'times',
            'new', 'york', 'breaking', 'update', 'opinion',
            'published', 'hours', 'ago', 'mins', 'minutes'
        }
        
        # Common patterns to remove
        self.patterns = {
            'special_chars': r'[^A-Za-z0-9\s]',
            'extra_spaces': r'\s+',
            'numbers': r'\d+',
            'urls': r'http\S+|www\S+',
            'emails': r'\S+@\S+',
        }
        
        # Enhanced time patterns
        self.time_patterns = [
            r'^VIDEO \d+ (hours?|mins?|days?) ago\s*',
            r'^\d+ (hours?|mins?|days?) ago\s*',
            r'\d+ (hours?|mins?|days?) ago\s*$',
            r'opinion\s+\d+\s*[a-z]*\s*ago',
            r'published\s+\d+\s*[a-z]*\s*ago',
            r'\s*\d+\s*(hours?|minutes?|mins?|days?)\s*ago',
            r'\s*opinion\s+\d+\s*[a-z]*\s*ago'
        ]
        
        self.section_patterns = [
            r'^(ELECTIONS|Media|VIDEO|Entertainment|Donald Trump|Kamala Harris|Iran|China|Israel)\s+',
            r'^[A-Z]+:\s+',
            r'^\[[A-Z\s]+\]\s*'
        ]
    
    def clean_title(self, title: str) -> str:
        if not title:
            return ""
        
        cleaned_title = title
        
        # Remove time patterns
        for pattern in self.time_patterns:
            cleaned_title = re.sub(pattern, '', cleaned_title, flags=re.IGNORECASE)
        
        # Remove section patterns
        for pattern in self.section_patterns:
            cleaned_title = re.sub(pattern, '', cleaned_title)
        
        # Remove extra whitespace and standardize
        cleaned_title = ' '.join(cleaned_title.split())
        
        return cleaned_title.strip()
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove URLs
        text = re.sub(self.patterns['urls'], '', text)
        
        # Remove email addresses
        text = re.sub(self.patterns['emails'], '', text)
        
        # Remove special characters
        text = re.sub(self.patterns['special_chars'], ' ', text)
        
        # Remove extra whitespace
        text = re.sub(self.patterns['extra_spaces'], ' ', text)
        
        # Remove custom stopwords
        words = text.split()
        words = [w for w in words if w not in self.custom_stopwords]
        
        return ' '.join(words)
    
class TopicAnalyzer:    
    def __init__(self):
        self.topic_keywords = {
            'politics': {'election', 'campaign', 'democrat', 'republican', 'trump', 'biden', 'harris', 
                        'voter', 'poll', 'ballot', 'congress', 'senate', 'political'},
            'economy': {'economy', 'economic', 'market', 'stock', 'trade', 'inflation', 'fed', 
                       'interest rate', 'debt', 'gdp', 'jobs'},
            'foreign_policy': {'foreign', 'international', 'china', 'russia', 'diplomacy', 
                             'treaty', 'war', 'military', 'defense'},
        }
    
    def analyze_topic(self, text: str) -> List[str]:
        text = text.lower()
        topics = []
        
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        return topics or ['other']    

class DataStorage:
    def __init__(self):
        # Initialize paths
        self.project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        self.base_path = self.project_root / "data"
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        
        # Create directories if they don't exist
        for path in [self.base_path, self.raw_path, self.processed_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize handlers
        self.datetime_handler = DateTimeHandler()
        self.content_cleaner = ContentCleaner()
        self.cleaner = TextCleaner()
        self.article_analytics = ArticleAnalytics()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        # Remove the _setup_logging call and put the code directly here
        log_file = self.base_path / "scraper.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def save_raw_articles(self, df: pd.DataFrame, source: str):
        try:
            # Only keep the raw columns
            raw_columns = ['title', 'source', 'url', 'timestamp']
            raw_df = df[raw_columns].copy()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_filename = self.raw_path / f"{source}_{timestamp}_raw.csv"
            raw_df.to_csv(raw_filename, index=False)
            self.logger.info(f"Saved raw data to {raw_filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving raw articles: {str(e)}")

    def clean_and_standardize_df(self, df):
        try:
            if df.empty:
                return df
            df = df.copy()
            
            # Standardize timestamps
            df['timestamp'] = df['timestamp'].apply(self.datetime_handler.standardize_timestamp)
            
            # Clean titles - remove time references and standardize format
            df['title'] = df['title'].apply(lambda x: self.cleaner.clean_title(x))
            
            # Additional title cleaning for time references
            time_patterns = [
                r'\d+\s*hours?\s*ago',
                r'\d+\s*mins?\s*ago',
                r'\d+\s*days?\s*ago',
                r'opinion\s+\d+\s*[a-z]*\s*ago',
                r'published\s+\d+\s*[a-z]*\s*ago'
            ]
            
            for pattern in time_patterns:
                df['title'] = df['title'].str.replace(pattern, '', case=False, regex=True)
            
            # Filter valid titles
            mask = df['title'].apply(ContentCleaner.is_valid_title)
            df = df[mask].copy()
            
            # Remove duplicates
            df['title_lower'] = df['title'].str.lower()
            df = df.drop_duplicates(subset=['title_lower', 'source']).copy()
            df = df.drop(columns=['title_lower'])
            
            # Add cleaned text and topics
            df['cleaned_text'] = df['title'].apply(self.cleaner.clean_text)
            df['topics'] = df['title'].apply(lambda x: ','.join(TopicAnalyzer().analyze_topic(x)))
            
            # Sort by timestamp
            df = df.sort_values('timestamp', ascending=False).copy()
            
            # Ensure required columns with consistent formatting
            required_columns = [
                'title', 'source', 'url', 'timestamp', 'cleaned_text', 'topics'
            ]
        
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
                    
            df = df[required_columns]
            
            return df
                     
        except Exception as e:
            self.logger.error(f"Error in clean_and_standardize_df: {str(e)}")
            return pd.DataFrame()        
        

    def save_articles(self, df, source):
        try:
            if df.empty:
                self.logger.warning(f"No articles to save for {source}")
                return
            
            # Save raw data first
            self.save_raw_articles(df, source)
            
            # Process and save cleaned data with sentiment
            processed_df = self.clean_and_standardize_df(df)
            
            if processed_df.empty:
                self.logger.warning(f"No valid articles remaining after cleaning for {source}")
                return
            
            # Use consistent naming format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_filename = self.processed_path / f"processed_{source}_{timestamp}.csv"
            processed_df.to_csv(processed_filename, index=False)
            self.logger.info(f"Saved processed data to {processed_filename}")
            
            # Update master file with new naming convention
            self._update_master_file(processed_df.copy())
            
            # Save daily summary with new naming convention - fixed daily_filename parameter
            daily_filename = self.processed_path / f"daily_summary_{datetime.now().strftime('%Y%m%d')}.csv"
            self._save_daily_summary(processed_df.copy(), daily_filename)
            
        except Exception as e:
            self.logger.error(f"Error in save_articles: {str(e)}")

    def _update_master_file(self, new_df: pd.DataFrame):
        """Update the master file with new articles."""
        try:
            master_file = self.processed_path / "all_articles.csv"
            
            if master_file.exists():
                master_df = pd.read_csv(master_file)
                master_df['timestamp'] = master_df['timestamp'].apply(
                    self.datetime_handler.standardize_timestamp
                )
                combined_df = pd.concat([master_df, new_df])
                combined_df = self.clean_and_standardize_df(combined_df)
                combined_df.to_csv(master_file, index=False)
                self.logger.info(f"Added {len(new_df)} new articles to master file")
            else:
                new_df.to_csv(master_file, index=False)
                self.logger.info(f"Created master file with {len(new_df)} articles")
        except Exception as e:
            self.logger.error(f"Error updating master file: {str(e)}")

    def _save_daily_summary(self, df: pd.DataFrame, daily_filename: Path):
        try:
            if daily_filename.exists():
                existing_df = pd.read_csv(daily_filename)
                combined_df = pd.concat([existing_df, df])
                combined_df = combined_df.drop_duplicates(subset=['title', 'source'])
                combined_df = combined_df.sort_values('timestamp', ascending=False)
                combined_df.to_csv(daily_filename, index=False)
            else:
                df.to_csv(daily_filename, index=False)
            self.logger.info(f"Updated daily summary at {daily_filename}")
        except Exception as e:
            self.logger.error(f"Error saving daily summary: {str(e)}")
            
    def get_analytics(self) -> Dict:
        """Get analytics for all articles."""
        try:
            master_file = self.processed_path / "all_articles.csv"
            if not master_file.exists():
                return {}
            
            df = pd.read_csv(master_file)
            return self.article_analytics.analyze_freshness(df)
        except Exception as e:
            self.logger.error(f"Error generating analytics: {str(e)}")
            return {}
        
        

class NewsScraperBase:  
    def __init__(self, rate_limit=0.5):
        self.cleaner = TextCleaner()
        self.content_cleaner = ContentCleaner()  # Add this!
        self.rate_limit = rate_limit
        self.last_request = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.website_status = WebsiteStatus()
        
        # Common headers for all requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        # Configure session for speed
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Very short timeout - fail fast and move on
        self.timeout = (2, 3)  # 2 seconds connect, 3 seconds read
        
        # Configure minimal retries to avoid waiting
        retry_strategy = requests.adapters.Retry(
            total=1,  # only retry once
            backoff_factor=0.1,  # very short backoff
            status_forcelist=[500, 502, 503, 504],  # only retry on server errors
            allowed_methods=["GET"]
        )
        
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _rate_limit(self):
        """Ensure minimum time between requests."""
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()
    
    def _make_request(self, url: str) -> Optional[str]:
        """Make HTTP request with fast timeouts - fail quickly if site is slow."""
        try:
            self._rate_limit()
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            self.website_status.update_status(self.__class__.__name__, True)
            return response.text
        except Exception as e:
            # Don't waste time with detailed error handling - just log and move on
            self.logger.debug(f"Failed to fetch {url}: {str(e)}")
            self.website_status.update_status(self.__class__.__name__, False, str(e))
            return None

    def create_article(self, raw_title: str, source: str, url: str, timestamp: datetime) -> NewsArticle:
        """Create a NewsArticle instance with cleaned text."""
        cleaned_title = self.cleaner.clean_title(raw_title)
        cleaned_text = self.cleaner.clean_text(cleaned_title)
        topics = TopicAnalyzer().analyze_topic(cleaned_title)
        
        return NewsArticle(
            title=cleaned_title,
            source=source,
            url=url,
            timestamp=timestamp,
            raw_text=raw_title,
            cleaned_text=cleaned_text,
            topics=topics
        )

class NYTScraper(NewsScraperBase):
    """Scraper for New York Times headlines."""
    
    def __init__(self):
        super().__init__(rate_limit=2.0)
        self.base_url = "https://www.nytimes.com"
        self.sections = [
            'section/politics',
            'section/us/politics', 
            'section/us',
            'news/politics',
            'section/opinion/politics',
            'section/upshot'
        ]
    
    def get_headlines(self) -> List[NewsArticle]:
        """Scrape latest headlines from NYT."""
        articles = []
        seen_urls = set()
        
        for section in self.sections:
            try:
                url = f"{self.base_url}/{section}"
                html_content = self._make_request(url)
                if not html_content:
                    continue
                    
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Updated selectors for NYT's structure
                selectors = [
                    {'class': ['css-1l4spti', 'css-1cp3ece', 'css-1ez5fsm']},
                    {'class': 'css-ye6x8s'},
                    {'data-testid': 'block-content'},
                    {'data-testid': 'standard-thumbnail'}
                ]
                
                for selector in selectors:
                    for article in soup.find_all(['article', 'div', 'li'], selector):
                        try:
                            headline_elem = (
                                article.find(['h2', 'h3', 'h4'], {'class': 'css-1j9dxys'}) or
                                article.find(['h2', 'h3', 'h4']) or
                                article.find('a', {'class': 'css-9mylee'})
                            )
                            
                            if not headline_elem:
                                continue
                                
                            raw_title = headline_elem.get_text().strip()
                            if not ContentCleaner.is_valid_title(raw_title):
                                continue
                                
                            url_elem = article.find('a', href=True)
                            if not url_elem:
                                continue
                                
                            article_url = url_elem['href']
                            if not article_url.startswith('http'):
                                article_url = urljoin(self.base_url, article_url)
                                
                            if article_url in seen_urls:
                                continue
                            seen_urls.add(article_url)
                            
                            article = self.create_article(
                                raw_title=raw_title,
                                source="New York Times",
                                url=article_url,
                                timestamp=datetime.now()
                            )
                            articles.append(article)
                            
                        except Exception as e:
                            self.logger.error(f"Error parsing NYT article: {str(e)}")
                            continue
                            
            except Exception as e:
                self.logger.error(f"Error processing NYT section {section}: {str(e)}")
                continue
                
        return articles

class NYTScraper(NewsScraperBase):
    """Scraper for New York Times headlines."""
    
    def __init__(self):
        super().__init__(rate_limit=2.0)
        self.base_url = "https://www.nytimes.com"
        self.sections = [
            'section/politics',
            'section/us/politics', 
            'section/us',
            'news/politics',
            'section/opinion/politics',
            'section/upshot'
        ]
    
    def get_headlines(self) -> List[NewsArticle]:
        """Scrape latest headlines from NYT."""
        articles = []
        seen_urls = set()
        
        for section in self.sections:
            try:
                url = f"{self.base_url}/{section}"
                html_content = self._make_request(url)
                if not html_content:
                    continue
                    
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Updated selectors for NYT's structure
                selectors = [
                    {'class': ['css-1l4spti', 'css-1cp3ece', 'css-1ez5fsm']},
                    {'class': 'css-ye6x8s'},
                    {'data-testid': 'block-content'},
                    {'data-testid': 'standard-thumbnail'}
                ]
                
                for selector in selectors:
                    for article in soup.find_all(['article', 'div', 'li'], selector):
                        try:
                            headline_elem = (
                                article.find(['h2', 'h3', 'h4'], {'class': 'css-1j9dxys'}) or
                                article.find(['h2', 'h3', 'h4']) or
                                article.find('a', {'class': 'css-9mylee'})
                            )
                            
                            if not headline_elem:
                                continue
                                
                            raw_title = headline_elem.get_text().strip()
                            if not ContentCleaner.is_valid_title(raw_title):
                                continue
                                
                            url_elem = article.find('a', href=True)
                            if not url_elem:
                                continue
                                
                            article_url = url_elem['href']
                            if not article_url.startswith('http'):
                                article_url = urljoin(self.base_url, article_url)
                                
                            if article_url in seen_urls:
                                continue
                            seen_urls.add(article_url)
                            
                            article = self.create_article(
                                raw_title=raw_title,
                                source="New York Times",
                                url=article_url,
                                timestamp=datetime.now()
                            )
                            articles.append(article)
                            
                        except Exception as e:
                            self.logger.error(f"Error parsing NYT article: {str(e)}")
                            continue
                            
            except Exception as e:
                self.logger.error(f"Error processing NYT section {section}: {str(e)}")
                continue
                
        return articles

class WaPoScraper(NewsScraperBase):
    """Scraper for Washington Post headlines."""
    
    def __init__(self):
        super().__init__(rate_limit=3.0)  # Increased rate limit for WaPo
        self.base_url = "https://www.washingtonpost.com"
        self.sections = [
            'politics',
            'elections',
            'congress',
            'white-house',
            'powerpost',
            'polling',
            'campaign',
            'the-trailer'
        ]
        
        # Set up session with longer timeout for initial connection
        self.timeout = (5, 10)  # (connect timeout, read timeout)
        
        # Configure session with keep-alive and connection pooling
        self.session.headers.update({
            'Connection': 'keep-alive',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Configure retries with longer backoff
        retry_strategy = requests.adapters.Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        # Use larger connection pool
        adapter = requests.adapters.HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,
            pool_maxsize=20
        )
        self.session.mount("https://", adapter)
        
    def get_headlines(self) -> List[NewsArticle]:
        """Scrape latest headlines from WaPo."""
        articles = []
        seen_urls = set()
        
        # Try the base politics URL first
        main_url = f"{self.base_url}/politics"
        html_content = self._make_request(main_url)
        if html_content:
            # If main page works, process other sections
            for section in self.sections:
                try:
                    if section != 'politics':  # Skip politics since we already got it
                        url = f"{self.base_url}/{section}"
                        section_content = self._make_request(url)
                        if section_content:
                            articles.extend(self._parse_page_content(section_content, seen_urls))
                    
                except Exception as e:
                    self.logger.error(f"Error processing WaPo section {section}: {str(e)}")
                    continue
                    
            # Process the main politics page content last
            articles.extend(self._parse_page_content(html_content, seen_urls))
            
        return articles
        
    def _parse_page_content(self, html_content: str, seen_urls: set) -> List[NewsArticle]:
        """Parse WaPo page content for articles."""
        articles = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Updated selectors for WaPo's structure
        selectors = [
            'div[data-feature-id="homepage/story"]',
            'div.story-headline',
            'div[data-qa="headline-text"]',
            'h2.headline',
            'h3.headline'
        ]
        
        for selector in selectors:
            for article in soup.select(selector):
                try:
                    headline_elem = (
                        article.select_one('a[data-qa="headline-text"]') or
                        article.select_one('a.headline-text') or
                        article.select_one('a')
                    )
                    
                    if not headline_elem:
                        continue
                        
                    raw_title = headline_elem.get_text().strip()
                    if not self.content_cleaner.is_valid_title(raw_title):
                        continue
                        
                    article_url = headline_elem.get('href', '')
                    if not article_url:
                        continue
                        
                    if not article_url.startswith('http'):
                        article_url = urljoin(self.base_url, article_url)
                        
                    if article_url in seen_urls:
                        continue
                    seen_urls.add(article_url)
                    
                    article = self.create_article(
                        raw_title=raw_title,
                        source="Washington Post",
                        url=article_url,
                        timestamp=datetime.now()
                    )
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.error(f"Error parsing WaPo article: {str(e)}")
                    continue
                    
        return articles

class CNNPoliticsScraper(NewsScraperBase):
    """Improved CNN scraper with updated selectors."""
    
    def __init__(self):
        super().__init__(rate_limit=2.0)
        self.base_url = "https://www.cnn.com"
        
    def get_headlines(self) -> List[NewsArticle]:
        articles = []
        seen_urls = set()
        
        # Updated CNN selectors based on current structure
        content_selectors = [
            {'class': 'card container__item container__item--type-section container__item--type-politics'},
            {'data-contenttype': 'article'},
            {'class': 'container__headline-text'},
            {'class': 'container__item-container'}
        ]
        
        urls = [
            f"{self.base_url}/politics",
            f"{self.base_url}/politics/2024-election"
        ]
        
        for url in urls:
            try:
                html_content = self._make_request(url)  # Fixed: Use _make_request instead of session.get
                if not html_content:
                    continue
                    
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # First try to find the articles container
                articles_container = soup.find('div', {'class': 'container__items-wrapper'})
                if not articles_container:
                    articles_container = soup  # Fallback to full page if no container found

                for selector in content_selectors:
                    for article in articles_container.find_all(['div', 'article'], selector):
                        try:
                            # Updated headline finding logic
                            headline = (
                                article.find('span', {'class': 'container__headline-text'}) or
                                article.find('h3', {'class': 'container__headline-text'}) or
                                article.find('a', {'class': 'container__link'})
                            )
                            
                            if not headline:
                                continue
                                
                            title = headline.get_text().strip()
                            if not self.content_cleaner.is_valid_title(title):
                                continue
                                
                            # Updated link finding logic
                            link = article.find('a', {'class': 'container__link', 'href': True})
                            if not link:
                                link = headline.find_parent('a', href=True)
                            if not link:
                                continue
                                
                            article_url = link['href']
                            if not article_url.startswith('http'):
                                article_url = urljoin(self.base_url, article_url)
                                
                            if article_url in seen_urls:
                                continue
                            seen_urls.add(article_url)
                            
                            article = self.create_article(
                                raw_title=title,
                                source="CNN",
                                url=article_url,
                                timestamp=datetime.now()
                            )
                            articles.append(article)
                            
                        except Exception as e:
                            self.logger.debug(f"Error parsing CNN article: {str(e)}")
                            continue
                            
            except Exception as e:
                self.logger.debug(f"Error processing CNN URL {url}: {str(e)}")
                continue
                
        return articles
    
class FoxNewsPoliticsScraper(NewsScraperBase):
    """Scraper for Fox News Politics headlines."""
    
    def __init__(self):
        super().__init__(rate_limit=2.0)
        self.base_url = "https://www.foxnews.com"
        self.sections = [
            'politics',
            'politics/elections',
            'politics/senate',
            'politics/house-of-representatives',
            'politics/executive',
            'politics/state-and-local',
            'person/donald-trump',
            'person/joe-biden',
            'person/kamala-harris'
        ]
    
    def get_headlines(self) -> List[NewsArticle]:
        """Scrape latest headlines from Fox News Politics."""
        articles = []
        seen_urls = set()
        
        for section in self.sections:
            try:
                url = f"{self.base_url}/{section}"
                html_content = self._make_request(url)
                if not html_content:
                    continue
                    
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Updated selectors for Fox News structure
                selectors = [
                    {'class': ['article', 'story']},
                    {'class': 'content'},
                    {'class': 'collection-article'},
                    {'data-type': 'article'}
                ]
                
                for selector in selectors:
                    for article in soup.find_all(['article', 'div'], selector):
                        try:
                            headline_elem = (
                                article.find(['h2', 'h3', 'h4'], {'class': 'title'}) or
                                article.find(['h2', 'h3', 'h4'], {'class': 'headline'}) or
                                article.find(['h2', 'h3', 'h4'])
                            )
                            
                            if not headline_elem:
                                continue
                                
                            raw_title = headline_elem.get_text().strip()
                            if not ContentCleaner.is_valid_title(raw_title):
                                continue
                                
                            url_elem = article.find('a', href=True)
                            if not url_elem:
                                continue
                                
                            article_url = url_elem['href']
                            if not article_url.startswith('http'):
                                article_url = urljoin(self.base_url, article_url)
                                
                            if article_url in seen_urls:
                                continue
                            seen_urls.add(article_url)
                            
                            # Try to get time info
                            time_elem = article.find('time') or article.find('span', {'class': 'time'})
                            timestamp = datetime.now()
                            if time_elem and time_elem.get('datetime'):
                                try:
                                    timestamp = pd.to_datetime(time_elem['datetime'])
                                except:
                                    pass
                            
                            article = self.create_article(
                                raw_title=raw_title,
                                source="Fox News Politics",
                                url=article_url,
                                timestamp=timestamp
                            )
                            articles.append(article)
                            
                        except Exception as e:
                            self.logger.error(f"Error parsing Fox News article: {str(e)}")
                            continue
                            
            except Exception as e:
                self.logger.error(f"Error processing Fox News section {section}: {str(e)}")
                continue
                
        return articles

class RedditPoliticsScraper(NewsScraperBase):
    """Scraper for Reddit r/politics headlines."""
    
    def __init__(self):
        super().__init__(rate_limit=2.0)
        self.subreddit = "politics"
        self.base_url = "https://www.reddit.com/r/politics"
        # Use old.reddit.com as it has simpler HTML structure
        self.api_url = "https://old.reddit.com/r/politics.json"
        # Instead of updating session headers, just set the headers directly
        self.headers = {
            'User-Agent': 'NewsScraperBot/1.0 (Research Project)'
        }
    
    def get_headlines(self) -> List[NewsArticle]:
        """Scrape latest headlines from r/politics."""
        articles = []
        seen_urls = set()
        
        try:
            # Get JSON data from Reddit API using the _make_request method with our custom headers
            response = self._make_request(self.api_url)
            if not response:
                return articles
                
            data = json.loads(response)
            posts = data.get('data', {}).get('children', [])
            
            for post in posts:
                try:
                    post_data = post.get('data', {})
                    
                    # Skip non-political or low quality posts
                    if post_data.get('removed_by_category') or \
                       post_data.get('removed') or \
                       post_data.get('score', 0) < 100:  # Minimum score threshold
                        continue
                    
                    title = post_data.get('title')
                    if not title or not ContentCleaner.is_valid_title(title):
                        continue
                        
                    url = post_data.get('url')
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    
                    # Convert Reddit timestamp
                    timestamp = datetime.fromtimestamp(post_data.get('created_utc', time.time()))
                    
                    # Get source domain
                    domain = post_data.get('domain', 'reddit.com')
                    if domain == 'self.politics':
                        domain = 'Reddit r/politics'
                    
                    article = self.create_article(
                        raw_title=title,
                        source=domain,
                        url=url,
                        timestamp=timestamp
                    )
                    
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.error(f"Error parsing Reddit post: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error fetching Reddit data: {str(e)}")
            
        return articles

def main():
    logger = setup_logging()
    storage = DataStorage()
    
    scrapers = {
        'nyt': NYTScraper(),
        'wapo': WaPoScraper(),
        'cnn': CNNPoliticsScraper(),
        'foxnews': FoxNewsPoliticsScraper(),
        'reddit': RedditPoliticsScraper()
    }
    
    results = {}
    
    for name, scraper in scrapers.items():
        try:
            logger.info(f"Fetching from {name}...")
            articles = scraper.get_headlines()
            
            if articles:
                df = pd.DataFrame([{
                    'title': a.title,
                    'source': a.source,
                    'url': a.url,
                    'timestamp': a.timestamp,
                    'cleaned_text': a.cleaned_text,
                    'topics': ','.join(TopicAnalyzer().analyze_topic(a.title))
                } for a in articles])
                
                storage.save_articles(df, name)
                results[name] = len(articles)
                logger.info(f"Collected {len(articles)} articles from {name}")
            else:
                logger.warning(f"No articles collected from {name}")
                
        except Exception as e:
            logger.error(f"Error processing {name}: {str(e)}")
            continue
            
    logger.info("\nCollection Summary:")
    for source, count in results.items():
        logger.info(f"{source}: {count} articles")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
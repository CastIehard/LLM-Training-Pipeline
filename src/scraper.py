"""
Web Scraper Module
Handles web scraping with keyword-based search, URL deduplication, and HTML caching.
"""
import os
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote_plus
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WebScraper:
    """Web scraper with keyword search and URL deduplication"""
    
    def __init__(self, config):
        """
        Initialize the web scraper.
        
        Args:
            config (dict): Configuration dictionary with scraping settings
        """
        self.config = config
        self.scraped_urls = set()  # Track scraped URLs to avoid duplicates
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config['scraping'].get('user_agent', 'Mozilla/5.0')
        })
    
    def is_english_site(self, url, html_content):
        """
        Check if a website is likely in English.
        
        Args:
            url (str): The URL of the site
            html_content (str): HTML content of the page
            
        Returns:
            bool: True if likely English, False otherwise
        """
        if not self.config['scraping'].get('english_only', False):
            return True
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Check html lang attribute
            html_tag = soup.find('html')
            if html_tag and html_tag.get('lang'):
                lang = html_tag.get('lang', '').lower()
                if lang.startswith('en'):
                    return True
                elif lang and not lang.startswith('en'):
                    return False
            
            # Check meta language tags
            meta_lang = soup.find('meta', attrs={'http-equiv': 'content-language'})
            if meta_lang:
                content = meta_lang.get('content', '').lower()
                if content.startswith('en'):
                    return True
                elif content:
                    return False
            
            # If no explicit language marker, assume it could be English
            return True
            
        except Exception as e:
            logger.warning(f"Error checking language for {url}: {e}")
            return True
    
    def search_keyword(self, keyword):
        """
        Search for URLs using a keyword via DuckDuckGo search.
        
        Args:
            keyword (str): Keyword to search for
            
        Returns:
            list: List of URLs found for the keyword
        """
        urls = []
        max_urls = self.config['scraping'].get('max_urls_per_keyword', 5)
        
        try:
            # Use DuckDuckGo HTML version for simpler scraping
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(keyword)}"
            
            logger.info(f"Searching for keyword: {keyword}")
            response = self.session.get(
                search_url,
                timeout=self.config['scraping'].get('timeout', 10)
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract URLs from search results
            for result in soup.find_all('a', class_='result__url', limit=max_urls * 2):
                href = result.get('href', '')
                if href and href.startswith('http'):
                    # Clean URL
                    parsed = urlparse(href)
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    
                    # Check for duplicates
                    if clean_url not in self.scraped_urls and len(urls) < max_urls:
                        urls.append(clean_url)
            
            logger.info(f"Found {len(urls)} URLs for keyword: {keyword}")
            
        except Exception as e:
            logger.error(f"Error searching for keyword '{keyword}': {e}")
        
        return urls
    
    def scrape_url(self, url, cache_dir):
        """
        Scrape a single URL and cache the HTML.
        
        Args:
            url (str): URL to scrape
            cache_dir (str): Directory to cache HTML files
            
        Returns:
            tuple: (html_content, cache_file_path) or (None, None) on error
        """
        if url in self.scraped_urls:
            logger.info(f"URL already scraped, skipping: {url}")
            return None, None
        
        try:
            logger.info(f"Scraping URL: {url}")
            response = self.session.get(
                url,
                timeout=self.config['scraping'].get('timeout', 10),
                allow_redirects=True
            )
            response.raise_for_status()
            
            html_content = response.text
            
            # Check if English only
            if not self.is_english_site(url, html_content):
                logger.info(f"Non-English site, skipping: {url}")
                return None, None
            
            # Generate cache filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{url_hash}.html")
            
            # Cache HTML
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Mark URL as scraped
            self.scraped_urls.add(url)
            
            logger.info(f"Successfully scraped and cached: {url}")
            
            # Be polite - delay between requests
            delay = self.config['scraping'].get('delay', 1)
            if delay > 0:
                time.sleep(delay)
            
            return html_content, cache_file
            
        except Exception as e:
            logger.error(f"Error scraping URL '{url}': {e}")
            return None, None
    
    def scrape_keywords(self, keywords, cache_dir):
        """
        Scrape URLs for multiple keywords.
        
        Args:
            keywords (list): List of keywords to search
            cache_dir (str): Directory to cache HTML files
            
        Returns:
            list: List of tuples (url, keyword, html_content, cache_file)
        """
        results = []
        
        for keyword in keywords:
            urls = self.search_keyword(keyword)
            
            for url in urls:
                html_content, cache_file = self.scrape_url(url, cache_dir)
                if html_content:
                    results.append((url, keyword, html_content, cache_file))
        
        logger.info(f"Total pages scraped: {len(results)}")
        return results

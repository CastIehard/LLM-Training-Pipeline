"""
Keyword Search Spider

Spider that searches DuckDuckGo for keywords and scrapes the resulting pages.
Uses Scrapy's built-in deduplication via dont_filter and custom URL tracking.
"""

from urllib.parse import quote_plus, urlparse

import scrapy

from web_scraper.items import WebPageItem


class KeywordSpider(scrapy.Spider):
    """
    Spider that performs keyword-based web scraping.

    Process:
    1. For each keyword, search DuckDuckGo
    2. Extract URLs from search results
    3. Scrape each URL and yield WebPageItem
    """

    name = "keyword_spider"
    allowed_domains = []  # Allow all domains

    def __init__(self, keywords=None, max_urls_per_keyword=5, *args, **kwargs):
        """
        Initialize the spider.

        Args:
            keywords: Comma-separated list of keywords or list of keywords
            max_urls_per_keyword: Maximum URLs to scrape per keyword
        """
        super().__init__(*args, **kwargs)

        # Parse keywords
        if isinstance(keywords, str):
            self.keywords = [k.strip() for k in keywords.split(",")]
        elif isinstance(keywords, list):
            self.keywords = keywords
        else:
            self.keywords = []

        self.max_urls_per_keyword = int(max_urls_per_keyword)
        self.scraped_urls = set()
        self.urls_per_keyword = {}

    def start_requests(self):
        """Generate initial requests to DuckDuckGo for each keyword."""
        for keyword in self.keywords:
            self.urls_per_keyword[keyword] = 0
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(keyword)}"
            self.logger.info(f"Searching for keyword: {keyword}")
            yield scrapy.Request(
                url=search_url,
                callback=self.parse_search_results,
                meta={"keyword": keyword},
                dont_filter=True,
            )

    def parse_search_results(self, response):
        """
        Parse DuckDuckGo search results and yield requests for found URLs.

        Args:
            response: Scrapy Response object from DuckDuckGo search
        """
        keyword = response.meta["keyword"]
        urls_found = 0

        # Extract URLs from search results using CSS selectors
        for result in response.css("a.result__url"):
            if urls_found >= self.max_urls_per_keyword:
                break

            href = result.attrib.get("href", "")
            if not href or not href.startswith("http"):
                continue

            # Clean URL
            parsed = urlparse(href)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            # Skip duplicates
            if clean_url in self.scraped_urls:
                continue

            # Check per-keyword limit
            if self.urls_per_keyword.get(keyword, 0) >= self.max_urls_per_keyword:
                break

            self.scraped_urls.add(clean_url)
            self.urls_per_keyword[keyword] = self.urls_per_keyword.get(keyword, 0) + 1
            urls_found += 1

            self.logger.info(f"Found URL for '{keyword}': {clean_url}")
            yield scrapy.Request(
                url=clean_url,
                callback=self.parse_page,
                meta={"keyword": keyword},
                errback=self.handle_error,
            )

        self.logger.info(f"Found {urls_found} URLs for keyword: {keyword}")

    def parse_page(self, response):
        """
        Parse a scraped page and yield a WebPageItem.

        Args:
            response: Scrapy Response object from scraped page
        """
        keyword = response.meta["keyword"]

        self.logger.info(f"Scraped: {response.url}")

        item = WebPageItem()
        item["url"] = response.url
        item["keyword"] = keyword
        item["html_content"] = response.text

        yield item

    def handle_error(self, failure):
        """Handle request errors."""
        self.logger.error(f"Request failed: {failure.request.url} - {failure.value}")

"""
URL Spider with Depth-based Crawling

Spider that scrapes URLs from a file and optionally follows links to a configurable depth.
"""

import json
import os
from urllib.parse import urljoin, urlparse

import scrapy

from web_scraper.items import WebPageItem


class UrlSpider(scrapy.Spider):
    """
    Spider that scrapes URLs with configurable depth crawling.

    Process:
    1. Accept a list of seed URLs (from url.txt)
    2. Check if URL was already scraped (via index.json)
    3. Scrape each URL and yield WebPageItem
    4. Extract links and follow them up to max_depth
    """

    name = "url_spider"
    allowed_domains = []  # Allow all domains

    def __init__(
        self,
        urls=None,
        max_depth=1,
        index_file=None,
        url_blacklist=None,
        *args,
        **kwargs,
    ):
        """
        Initialize the spider.

        Args:
            urls: List of seed URLs to scrape
            max_depth: Maximum crawl depth (0 = only seed URLs, 1 = +links from seeds, etc.)
            index_file: Path to index.json for checking already scraped URLs
            url_blacklist: List of strings - URLs containing these will be skipped
        """
        super().__init__(*args, **kwargs)

        # Parse URLs
        if isinstance(urls, str):
            self.urls = [u.strip() for u in urls.split(",") if u.strip()]
        elif isinstance(urls, list):
            self.urls = urls
        else:
            self.urls = []

        self.max_depth = max_depth
        self.index_file = index_file
        self.url_blacklist = url_blacklist or []

        # Track URLs to avoid duplicates
        self.scraped_urls = set()
        self.queued_urls = set()

        # Track blacklist skips
        self.blacklist_skipped = 0

        # Load already scraped URLs from index
        self._load_scraped_urls()

    def _load_scraped_urls(self):
        """Load already scraped URLs from index.json."""
        if not self.index_file or not os.path.exists(self.index_file):
            self.logger.info("No existing index found, starting fresh")
            return

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)

            for entry in index_data:
                url = entry.get("source_url", "")
                if url:
                    # Normalize URL for consistent comparison
                    normalized = self._normalize_url(url)
                    self.scraped_urls.add(normalized)

            self.logger.info(
                f"Loaded {len(self.scraped_urls)} already scraped URLs from index"
            )
        except Exception as e:
            self.logger.warning(f"Error loading index file: {e}")

    def _normalize_url(self, url):
        """Normalize URL for comparison (remove trailing slash, fragments, www)."""
        parsed = urlparse(url)
        # Remove www. prefix for consistent comparison
        netloc = parsed.netloc
        if netloc.startswith("www."):
            netloc = netloc[4:]
        # Reconstruct without fragment
        path = parsed.path
        # Remove trailing slash for consistency
        if path.endswith("/") and len(path) > 1:
            path = path[:-1]
        elif path == "/":
            path = ""
        normalized = f"{parsed.scheme}://{netloc}{path}"
        return normalized

    def _is_valid_url(self, url, base_url):
        """Check if URL is valid for crawling."""
        if not url:
            return False

        # Skip non-http(s) URLs
        if not url.startswith(("http://", "https://")):
            return False

        # Skip mailto, tel, javascript links
        if url.startswith(("mailto:", "tel:", "javascript:")):
            return False

        # Check against URL blacklist (case-insensitive)
        url_lower = url.lower()
        for blacklisted in self.url_blacklist:
            if blacklisted.lower() in url_lower:
                self.blacklist_skipped += 1
                return False

        return True

    async def start(self):
        """Generate requests for seed URLs."""
        for url in self.urls:
            normalized = self._normalize_url(url)

            if normalized in self.scraped_urls:
                self.logger.info(f"Skipping already scraped URL: {url}")
                continue

            if normalized in self.queued_urls:
                continue

            self.queued_urls.add(normalized)
            self.logger.info(f"Scraping seed URL (depth 0): {url}")

            yield scrapy.Request(
                url=url,
                callback=self.parse_page,
                errback=self.handle_error,
                meta={"depth": 0},
            )

    def parse_page(self, response):
        """
        Parse a scraped page, yield item, and optionally follow links.

        Args:
            response: Scrapy Response object from scraped page
        """
        current_depth = response.meta.get("depth", 0)
        normalized_url = self._normalize_url(response.url)

        self.logger.info(f"Scraped (depth {current_depth}): {response.url}")

        # Mark as scraped
        self.scraped_urls.add(normalized_url)

        # Create and yield item
        item = WebPageItem()
        item["url"] = response.url
        item["keyword"] = ""
        item["html_content"] = response.text
        item["depth"] = current_depth

        yield item

        # Extract and follow links if within max_depth
        if current_depth < self.max_depth:
            yield from self._extract_and_follow_links(response, current_depth)

    def _extract_and_follow_links(self, response, current_depth):
        """Extract links from page and yield requests for them."""
        next_depth = current_depth + 1

        # Extract all links from the page
        links = response.css("a::attr(href)").getall()
        self.logger.info(f"Found {len(links)} links on {response.url}")

        # Collect valid links first
        valid_links = []
        for link in links:
            # Convert relative URLs to absolute
            absolute_url = urljoin(response.url, link)
            normalized = self._normalize_url(absolute_url)

            # Skip if not valid or already processed
            if not self._is_valid_url(absolute_url, response.url):
                continue

            if normalized in self.scraped_urls or normalized in self.queued_urls:
                continue

            valid_links.append((absolute_url, normalized))

        # Store count for progress tracking
        if not hasattr(self, "depth_totals"):
            self.depth_totals = {}
        self.depth_totals[next_depth] = self.depth_totals.get(next_depth, 0) + len(
            valid_links
        )

        self.logger.info(f"Following {len(valid_links)} new links from {response.url}")

        # Now yield requests for valid links
        for absolute_url, normalized in valid_links:
            self.queued_urls.add(normalized)

            self.logger.debug(f"Following link (depth {next_depth}): {absolute_url}")

            yield scrapy.Request(
                url=absolute_url,
                callback=self.parse_page,
                errback=self.handle_error,
                meta={"depth": next_depth},
            )

    def handle_error(self, failure):
        """Handle request errors."""
        self.logger.error(f"Request failed: {failure.request.url} - {failure.value}")

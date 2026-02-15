"""
Main Pipeline Script

Runs the Scrapy-based web scraper with depth-based crawling and full processing pipeline.
Loads configuration from config.yaml and URLs from url.txt.
"""

import json
import logging
import os
import shutil
import sys
from collections import defaultdict

import yaml
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings
from tqdm import tqdm

# Suppress all logging except critical errors
logging.getLogger("scrapy").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("web_scraper").setLevel(logging.ERROR)


class ProgressTracker:
    """Track scraping progress with tqdm bars."""

    def __init__(self, seed_count, max_depth):
        self.seed_count = seed_count
        self.max_depth = max_depth
        self.stats = defaultdict(int)
        self.skipped = 0
        self.errors = 0
        self.progress_bars = {}
        self.depth_totals = {}
        self.depth_0_complete = False
        self.pending_depth_1_count = 0  # Items scraped before progress bar created

        # Create progress bar for seed URLs (depth 0)
        self.progress_bars[0] = tqdm(
            total=seed_count,
            desc="Depth 0 (seeds)",
            position=0,
            leave=True,
            bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}%",
        )

    def item_scraped(self, item, spider):
        """Called when an item is scraped."""
        depth = item.get("depth", 0)
        self.stats[depth] += 1

        # Update depth 0 progress
        if depth == 0:
            self.progress_bars[0].update(1)
            # Check if depth 0 is complete
            if self.stats[0] >= self.seed_count:
                self.depth_0_complete = True
                # Now create depth 1 progress bar with correct total
                if hasattr(spider, "depth_totals") and 1 in spider.depth_totals:
                    total = spider.depth_totals[1]
                    if 1 not in self.progress_bars and self.max_depth >= 1:
                        self.progress_bars[1] = tqdm(
                            total=total,
                            desc=f"Depth 1 (links)",
                            position=1,
                            leave=True,
                            bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}%",
                        )
                        # Update with any pending items
                        if self.pending_depth_1_count > 0:
                            self.progress_bars[1].update(self.pending_depth_1_count)
                            self.pending_depth_1_count = 0
        elif depth == 1:
            if 1 in self.progress_bars:
                self.progress_bars[1].update(1)
            else:
                # Progress bar not created yet, track pending
                self.pending_depth_1_count += 1

    def item_dropped(self, item, spider, exception):
        """Called when an item is dropped (filtered)."""
        self.skipped += 1

    def spider_error(self, failure, response, spider):
        """Called on spider errors."""
        self.errors += 1

    def close(self):
        """Close all progress bars and print summary."""
        for bar in self.progress_bars.values():
            bar.close()

    def print_summary(self, index_skipped=0, blacklist_skipped=0):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("SCRAPING SUMMARY")
        print("=" * 60)

        total = 0
        for depth in sorted(self.stats.keys()):
            count = self.stats[depth]
            total += count
            label = "seed URLs" if depth == 0 else "discovered links"
            print(f"  Depth {depth} ({label}): {count} pages scraped")

        print("-" * 60)
        print(f"  Total scraped: {total} pages")
        if index_skipped > 0:
            print(f"  Skipped (already in index): {index_skipped} URLs")
        if blacklist_skipped > 0:
            print(f"  Skipped (blacklisted): {blacklist_skipped} URLs")
        if self.skipped > 0:
            print(f"  Dropped (filtered): {self.skipped} pages")
        if self.errors > 0:
            print(f"  Errors: {self.errors}")
        print("=" * 60)


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def load_urls(url_file):
    """
    Load URLs from text file (one URL per line).

    Args:
        url_file: Path to file containing URLs

    Returns:
        List of URLs
    """
    if not os.path.exists(url_file):
        print(f"Error: URL file not found: {url_file}")
        sys.exit(1)

    urls = []
    try:
        with open(url_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    urls.append(line)
        return urls
    except Exception as e:
        print(f"Error loading URL file: {e}")
        sys.exit(1)


def main():
    """Main pipeline execution using Scrapy."""
    # Load configuration
    config = load_config()

    # Extract settings from config
    url_file = config.get("url_file", "url.txt")
    max_depth = config["scraping"].get("max_depth", 1)
    english_only = config["scraping"].get("english_only", True)
    delay = config["scraping"].get("delay", 1)
    timeout = config["scraping"].get("timeout", 10)
    user_agent = config["scraping"].get("user_agent", "Mozilla/5.0")
    url_blacklist = config["scraping"].get("url_blacklist", [])
    concurrent_requests = config["scraping"].get("concurrent_requests", 8)
    concurrent_requests_per_domain = config["scraping"].get(
        "concurrent_requests_per_domain", 2
    )
    retry_times = config["scraping"].get("retry_times", 2)

    cache_dir = config["output"].get("cache_dir", "cache")
    output_dir = config["output"].get("output_dir", "output")
    index_file = config["output"].get("index_file", "index.json")

    # Clear cache at start so it can be inspected after run for debugging
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    remove_urls = config["cleaning"].get("remove_urls", True)
    remove_emails = config["cleaning"].get("remove_emails", True)
    normalize_whitespace = config["cleaning"].get("normalize_whitespace", True)

    # Load URLs from file
    urls = load_urls(url_file)

    if not urls:
        print("Error: No URLs found in URL file")
        sys.exit(1)

    # Index file in main directory
    index_file_path = index_file

    # Print startup info
    print("=" * 60)
    print("WEB SCRAPER")
    print("=" * 60)
    print(f"  Seed URLs: {len(urls)}")
    print(f"  Max depth: {max_depth}")
    print(f"  URL blacklist: {len(url_blacklist)} patterns")
    print(f"  Output: {output_dir}/")
    print("=" * 60 + "\n")

    # Build Scrapy settings from config.yaml
    settings = Settings()
    settings.setdict(
        {
            # Core Scrapy settings
            "BOT_NAME": "web_scraper",
            "SPIDER_MODULES": ["web_scraper.spiders"],
            "NEWSPIDER_MODULE": "web_scraper.spiders",
            "ROBOTSTXT_OBEY": False,
            "COOKIES_ENABLED": True,
            "TELNETCONSOLE_ENABLED": False,
            "DOWNLOAD_DELAY": 0,
            "AUTOTHROTTLE_ENABLED": False,
            "HTTPCACHE_ENABLED": False,
            "RETRY_ENABLED": True,
            "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
            "LOG_ENABLED": False,
            "LOG_LEVEL": "ERROR",
            "DEFAULT_REQUEST_HEADERS": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            # Middlewares and pipelines
            "DOWNLOADER_MIDDLEWARES": {
                "web_scraper.middlewares.SameDomainDelayMiddleware": 100,
            },
            "ITEM_PIPELINES": {
                "web_scraper.pipelines.LanguageFilterPipeline": 100,
                "web_scraper.pipelines.HtmlCachePipeline": 200,
                "web_scraper.pipelines.MarkdownConversionPipeline": 300,
                "web_scraper.pipelines.MarkdownCleaningPipeline": 400,
                "web_scraper.pipelines.HashGenerationPipeline": 500,
                "web_scraper.pipelines.OutputPipeline": 600,
            },
            # Settings from config.yaml
            "USER_AGENT": user_agent,
            "SAME_DOMAIN_DELAY": delay,
            "DOWNLOAD_TIMEOUT": timeout,
            "CONCURRENT_REQUESTS": concurrent_requests,
            "CONCURRENT_REQUESTS_PER_DOMAIN": concurrent_requests_per_domain,
            "RETRY_TIMES": retry_times,
            "ENGLISH_ONLY": english_only,
            "CACHE_DIR": cache_dir,
            "OUTPUT_DIR": output_dir,
            "INDEX_FILE": index_file,
            "CLEANING_REMOVE_URLS": remove_urls,
            "CLEANING_REMOVE_EMAILS": remove_emails,
            "CLEANING_NORMALIZE_WHITESPACE": normalize_whitespace,
        }
    )

    # Ensure directories exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create progress tracker
    tracker = ProgressTracker(seed_count=len(urls), max_depth=max_depth)

    # Create crawler
    process = CrawlerProcess(settings)

    # Get the crawler and connect signals
    crawler = process.create_crawler("url_spider")
    crawler.signals.connect(tracker.item_scraped, signal=signals.item_scraped)
    crawler.signals.connect(tracker.item_dropped, signal=signals.item_dropped)
    crawler.signals.connect(tracker.spider_error, signal=signals.spider_error)

    # Start crawling
    process.crawl(
        crawler,
        urls=urls,
        max_depth=max_depth,
        index_file=index_file_path,
        url_blacklist=url_blacklist,
    )

    # Run the crawler (blocks until done)
    process.start()

    # Close progress bars
    tracker.close()

    # Get blacklist skip count from spider
    blacklist_skipped = getattr(crawler.spider, "blacklist_skipped", 0)

    # Print summary
    tracker.print_summary(
        index_skipped=max(0, len(urls) - tracker.stats.get(0, 0)),
        blacklist_skipped=blacklist_skipped,
    )


if __name__ == "__main__":
    main()

"""
Scrapy Settings Module

Configuration for the web scraper Scrapy project.
All settings can be overridden when running the spider.
"""

BOT_NAME = "web_scraper"

SPIDER_MODULES = ["web_scraper.spiders"]
NEWSPIDER_MODULE = "web_scraper.spiders"

# Crawl responsibly by identifying yourself
USER_AGENT = "Mozilla/5.0 (compatible; UTN-Scraper/1.0)"

# Obey robots.txt rules
ROBOTSTXT_OBEY = False  # DuckDuckGo doesn't like automated access

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = 8

# Configure a delay for requests for the same website
DOWNLOAD_DELAY = 1

# The download delay setting will honor only one of:
CONCURRENT_REQUESTS_PER_DOMAIN = 2
CONCURRENT_REQUESTS_PER_IP = 2

# Disable cookies (enabled by default)
COOKIES_ENABLED = True

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Override the default request headers
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Configure item pipelines - ORDER MATTERS!
# Lower numbers = higher priority (runs first)
ITEM_PIPELINES = {
    "web_scraper.pipelines.LanguageFilterPipeline": 100,
    "web_scraper.pipelines.HtmlCachePipeline": 200,
    "web_scraper.pipelines.MarkdownConversionPipeline": 300,
    "web_scraper.pipelines.MarkdownCleaningPipeline": 400,
    "web_scraper.pipelines.HashGenerationPipeline": 500,
    "web_scraper.pipelines.OutputPipeline": 600,
}

# Enable and configure the AutoThrottle extension
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
HTTPCACHE_ENABLED = False

# Request timeout
DOWNLOAD_TIMEOUT = 15

# Retry configuration
RETRY_ENABLED = True
RETRY_TIMES = 2
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Log settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

# ============================================================================
# Custom Settings for Pipelines
# ============================================================================

# Language Filter Settings
ENGLISH_ONLY = True

# Cache Settings
CACHE_DIR = "cache"

# Output Settings
OUTPUT_DIR = "output"
INDEX_FILE = "index.json"

# Cleaning Settings
CLEANING_REMOVE_URLS = True
CLEANING_REMOVE_EMAILS = True
CLEANING_NORMALIZE_WHITESPACE = True

# ============================================================================
# Spider Settings (can be overridden via command line)
# ============================================================================

# Default keywords (override via -a keywords="keyword1,keyword2")
DEFAULT_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "natural language processing",
]

# Maximum URLs to scrape per keyword
MAX_URLS_PER_KEYWORD = 5

# Set settings whose default value is deprecated to a future-proof value
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"

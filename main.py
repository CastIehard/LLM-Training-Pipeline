"""
Main Pipeline Script

Runs the Scrapy-based web scraper with keyword search and full processing pipeline.
Loads configuration from config.yaml and runs the spider programmatically.
"""

import logging
import os
import sys

import yaml
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)


def main():
    """Main pipeline execution using Scrapy."""
    logger.info("=" * 80)
    logger.info("Starting Scrapy Web Scraper Pipeline")
    logger.info("=" * 80)

    # Load configuration
    config = load_config()

    # Extract settings from config
    keywords = config.get("keywords", [])
    max_urls = config["scraping"].get("max_urls_per_keyword", 5)
    english_only = config["scraping"].get("english_only", True)
    delay = config["scraping"].get("delay", 1)
    timeout = config["scraping"].get("timeout", 10)
    user_agent = config["scraping"].get("user_agent", "Mozilla/5.0")

    cache_dir = config["output"].get("cache_dir", "cache")
    output_dir = config["output"].get("output_dir", "output")
    index_file = config["output"].get("index_file", "index.json")

    remove_urls = config["cleaning"].get("remove_urls", True)
    remove_emails = config["cleaning"].get("remove_emails", True)
    normalize_whitespace = config["cleaning"].get("normalize_whitespace", True)

    if not keywords:
        logger.error("No keywords found in configuration")
        sys.exit(1)

    logger.info(f"Keywords to search: {', '.join(keywords)}")
    logger.info(f"Max URLs per keyword: {max_urls}")
    logger.info(f"English only: {english_only}")

    # Get Scrapy project settings and override with config values
    settings = get_project_settings()

    # Override settings from config.yaml
    settings.set("USER_AGENT", user_agent)
    settings.set("DOWNLOAD_DELAY", delay)
    settings.set("DOWNLOAD_TIMEOUT", timeout)
    settings.set("ENGLISH_ONLY", english_only)
    settings.set("CACHE_DIR", cache_dir)
    settings.set("OUTPUT_DIR", output_dir)
    settings.set("INDEX_FILE", index_file)
    settings.set("CLEANING_REMOVE_URLS", remove_urls)
    settings.set("CLEANING_REMOVE_EMAILS", remove_emails)
    settings.set("CLEANING_NORMALIZE_WHITESPACE", normalize_whitespace)

    # Ensure directories exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create and run crawler
    process = CrawlerProcess(settings)

    # Run the spider with keywords from config
    process.crawl(
        "keyword_spider",
        keywords=keywords,
        max_urls_per_keyword=max_urls,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Starting Scrapy spider...")
    logger.info("=" * 80 + "\n")

    # This will block until the spider is finished
    process.start()

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed!")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Index file: {os.path.join(output_dir, index_file)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

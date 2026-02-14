"""
Main Pipeline Script
Orchestrates the web scraping, HTML to Markdown conversion, cleaning, and output generation.
"""
import os
import sys
import yaml
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.scraper import WebScraper
from src.parser import MarkdownParser
from src.cleaner import MarkdownCleaner
from src.output_manager import OutputManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)


def main():
    """Main pipeline execution"""
    logger.info("=" * 80)
    logger.info("Starting Web Scraper Pipeline")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Extract settings
    keywords = config.get('keywords', [])
    cache_dir = config['output'].get('cache_dir', 'cache')
    
    if not keywords:
        logger.error("No keywords found in configuration")
        sys.exit(1)
    
    logger.info(f"Keywords to search: {', '.join(keywords)}")
    
    # Initialize components
    logger.info("\nInitializing components...")
    scraper = WebScraper(config)
    parser = MarkdownParser(config)
    cleaner = MarkdownCleaner(config)
    output_manager = OutputManager(config)
    
    # Step 1: Scrape websites
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Scraping websites")
    logger.info("=" * 80)
    scraped_results = scraper.scrape_keywords(keywords, cache_dir)
    
    if not scraped_results:
        logger.warning("No content was scraped. Exiting.")
        sys.exit(0)
    
    # Step 2: Convert HTML to Markdown
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Converting HTML to Markdown")
    logger.info("=" * 80)
    html_contents = [result[2] for result in scraped_results]  # Extract HTML content
    markdown_contents = parser.parse_batch(html_contents)
    
    # Step 3: Clean Markdown
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Cleaning Markdown content")
    logger.info("=" * 80)
    cleaned_contents = cleaner.clean_batch(markdown_contents)
    
    # Step 4: Generate output files and index
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Generating output files and index")
    logger.info("=" * 80)
    output_files = output_manager.process_scraped_data(scraped_results, cleaned_contents)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Total URLs scraped: {len(scraped_results)}")
    logger.info(f"Total markdown files created: {len(output_files)}")
    logger.info(f"Output directory: {config['output']['output_dir']}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Index file: {os.path.join(config['output']['output_dir'], config['output']['index_file'])}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

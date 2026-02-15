"""
Scrapy Items Module

Defines the data structures for scraped content flowing through the pipeline.
"""

import scrapy


class WebPageItem(scrapy.Item):
    """
    Item representing a scraped web page.

    This item flows through the pipeline and gets enriched at each stage:
    1. Spider: url, keyword, html_content
    2. LanguageFilterPipeline: language_detected
    3. HtmlCachePipeline: cache_file
    4. MarkdownConversionPipeline: markdown_content
    5. MarkdownCleaningPipeline: cleaned_markdown
    6. HashGenerationPipeline: content_hash
    7. OutputPipeline: output_file, timestamp
    """

    # Core fields from spider
    url = scrapy.Field()
    keyword = scrapy.Field()
    html_content = scrapy.Field()
    depth = scrapy.Field()  # 0 = from url.txt, 1+ = discovered from links

    # Language detection
    language_detected = scrapy.Field()

    # Caching
    cache_file = scrapy.Field()

    # Processing fields
    markdown_content = scrapy.Field()
    cleaned_markdown = scrapy.Field()

    # Output fields
    content_hash = scrapy.Field()
    output_file = scrapy.Field()
    timestamp = scrapy.Field()

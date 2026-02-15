"""
Scrapy Pipelines Module

Defines processing pipelines for scraped items. Each pipeline handles
a specific step in the data processing workflow.

Pipeline Order (configured in settings.py):
1. LanguageFilterPipeline (100) - Filter non-English pages
2. HtmlCachePipeline (200) - Cache HTML to disk
3. MarkdownConversionPipeline (300) - Convert HTML to Markdown
4. MarkdownCleaningPipeline (400) - Clean markdown content
5. HashGenerationPipeline (500) - Generate content hash
6. OutputPipeline (600) - Save markdown files and create index
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime

import html2text
from bs4 import BeautifulSoup
from scrapy.exceptions import DropItem

logger = logging.getLogger(__name__)


class LanguageFilterPipeline:
    """
    Filter out non-English pages.

    Checks HTML lang attribute and meta tags to determine page language.
    Drops items that are not in English when english_only is enabled.
    """

    def __init__(self, english_only=True):
        """
        Initialize the language filter.

        Args:
            english_only: If True, drop non-English pages
        """
        self.english_only = english_only

    @classmethod
    def from_crawler(cls, crawler):
        """Create pipeline instance from crawler settings."""
        return cls(
            english_only=crawler.settings.getbool("ENGLISH_ONLY", True),
        )

    def process_item(self, item):
        """Check if page is in English and filter accordingly."""
        if not self.english_only:
            item["language_detected"] = "unknown"
            return item

        html_content = item.get("html_content", "")
        url = item.get("url", "")

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Check html lang attribute
            html_tag = soup.find("html")
            if html_tag and html_tag.get("lang"):
                lang = html_tag.get("lang", "").lower()
                if lang.startswith("en"):
                    item["language_detected"] = lang
                    return item
                elif lang:
                    logger.info(f"Non-English page ({lang}), dropping: {url}")
                    raise DropItem(f"Non-English page: {lang}")

            # Check meta language tags
            meta_lang = soup.find("meta", attrs={"http-equiv": "content-language"})
            if meta_lang:
                content = meta_lang.get("content", "").lower()
                if content.startswith("en"):
                    item["language_detected"] = content
                    return item
                elif content:
                    logger.info(f"Non-English page ({content}), dropping: {url}")
                    raise DropItem(f"Non-English page: {content}")

            # No explicit language marker, assume English
            item["language_detected"] = "unknown-assumed-en"
            return item

        except DropItem:
            raise
        except Exception as e:
            logger.warning(f"Error checking language for {url}: {e}")
            item["language_detected"] = "error"
            return item


class HtmlCachePipeline:
    """
    Cache HTML content to disk.

    Saves raw HTML to cache directory for potential reprocessing.
    Uses MD5 hash of URL as filename.
    """

    def __init__(self, cache_dir):
        """
        Initialize the HTML cache pipeline.

        Args:
            cache_dir: Directory to store cached HTML files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @classmethod
    def from_crawler(cls, crawler):
        """Create pipeline instance from crawler settings."""
        return cls(
            cache_dir=crawler.settings.get("CACHE_DIR", "cache"),
        )

    def process_item(self, item):
        """Cache HTML content to disk."""
        url = item.get("url", "")
        html_content = item.get("html_content", "")

        # Generate cache filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{url_hash}.html")

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.debug(f"Cached HTML: {cache_file}")
            item["cache_file"] = cache_file
        except Exception as e:
            logger.error(f"Error caching HTML for {url}: {e}")
            item["cache_file"] = None

        return item


class MarkdownConversionPipeline:
    """
    Convert HTML content to Markdown.

    Uses html2text library for conversion with sensible defaults.
    """

    def __init__(self):
        """Initialize the Markdown converter."""
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = False
        self.converter.ignore_images = False
        self.converter.ignore_emphasis = False
        self.converter.body_width = 0  # Don't wrap lines
        self.converter.single_line_break = False

    def process_item(self, item):
        """Convert HTML to Markdown."""
        html_content = item.get("html_content", "")
        url = item.get("url", "")

        try:
            markdown = self.converter.handle(html_content)
            if not markdown or not markdown.strip():
                logger.warning(f"Empty markdown result for {url}")
                raise DropItem(f"Empty markdown content for {url}")
            item["markdown_content"] = markdown
            logger.debug(f"Converted to markdown: {url}")
        except DropItem:
            raise
        except Exception as e:
            logger.error(f"Error converting HTML to markdown for {url}: {e}")
            raise DropItem(f"Markdown conversion failed: {e}")

        return item


class MarkdownCleaningPipeline:
    """
    Clean markdown content.

    Removes URLs, emails, common junk patterns, and normalizes whitespace.
    """

    def __init__(self, remove_urls=True, remove_emails=True, normalize_whitespace=True):
        """
        Initialize the Markdown cleaner.

        Args:
            remove_urls: If True, remove URLs from content
            remove_emails: If True, remove email addresses
            normalize_whitespace: If True, normalize whitespace
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace

        # Common junk patterns to remove
        self.junk_patterns = [
            r"(?i)(skip to (main )?content|jump to navigation)",
            r"(?i)(cookie policy|privacy policy|terms of service|terms and conditions)",
            r"(?i)(subscribe to (our )?newsletter)",
            r"(?i)(follow us on|share on social media)",
            r"(?i)(copyright|©)\s*\d{4}",
            r"(?i)(all rights reserved)",
        ]

    @classmethod
    def from_crawler(cls, crawler):
        """Create pipeline instance from crawler settings."""
        return cls(
            remove_urls=crawler.settings.getbool("CLEANING_REMOVE_URLS", True),
            remove_emails=crawler.settings.getbool("CLEANING_REMOVE_EMAILS", True),
            normalize_whitespace=crawler.settings.getbool(
                "CLEANING_NORMALIZE_WHITESPACE", True
            ),
        )

    def _remove_urls(self, text):
        """Remove URLs from text."""
        # Remove markdown links but keep the link text
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        # Remove standalone URLs
        text = re.sub(r"https?://[^\s\)]+", "", text)
        text = re.sub(r"www\.[^\s]+", "", text)
        return text

    def _remove_emails(self, text):
        """Remove email addresses from text."""
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
        return text

    def _normalize_whitespace(self, text):
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)
        # Replace multiple newlines with maximum 2 newlines
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        # Remove trailing whitespace from lines
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

        # Handle leading whitespace (preserve code blocks)
        lines = text.split("\n")
        cleaned_lines = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                cleaned_lines.append(line)
            elif in_code_block:
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line.lstrip())

        text = "\n".join(cleaned_lines)
        return text.strip()

    def _remove_junk(self, text):
        """Remove common junk patterns from text."""
        for pattern in self.junk_patterns:
            text = re.sub(pattern, "", text)
        return text

    def _remove_empty_images(self, text):
        """Remove markdown image syntax with empty URLs like ![alt text]()."""
        text = re.sub(r"!\[[^\]]*\]\(\)", "", text)
        return text

    def process_item(self, item):
        """Clean markdown content."""
        markdown = item.get("markdown_content", "")

        if self.remove_urls:
            markdown = self._remove_urls(markdown)

        if self.remove_emails:
            markdown = self._remove_emails(markdown)

        # Always remove empty images and junk
        markdown = self._remove_empty_images(markdown)
        markdown = self._remove_junk(markdown)

        if self.normalize_whitespace:
            markdown = self._normalize_whitespace(markdown)

        item["cleaned_markdown"] = markdown
        logger.debug(f"Cleaned markdown for: {item.get('url', '')}")

        return item


class HashGenerationPipeline:
    """
    Generate SHA256 hash for content.

    Hash is used as filename for deduplication and identification.
    """

    def process_item(self, item):
        """Generate content hash."""
        cleaned_markdown = item.get("cleaned_markdown", "")
        content_hash = hashlib.sha256(cleaned_markdown.encode("utf-8")).hexdigest()
        item["content_hash"] = content_hash
        item["timestamp"] = datetime.now().isoformat()
        logger.debug(f"Generated hash: {content_hash[:16]}...")

        return item


class OutputPipeline:
    """
    Save markdown files and generate index.

    Final pipeline that persists processed content to disk and
    maintains a JSON index of all scraped content.
    """

    def __init__(self, output_dir, index_file, save_interval=10):
        """
        Initialize the output pipeline.

        Args:
            output_dir: Directory to save markdown files
            index_file: Path to the JSON index file (in main directory)
            save_interval: Save index every N items (for crash recovery)
        """
        self.output_dir = output_dir
        self.index_file = index_file
        self.save_interval = save_interval
        self.items_since_save = 0
        os.makedirs(output_dir, exist_ok=True)

        # Load existing index for resuming
        self.index_data = self._load_existing_index()

    def _load_existing_index(self):
        """Load existing index file to allow resuming."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded existing index with {len(data)} entries")
                return data
            except Exception as e:
                logger.warning(f"Error loading existing index: {e}")
        return []

    def _save_index(self):
        """Save index file to disk."""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.index_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved index: {len(self.index_data)} entries")
        except Exception as e:
            logger.error(f"Error saving index file: {e}")

    @classmethod
    def from_crawler(cls, crawler):
        """Create pipeline instance from crawler settings."""
        return cls(
            output_dir=crawler.settings.get("OUTPUT_DIR", "output"),
            index_file=crawler.settings.get("INDEX_FILE", "index.json"),
            save_interval=crawler.settings.getint("INDEX_SAVE_INTERVAL", 10),
        )

    def process_item(self, item):
        """Save markdown file and add to index."""
        content_hash = item.get("content_hash", "")
        cleaned_markdown = item.get("cleaned_markdown", "")
        url = item.get("url", "")

        filename = f"{content_hash}.md"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(cleaned_markdown)

            logger.info(f"Saved: {filename}")
            item["output_file"] = filepath

            # Add to index
            index_entry = {
                "hash": content_hash,
                "filename": filename,
                "source_url": url,
                "depth": item.get("depth", 0),
                "scraped_at": item.get("timestamp", ""),
                "content_length": len(cleaned_markdown),
                "language_detected": item.get("language_detected", ""),
            }
            self.index_data.append(index_entry)

            # Periodic save for crash recovery
            self.items_since_save += 1
            if self.items_since_save >= self.save_interval:
                self._save_index()
                self.items_since_save = 0

        except Exception as e:
            logger.error(f"Error saving markdown for {url}: {e}")
            item["output_file"] = None

        return item

    def close_spider(self, spider):
        """Save final index file when spider closes."""
        self._save_index()
        logger.info(f"Final index saved: {self.index_file}")
        logger.info(f"Total entries: {len(self.index_data)}")

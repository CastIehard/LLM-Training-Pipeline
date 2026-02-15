# UTN-3-LLM-Final-Project

A Scrapy-based web scraping pipeline that searches for content using keywords, converts HTML to Markdown, cleans the content, and outputs organized files with metadata.

## Features

- **Keyword-based web scraping**: Search DuckDuckGo and scrape content based on configurable keywords
- **Asynchronous scraping**: Built on Scrapy for fast, concurrent requests
- **URL deduplication**: Automatically avoids scraping duplicate URLs
- **English site filtering**: Optionally filter for English-language websites only
- **HTML caching**: Caches HTML content for reference
- **HTML to Markdown conversion**: Converts HTML to clean Markdown format via pipeline
- **Content cleaning**: Uses regex patterns to remove URLs, emails, and other junk
- **Hash-based file naming**: Generates SHA256 hash from content for unique file naming
- **Metadata tracking**: Creates JSON index with source URL, keywords, timestamps, and more

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CastIehard/UTN-3-LLM-Final-Project.git
cd UTN-3-LLM-Final-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize the scraping behavior:

- **keywords**: List of search keywords
- **scraping**: Scraping settings (max URLs, timeout, delays, English-only mode)
- **output**: Output and cache directory paths
- **cleaning**: Content cleaning options (remove URLs, emails, whitespace normalization)

Example configuration:
```yaml
keywords:
  - "artificial intelligence"
  - "machine learning"

scraping:
  max_urls_per_keyword: 5
  english_only: true
  timeout: 10
  delay: 1
```

## Usage

Run the pipeline:
```bash
python main.py
```

Or use Scrapy directly:
```bash
scrapy crawl keyword_spider -a keywords="artificial intelligence,machine learning"
```

## Pipeline Architecture

The scraper uses Scrapy pipelines for modular data processing:

1. **LanguageFilterPipeline** - Filters out non-English pages
2. **HtmlCachePipeline** - Caches raw HTML to disk
3. **MarkdownConversionPipeline** - Converts HTML to Markdown
4. **MarkdownCleaningPipeline** - Removes URLs, emails, junk patterns
5. **HashGenerationPipeline** - Generates SHA256 content hash
6. **OutputPipeline** - Saves markdown files and creates JSON index

## Output Structure

```
output/
├── <hash1>.md          # Cleaned markdown content
├── <hash2>.md
├── ...
└── index.json          # Metadata index

cache/
├── <url_hash1>.html    # Cached HTML files
├── <url_hash2>.html
└── ...
```

### Index Format

The `index.json` file contains metadata for each scraped page:
```json
[
  {
    "hash": "abc123...",
    "filename": "abc123....md",
    "source_url": "https://example.com/article",
    "keyword": "artificial intelligence",
    "scraped_at": "2026-02-14T15:23:00",
    "content_length": 5432,
    "cache_file": "cache/def456....html",
    "language_detected": "en"
  }
]
```

## Project Structure

```
.
├── main.py                 # Entry point (runs Scrapy programmatically)
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── scrapy.cfg              # Scrapy project configuration
├── web_scraper/
│   ├── __init__.py
│   ├── items.py            # Scrapy item definitions
│   ├── pipelines.py        # Processing pipelines
│   ├── settings.py         # Scrapy settings
│   └── spiders/
│       ├── __init__.py
│       └── keyword_spider.py  # Main spider
├── output/                 # Output markdown files (generated)
└── cache/                  # Cached HTML files (generated)
```

## Requirements

- Python 3.8+
- scrapy
- beautifulsoup4
- pyyaml
- html2text
- lxml

## License

See LICENSE file for details.
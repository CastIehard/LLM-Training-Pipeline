# UTN-3-LLM-Final-Project

A Scrapy-based web scraping pipeline that scrapes URLs from a file, optionally follows links up to a configurable depth, converts HTML to Markdown, cleans the content, and outputs organized files with metadata.

## Features

- **URL-based web scraping**: Scrape URLs from `url.txt` file with configurable depth
- **Depth-based crawling**: Follow links found on scraped pages up to maximum depth (0 = seed URLs only, 1 = links from seeds, etc.)
- **Asynchronous scraping**: Built on Scrapy for fast, concurrent requests
- **URL deduplication**: Automatically avoids scraping duplicate URLs
- **English site filtering**: Optionally filter for English-language websites only
- **HTML caching**: Caches HTML content for reference
- **HTML to Markdown conversion**: Converts HTML to clean Markdown format via pipeline
- **Content cleaning**: Uses regex patterns to remove URLs, emails, and other junk
- **Hash-based file naming**: Generates SHA256 hash from content for unique file naming
- **Metadata tracking**: Creates JSON index with source URL, depth level, timestamps, and more

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

- **url_file**: Path to the file containing URLs to scrape (one per line)
- **scraping**: Scraping settings (max crawl depth, timeout, delays, English-only mode, concurrent requests, etc.)
- **output**: Output and cache directory paths
- **cleaning**: Content cleaning options (remove URLs, emails, whitespace normalization)

Example configuration:
```yaml
url_file: "url.txt"

scraping:
  max_depth: 1
  english_only: false
  timeout: 3
  delay: 0.1
  concurrent_requests: 8
```

## Usage

1. Add URLs to scrape in `url.txt` (one URL per line):
```
https://example.com
https://another.com
https://example.org/page
```

2. Run the pipeline:
```bash
python main.py
```

The pipeline will:
1. Read URLs from `url.txt`
2. Check against existing entries in `index.json` to avoid duplicates
3. Scrape each URL and follow links up to `max_depth` level
4. Convert HTML to Markdown, clean content, and generate hashes
5. Save processed files to `output/` directory with metadata in `index.json`

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
    "depth": 0,
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
├── url.txt                 # Input file with URLs to scrape (one per line)
├── scrapy.cfg              # Scrapy project configuration
├── web_scraper/
│   ├── __init__.py
│   ├── items.py            # Scrapy item definitions
│   ├── pipelines.py        # Processing pipelines
│   ├── middlewares.py      # Custom middlewares
│   └── spiders/
│       ├── __init__.py
│       └── url_spider.py   # Main spider for URL scraping
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
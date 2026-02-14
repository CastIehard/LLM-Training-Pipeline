# UTN-3-LLM-Final-Project

A Python-based web scraping pipeline that searches for content using keywords, converts HTML to Markdown, cleans the content, and outputs organized files with metadata.

## Features

- **Keyword-based web scraping**: Search and scrape content based on configurable keywords
- **URL deduplication**: Automatically avoids scraping duplicate URLs
- **English site filtering**: Optionally filter for English-language websites only
- **HTML caching**: Caches HTML content for reference
- **HTML to Markdown conversion**: Converts HTML to clean Markdown format on-the-fly
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

The pipeline will:
1. Search for URLs using the configured keywords
2. Scrape the found URLs (avoiding duplicates)
3. Cache HTML content in the `cache/` directory
4. Convert HTML to Markdown
5. Clean the Markdown content (remove URLs, emails, junk)
6. Generate SHA256 hash from content
7. Save cleaned content as `<hash>.md` in the `output/` directory
8. Create an `index.json` file with metadata for all scraped content

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
    "cache_file": "cache/def456....html"
  }
]
```

## Project Structure

```
.
├── main.py                 # Main pipeline script
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── scraper.py         # Web scraper module
│   ├── parser.py          # HTML to Markdown parser
│   ├── cleaner.py         # Markdown cleaner
│   └── output_manager.py  # Output and index manager
├── output/                 # Output markdown files (generated)
└── cache/                  # Cached HTML files (generated)
```

## Requirements

- Python 3.7+
- requests
- beautifulsoup4
- pyyaml
- html2text
- lxml

## License

See LICENSE file for details.
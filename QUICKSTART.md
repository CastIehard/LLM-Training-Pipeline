# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/CastIehard/UTN-3-LLM-Final-Project.git
cd UTN-3-LLM-Final-Project

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Configure your keywords and settings

Edit `config.yaml` to set your keywords and preferences:

```yaml
keywords:
  - "your keyword 1"
  - "your keyword 2"

scraping:
  max_urls_per_keyword: 5  # Number of URLs to scrape per keyword
  english_only: true        # Only scrape English sites
```

### 2. Run the pipeline

**Option A: With real web scraping**
```bash
python main.py
```

**Option B: Test with demo data (no internet required)**
```bash
python demo.py
```

### 3. Check the results

After running, check the `output/` directory:
- `output/<hash>.md` - Your cleaned markdown files
- `output/index.json` - Metadata about each file

## Output Format

### Markdown Files
Each file is named using a SHA256 hash of its content:
```
8a964b8ed84a307cb82b4a4167d10553b6d94c174da80380cebb4009b49c4ef8.md
```

### Index File
The `index.json` file contains metadata:
```json
{
  "hash": "8a964b8...",
  "filename": "8a964b8....md",
  "source_url": "https://example.com/article",
  "keyword": "artificial intelligence",
  "scraped_at": "2026-02-14T15:23:00",
  "content_length": 1234,
  "cache_file": "cache/abc123.html"
}
```

## Configuration Options

### Keywords
```yaml
keywords:
  - "machine learning"
  - "deep learning"
```

### Scraping Settings
```yaml
scraping:
  max_urls_per_keyword: 5    # URLs to scrape per keyword
  english_only: true          # Filter for English sites
  timeout: 10                 # Request timeout (seconds)
  delay: 1                    # Delay between requests (seconds)
```

### Cleaning Settings
```yaml
cleaning:
  remove_urls: true           # Remove URLs from content
  remove_emails: true         # Remove email addresses
  normalize_whitespace: true  # Clean up whitespace
```

## Testing

Run the test suite:
```bash
python test_pipeline.py
```

## Troubleshooting

### No results found
- Check your internet connection
- Try different keywords
- Increase `max_urls_per_keyword` in config.yaml

### Permission errors
- Make sure you have write permissions in the project directory
- The script will create `output/` and `cache/` directories automatically

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Use Python 3.7 or higher

## Advanced Usage

### Custom configuration file
```bash
# Edit main.py and change the config path
config = load_config('my_custom_config.yaml')
```

### Process existing HTML files
You can modify the scraper to process local HTML files instead of scraping:
```python
from src.parser import MarkdownParser
from src.cleaner import MarkdownCleaner

parser = MarkdownParser()
cleaner = MarkdownCleaner(config)

with open('myfile.html', 'r') as f:
    html = f.read()

markdown = parser.html_to_markdown(html)
cleaned = cleaner.clean(markdown)
```

## Project Structure
```
UTN-3-LLM-Final-Project/
├── config.yaml          # Configuration file
├── main.py              # Main pipeline script
├── demo.py              # Demo with mock data
├── test_pipeline.py     # Unit tests
├── requirements.txt     # Python dependencies
├── src/
│   ├── scraper.py       # Web scraping module
│   ├── parser.py        # HTML to Markdown converter
│   ├── cleaner.py       # Content cleaning module
│   └── output_manager.py # File output and indexing
├── output/              # Generated markdown files
│   ├── <hash>.md
│   └── index.json
└── cache/               # Cached HTML files
    └── <url_hash>.html
```

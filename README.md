# UTN-3-LLM-Final-Project

A multi-stage data pipeline for scraping web content, classifying it, generating Q&A pairs, and benchmarking LLM performance. The project consists of four main modules:

1. **Web Scraping** (`1_webscraping/`) - Scrapes URLs, converts HTML to Markdown, and creates a structured index
2. **LLM Classification** (`2_llm_classification/`) - Classifies URLs into categories using OpenAI API or local LLM (LM Studio)
3. **LLM Q&A Generation** (`3_llm_questions/`) - Generates question-answer pairs from content
4. **Benchmark** (`4_benchmark/`) - Benchmarks LLM performance on Q&A pairs with judge evaluation

## Project Structure

```
.
├── .env                        # Environment variables (API keys)
├── requirements.txt            # Python dependencies
├── README.md
├── LICENSE
│
├── 1_webscraping/              # Stage 1: Web Scraping
│   ├── main.py                 # Entry point for scraping
│   ├── config.yaml             # Scraping configuration
│   ├── url.txt                 # Input URLs (one per line)
│   ├── cache/                  # Cached HTML files
│   └── web_scraper/            # Scrapy spider and pipelines
│
├── 2_llm_classification/       # Stage 2: LLM Classification
│   ├── main.py                 # Entry point for classification
│   └── config.yaml             # LLM and category configuration
│
├── 3_llm_questions/            # Stage 3: Q&A Generation
│   ├── main.py                 # Entry point for Q&A generation
│   └── config.yaml             # Generation settings
│
├── 4_benchmark/                # Stage 4: LLM Benchmark
│   ├── main.py                 # Entry point for benchmarking
│   ├── config.yaml             # Benchmark configuration
│   ├── benchmark.jsonl         # Sampled questions for benchmark
│   └── results/                # Timestamped result folders
│
├── model/                      # Local HuggingFace models (optional)
│
└── data/                       # Shared data directory
    ├── index.json              # Metadata index (updated by stages 1 & 2)
    ├── raw_md/                 # Markdown files from scraping
    └── llm_qna.jsonl           # Generated Q&A pairs (JSONL format)
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key (if using OpenAI):
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

3. Run the pipeline stages in order:
```bash
# Stage 1: Scrape URLs
python 1_webscraping/main.py

# Stage 2: Classify URLs
python 2_llm_classification/main.py

# Stage 3: Generate Q&A pairs
python 3_llm_questions/main.py

# Stage 4: Benchmark LLM performance
python 4_benchmark/main.py
```

---

## Stage 1: Web Scraping

A Scrapy-based web scraping pipeline that scrapes URLs from a file, optionally follows links up to a configurable depth, converts HTML to Markdown, cleans the content, and outputs organized files with metadata.

### URL List Generation

The `url.txt` file was generated using **ChatGPT Deep Research mode**. The prompt asked for relevant URLs for international students at UTN Nuremberg. ChatGPT performed 427 searches and compiled the resulting links.

See the full prompt and execution: [ChatGPT Deep Research Session](https://chatgpt.com/share/e/69921396-c498-8003-9298-2cfc4ea71616)

### Features

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
    "language_detected": "en",
    "category": "UTN"
  }
]
```

---

## Stage 2: LLM Classification

Classifies each URL in the index into predefined categories using either OpenAI API or a local LLM via LM Studio.

### Features

- **Dual LLM Support**: Use OpenAI API or local LLM (LM Studio)
- **Configurable Categories**: Define custom categories in config.yaml
- **Incremental Processing**: Only classifies URLs without valid category
- **Idempotent**: Running multiple times won't re-classify existing entries

### Configuration

Edit `2_llm_classification/config.yaml`:

```yaml
llm:
  provider: "local"  # or "openai"
  
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o-mini"
    
  local:
    base_url: "http://localhost:1234/v1"
    model: "local-model"

categories:
  - name: "UTN"
    description: "Content related to UTN"
  - name: "Germany"
    description: "General Germany information"
  - name: "Nuremberg"
    description: "Nuremberg-specific content"
  - name: "Studies"
    description: "Study-related content"
  - name: "Other"
    description: "Other content"
```

### Usage

```bash
python 2_llm_classification/main.py
```

The script will:
1. Load `data/index.json`
2. Check each entry for existing valid category
3. Send uncategorized URLs to LLM for classification
4. Update `index.json` with category field

### Using Local LLM (LM Studio)

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load a model and start the local server (default: `http://localhost:1234/v1`)
3. Set `provider: "local"` in config.yaml
4. Run the classification script

---

## Stage 3: LLM Q&A Generation

Generates question-answer pairs from the scraped content. The number of questions is calculated based on content length.

### Features

- **Dynamic Question Count**: Number of questions scales with content length
- **Dual LLM Support**: Use OpenAI API or local LLM (LM Studio)
- **Incremental Processing**: Skips already processed documents
- **Skip Trash**: Optionally skip documents classified as "Trash"
- **JSONL Output**: Each Q&A pair stored as a separate JSON line

### Configuration

Edit `3_llm_questions/config.yaml`:

```yaml
llm:
  provider: "openai"  # or "local"
  openai:
    model: "gpt-4.1-nano"

generation:
  # Characters of content per question
  content_length_per_question: 500
  # Range for question count (±)
  question_range: 2
  min_questions: 1
  max_questions: 20

processing:
  skip_trash: true
```

### Usage

```bash
python 3_llm_questions/main.py
```

The script will:
1. Load `data/index.json` and check which hashes are already in `llm_qna.jsonl`
2. Calculate question count based on content length
3. Send content to LLM for Q&A generation
4. Append Q&A pairs to `data/llm_qna.jsonl`

### Output Format (JSONL)

Each line in `llm_qna.jsonl` is a JSON object:
```json
{"hash": "abc123...", "question": "What is...?", "answer": "It is...", "model": "gpt-4.1-nano", "category": "UTN"}
{"hash": "abc123...", "question": "How does...?", "answer": "It works by...", "model": "gpt-4.1-nano", "category": "UTN"}
```

---

## Stage 4: LLM Benchmark

Benchmarks LLM performance by sampling questions from each category, getting answers from a test LLM, and using a judge LLM to evaluate the responses.

### Features

- **Three LLM Providers**: OpenAI API, local LLM (LM Studio), or HuggingFace models
- **Category-based Sampling**: Sample equal questions per category
- **Judge LLM Evaluation**: Automated scoring (0.0, 0.5, 1.0)
- **Detailed Results**: Timestamped folders with summary and detailed results
- **HuggingFace Support**: Test local models downloaded from HuggingFace Hub

### Configuration

Edit `4_benchmark/config.yaml`:

```yaml
answer_llm:
  # Provider: "openai", "local", or "huggingface"
  provider: "huggingface"
  
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4.1-nano"
    
  local:
    base_url: "http://localhost:1234/v1"
    model: "local-model"
    
  huggingface:
    model: "Qwen/Qwen2.5-0.5B-Instruct"
    model_dir: "model"  # Local cache directory
    device: "auto"      # auto, cpu, cuda, mps
    dtype: "auto"       # auto, float16, float32, bfloat16

judge_llm:
  provider: "openai"
  openai:
    model: "gpt-4.1-nano"

benchmark:
  questions_per_category: 10
  categories: []  # Empty = all categories
```

### Usage

```bash
python 4_benchmark/main.py
```

The script will:
1. Sample questions from each category (or reuse existing `benchmark.jsonl`)
2. Get answers from the test LLM (answer_llm)
3. Have the judge LLM score each answer
4. Save results to timestamped folder in `4_benchmark/results/`

### Using HuggingFace Models

1. Set `provider: "huggingface"` in answer_llm config
2. Specify the model name (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)
3. Run the benchmark - model will be downloaded automatically to `model/` directory
4. Subsequent runs will use the cached model

### Output Structure

```
4_benchmark/results/
└── 20260216_120000/
    ├── summary.json          # Overall stats and per-category breakdown
    └── detailed_results.jsonl # Individual question results
```

### Summary Format

```json
{
  "answer_model": {"provider": "huggingface", "model": "Qwen/Qwen2.5-0.5B-Instruct"},
  "judge_model": {"provider": "openai", "model": "gpt-4.1-nano"},
  "stats": {
    "total_questions": 50,
    "average_score": 0.72,
    "full_correct": 30,
    "partial_correct": 12,
    "wrong": 8
  },
  "stats_by_category": {...}
}
```

---

## Requirements

- Python 3.10+
- scrapy
- beautifulsoup4
- pyyaml
- html2text
- lxml
- openai
- python-dotenv
- tenacity
- tqdm
- transformers (optional, for HuggingFace models)
- torch (optional, for HuggingFace models)
- accelerate (optional, for HuggingFace models)

## License

See LICENSE file for details.
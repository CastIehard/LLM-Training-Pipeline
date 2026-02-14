"""
Example Demo Script
Demonstrates the pipeline with mock data (no actual web scraping).
"""
import os
import sys
import yaml

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.parser import MarkdownParser
from src.cleaner import MarkdownCleaner
from src.output_manager import OutputManager

# Mock HTML content for demonstration
MOCK_HTML_PAGES = [
    {
        'url': 'http://example.com/ai-article-1',
        'keyword': 'artificial intelligence',
        'html': """
        <html>
        <head><title>Introduction to Artificial Intelligence</title></head>
        <body>
            <h1>Introduction to Artificial Intelligence</h1>
            <p>Artificial Intelligence (AI) is transforming the world. Visit http://example.com for more.</p>
            <p>AI systems can learn from data and make decisions. Contact us at info@example.com</p>
            <h2>Applications</h2>
            <ul>
                <li>Computer Vision</li>
                <li>Natural Language Processing</li>
                <li>Robotics</li>
            </ul>
            <footer>Copyright 2026 | Privacy Policy | Terms of Service</footer>
        </body>
        </html>
        """
    },
    {
        'url': 'http://example.com/ml-guide',
        'keyword': 'machine learning',
        'html': """
        <html>
        <head><title>Machine Learning Guide</title></head>
        <body>
            <h1>Getting Started with Machine Learning</h1>
            <p>Machine Learning is a subset of AI. Learn more at http://mlguide.com</p>
            <p>Key concepts include:</p>
            <ul>
                <li>Supervised Learning</li>
                <li>Unsupervised Learning</li>
                <li>Reinforcement Learning</li>
            </ul>
            <p>Subscribe to our newsletter at newsletter@example.com</p>
            <footer>All rights reserved 2026</footer>
        </body>
        </html>
        """
    }
]


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Run demo with mock data"""
    print("=" * 80)
    print("Web Scraper Pipeline Demo (with mock data)")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    print("\nInitializing components...")
    parser = MarkdownParser(config)
    cleaner = MarkdownCleaner(config)
    output_manager = OutputManager(config)
    
    print(f"Processing {len(MOCK_HTML_PAGES)} mock pages...")
    print()
    
    # Process each mock page
    for i, page in enumerate(MOCK_HTML_PAGES, 1):
        print(f"Processing page {i}/{len(MOCK_HTML_PAGES)}")
        print(f"  URL: {page['url']}")
        print(f"  Keyword: {page['keyword']}")
        
        # Convert HTML to Markdown
        markdown = parser.html_to_markdown(page['html'])
        print(f"  ✓ Converted to Markdown ({len(markdown)} chars)")
        
        # Clean Markdown
        cleaned = cleaner.clean(markdown)
        print(f"  ✓ Cleaned Markdown ({len(cleaned)} chars)")
        
        # Generate hash and save
        content_hash = output_manager.generate_hash(cleaned)
        metadata = {
            'url': page['url'],
            'keyword': page['keyword'],
            'cache_file': f"cache/mock_{i}.html"
        }
        
        filepath = output_manager.save_markdown(cleaned, content_hash, metadata)
        print(f"  ✓ Saved as: {os.path.basename(filepath)}")
        print()
    
    # Save index
    index_file = output_manager.save_index()
    
    # Display results
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
    print(f"Output directory: {config['output']['output_dir']}")
    print(f"Files created: {len(MOCK_HTML_PAGES)} markdown files + 1 index file")
    print()
    
    # Show index content
    import json
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    print("Index contents:")
    print(json.dumps(index_data, indent=2))
    print("=" * 80)
    
    # Show sample output
    print("\nSample output file content:")
    print("-" * 80)
    first_file = os.path.join(config['output']['output_dir'], index_data[0]['filename'])
    with open(first_file, 'r') as f:
        content = f.read()
    print(content[:500] + "..." if len(content) > 500 else content)
    print("-" * 80)


if __name__ == '__main__':
    main()

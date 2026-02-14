"""
Unit Tests for Web Scraper Pipeline Components
"""
import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.parser import MarkdownParser
from src.cleaner import MarkdownCleaner
from src.output_manager import OutputManager


def test_markdown_parser():
    """Test HTML to Markdown conversion"""
    print("Testing MarkdownParser...")
    
    parser = MarkdownParser()
    
    # Test basic HTML
    html = "<h1>Test Title</h1><p>Test paragraph with <a href='http://example.com'>a link</a>.</p>"
    markdown = parser.html_to_markdown(html)
    
    assert "Test Title" in markdown
    assert "Test paragraph" in markdown
    print("✓ MarkdownParser test passed")


def test_markdown_cleaner():
    """Test Markdown cleaning"""
    print("Testing MarkdownCleaner...")
    
    config = {
        'cleaning': {
            'remove_urls': True,
            'remove_emails': True,
            'normalize_whitespace': True
        }
    }
    
    cleaner = MarkdownCleaner(config)
    
    # Test URL removal
    markdown = "Check out [this link](http://example.com) and http://test.com for more info."
    cleaned = cleaner.clean(markdown)
    
    assert "http://example.com" not in cleaned
    assert "http://test.com" not in cleaned
    assert "this link" in cleaned
    
    # Test email removal
    markdown_with_email = "Contact us at test@example.com for more info."
    cleaned_email = cleaner.clean(markdown_with_email)
    
    assert "test@example.com" not in cleaned_email
    
    print("✓ MarkdownCleaner test passed")


def test_output_manager():
    """Test OutputManager hash generation and file saving"""
    print("Testing OutputManager...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        config = {
            'output': {
                'output_dir': temp_dir,
                'index_file': 'test_index.json'
            }
        }
        
        output_manager = OutputManager(config)
        
        # Test hash generation
        content = "Test content for hashing"
        hash1 = output_manager.generate_hash(content)
        hash2 = output_manager.generate_hash(content)
        
        assert hash1 == hash2, "Same content should produce same hash"
        assert len(hash1) == 64, "SHA256 hash should be 64 characters"
        
        # Test file saving
        metadata = {
            'url': 'http://test.com',
            'keyword': 'test',
            'cache_file': 'cache/test.html'
        }
        
        filepath = output_manager.save_markdown(content, hash1, metadata)
        assert filepath is not None
        assert os.path.exists(filepath)
        
        # Verify file content
        with open(filepath, 'r') as f:
            saved_content = f.read()
        assert saved_content == content
        
        # Test index saving
        index_file = output_manager.save_index()
        assert index_file is not None
        assert os.path.exists(index_file)
        
        print("✓ OutputManager test passed")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_integration():
    """Test integration of parser, cleaner, and output manager"""
    print("Testing integration...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        config = {
            'output': {
                'output_dir': temp_dir,
                'index_file': 'test_index.json'
            },
            'cleaning': {
                'remove_urls': True,
                'remove_emails': True,
                'normalize_whitespace': True
            }
        }
        
        # Initialize components
        parser = MarkdownParser(config)
        cleaner = MarkdownCleaner(config)
        output_manager = OutputManager(config)
        
        # Test HTML content
        html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Test Article</h1>
            <p>This is a test paragraph with a <a href="http://example.com">link</a>.</p>
            <p>Contact: test@example.com</p>
            <p>More information at http://test.com</p>
        </body>
        </html>
        """
        
        # Pipeline: HTML -> Markdown -> Clean -> Save
        markdown = parser.html_to_markdown(html)
        cleaned = cleaner.clean(markdown)
        content_hash = output_manager.generate_hash(cleaned)
        
        metadata = {
            'url': 'http://example.com/test',
            'keyword': 'test keyword',
            'cache_file': 'cache/test.html'
        }
        
        filepath = output_manager.save_markdown(cleaned, content_hash, metadata)
        output_manager.save_index()
        
        # Verify results
        assert os.path.exists(filepath)
        assert "Test Article" in cleaned
        assert "http://example.com" not in cleaned
        assert "test@example.com" not in cleaned
        
        print("✓ Integration test passed")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def main():
    """Run all tests"""
    print("=" * 60)
    print("Running Unit Tests")
    print("=" * 60)
    
    try:
        test_markdown_parser()
        test_markdown_cleaner()
        test_output_manager()
        test_integration()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
Output Manager Module
Handles hash generation, file output, and index creation.
"""
import os
import json
import hashlib
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OutputManager:
    """Manage output files and index"""
    
    def __init__(self, config):
        """
        Initialize the output manager.
        
        Args:
            config (dict): Configuration dictionary with output settings
        """
        self.config = config
        self.output_dir = config['output'].get('output_dir', 'output')
        self.index_file = os.path.join(
            self.output_dir,
            config['output'].get('index_file', 'index.json')
        )
        self.index_data = []
    
    def generate_hash(self, content):
        """
        Generate SHA256 hash from content.
        
        Args:
            content (str): Content to hash
            
        Returns:
            str: Hexadecimal hash string
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def save_markdown(self, content, content_hash, metadata):
        """
        Save markdown content to a file named by its hash.
        
        Args:
            content (str): Markdown content
            content_hash (str): Hash of the content
            metadata (dict): Metadata about the content
            
        Returns:
            str: Path to saved file
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        filename = f"{content_hash}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Saved markdown file: {filename}")
            
            # Add to index
            index_entry = {
                'hash': content_hash,
                'filename': filename,
                'source_url': metadata.get('url', ''),
                'keyword': metadata.get('keyword', ''),
                'scraped_at': metadata.get('timestamp', datetime.now().isoformat()),
                'content_length': len(content),
                'cache_file': metadata.get('cache_file', '')
            }
            self.index_data.append(index_entry)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving markdown file {filename}: {e}")
            return None
    
    def save_index(self):
        """
        Save the index to a JSON file.
        
        Returns:
            str: Path to index file
        """
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved index file: {self.index_file}")
            logger.info(f"Total entries in index: {len(self.index_data)}")
            
            return self.index_file
            
        except Exception as e:
            logger.error(f"Error saving index file: {e}")
            return None
    
    def process_scraped_data(self, scraped_results, markdown_contents):
        """
        Process scraped data and save output files.
        
        Args:
            scraped_results (list): List of tuples (url, keyword, html_content, cache_file)
            markdown_contents (list): List of cleaned markdown content strings
            
        Returns:
            list: List of output file paths
        """
        output_files = []
        
        for i, (result, markdown) in enumerate(zip(scraped_results, markdown_contents)):
            url, keyword, html_content, cache_file = result
            
            # Generate hash from cleaned markdown content
            content_hash = self.generate_hash(markdown)
            
            # Prepare metadata
            metadata = {
                'url': url,
                'keyword': keyword,
                'timestamp': datetime.now().isoformat(),
                'cache_file': cache_file
            }
            
            # Save markdown file
            filepath = self.save_markdown(markdown, content_hash, metadata)
            if filepath:
                output_files.append(filepath)
        
        # Save index
        self.save_index()
        
        return output_files

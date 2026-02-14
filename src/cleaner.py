"""
Markdown Cleaning Module
Cleans markdown content by removing URLs, emails, and other junk using regex.
"""
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarkdownCleaner:
    """Clean markdown content using regex patterns"""
    
    def __init__(self, config):
        """
        Initialize the Markdown cleaner.
        
        Args:
            config (dict): Configuration dictionary with cleaning settings
        """
        self.config = config
        self.cleaning_config = config.get('cleaning', {})
    
    def remove_urls(self, text):
        """
        Remove URLs from text.
        
        Args:
            text (str): Text content
            
        Returns:
            str: Text with URLs removed
        """
        # Remove markdown links but keep the link text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove standalone URLs
        text = re.sub(r'https?://[^\s\)]+', '', text)
        text = re.sub(r'www\.[^\s]+', '', text)
        
        return text
    
    def remove_emails(self, text):
        """
        Remove email addresses from text.
        
        Args:
            text (str): Text content
            
        Returns:
            str: Text with emails removed
        """
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        return text
    
    def normalize_whitespace(self, text):
        """
        Normalize whitespace in text.
        
        Args:
            text (str): Text content
            
        Returns:
            str: Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with maximum 2 newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing whitespace from lines
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        
        # Remove leading whitespace from lines (but preserve indentation in code blocks)
        lines = text.split('\n')
        cleaned_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                cleaned_lines.append(line)
            elif in_code_block:
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line.lstrip())
        
        text = '\n'.join(cleaned_lines)
        
        return text.strip()
    
    def remove_common_junk(self, text):
        """
        Remove common junk patterns from markdown.
        
        Args:
            text (str): Text content
            
        Returns:
            str: Cleaned text
        """
        # Remove common navigation/footer text patterns
        junk_patterns = [
            r'(?i)(skip to (main )?content|jump to navigation)',
            r'(?i)(cookie policy|privacy policy|terms of service|terms and conditions)',
            r'(?i)(subscribe to (our )?newsletter)',
            r'(?i)(follow us on|share on social media)',
            r'(?i)(copyright|©)\s*\d{4}',
            r'(?i)(all rights reserved)',
        ]
        
        for pattern in junk_patterns:
            text = re.sub(pattern, '', text)
        
        return text
    
    def clean(self, markdown_content):
        """
        Clean markdown content based on configuration.
        
        Args:
            markdown_content (str): Markdown content to clean
            
        Returns:
            str: Cleaned markdown content
        """
        text = markdown_content
        
        # Apply cleaning based on configuration
        if self.cleaning_config.get('remove_urls', True):
            logger.debug("Removing URLs from markdown")
            text = self.remove_urls(text)
        
        if self.cleaning_config.get('remove_emails', True):
            logger.debug("Removing emails from markdown")
            text = self.remove_emails(text)
        
        # Always remove common junk
        logger.debug("Removing common junk patterns")
        text = self.remove_common_junk(text)
        
        if self.cleaning_config.get('normalize_whitespace', True):
            logger.debug("Normalizing whitespace")
            text = self.normalize_whitespace(text)
        
        return text
    
    def clean_batch(self, markdown_contents):
        """
        Clean multiple markdown contents.
        
        Args:
            markdown_contents (list): List of markdown content strings
            
        Returns:
            list: List of cleaned markdown content strings
        """
        results = []
        for i, content in enumerate(markdown_contents):
            logger.info(f"Cleaning markdown content ({i+1}/{len(markdown_contents)})")
            cleaned = self.clean(content)
            results.append(cleaned)
        return results

"""
HTML to Markdown Parser Module
Converts HTML content to Markdown format using html2text library.
"""
import html2text
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarkdownParser:
    """Convert HTML to Markdown"""
    
    def __init__(self, config=None):
        """
        Initialize the Markdown parser.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
        self.converter = html2text.HTML2Text()
        
        # Configure html2text
        self.converter.ignore_links = False
        self.converter.ignore_images = False
        self.converter.ignore_emphasis = False
        self.converter.body_width = 0  # Don't wrap lines
        self.converter.single_line_break = False
    
    def html_to_markdown(self, html_content):
        """
        Convert HTML content to Markdown.
        
        Args:
            html_content (str): HTML content to convert
            
        Returns:
            str: Markdown content
        """
        try:
            markdown = self.converter.handle(html_content)
            logger.debug("Successfully converted HTML to Markdown")
            return markdown
        except Exception as e:
            logger.error(f"Error converting HTML to Markdown: {e}")
            return ""
    
    def parse_batch(self, html_contents):
        """
        Convert multiple HTML contents to Markdown.
        
        Args:
            html_contents (list): List of HTML content strings
            
        Returns:
            list: List of Markdown content strings
        """
        results = []
        for i, html in enumerate(html_contents):
            logger.info(f"Converting HTML to Markdown ({i+1}/{len(html_contents)})")
            markdown = self.html_to_markdown(html)
            results.append(markdown)
        return results

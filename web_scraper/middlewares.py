"""
Custom Scrapy Middlewares

Contains custom downloader middlewares for the web scraper.
"""

import asyncio
from urllib.parse import urlparse

from scrapy import signals


class SameDomainDelayMiddleware:
    """
    Middleware that only applies download delay when staying on the same domain.

    This avoids unnecessary waiting when switching between different domains.
    Delay is only applied when consecutive requests go to the same domain.
    """

    def __init__(self, delay):
        self.delay = delay
        self.last_domain = None

    @classmethod
    def from_crawler(cls, crawler):
        delay = crawler.settings.getfloat("SAME_DOMAIN_DELAY", 1.0)
        middleware = cls(delay)
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        return middleware

    def spider_opened(self, spider):
        spider.logger.info(
            f"SameDomainDelayMiddleware enabled with {self.delay}s delay"
        )

    async def process_request(self, request, spider):
        """Process request and add delay only for same-domain requests."""
        current_domain = urlparse(request.url).netloc

        if self.last_domain is not None and self.last_domain == current_domain:
            # Same domain - apply delay
            spider.logger.debug(
                f"Same domain ({current_domain}), waiting {self.delay}s"
            )
            await asyncio.sleep(self.delay)
        elif self.last_domain is not None:
            # Different domain - no delay needed
            spider.logger.debug(
                f"Domain changed: {self.last_domain} -> {current_domain}, no delay"
            )

        self.last_domain = current_domain
        return None

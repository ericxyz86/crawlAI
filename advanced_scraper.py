import asyncio
import logging
import uuid
import time
import requests
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from dataclasses import dataclass, asdict

from crawl4ai import (
    AsyncWebCrawler, 
    CrawlerRunConfig, 
    BrowserConfig, 
    PruningContentFilter, 
    DefaultMarkdownGenerator,
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
    URLPatternFilter,
    DomainFilter
)

@dataclass
class CrawlJobStatus:
    job_id: str
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    message: str
    completed: int
    total: int
    results: List[Dict[str, Any]]
    error: Optional[str] = None
    started_at: float = 0
    completed_at: Optional[float] = None

class AdvancedWebScraper:
    def __init__(self, websocket_manager=None):
        self.logger = logging.getLogger(__name__)
        self.active_jobs: Dict[str, CrawlJobStatus] = {}
        self.websocket_manager = websocket_manager
        
    def is_valid_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    async def _send_progress_update(self, job_id: str):
        """Send progress update via WebSocket if available"""
        if self.websocket_manager and job_id in self.active_jobs:
            job_status = self.active_jobs[job_id]
            progress_data = {
                "type": "progress",
                "job_id": job_id,
                "status": job_status.status,
                "message": job_status.message,
                "completed": job_status.completed,
                "total": job_status.total,
                "error": job_status.error
            }
            try:
                await self.websocket_manager.send_progress_update(job_id, progress_data)
                self.logger.debug(f"Sent progress update for job {job_id}: {job_status.status} - {job_status.message}")
            except Exception as e:
                self.logger.warning(f"Failed to send WebSocket update for job {job_id}: {str(e)}")
    
    def _check_robots_txt(self, url: str, user_agent: str = "*") -> bool:
        """Check if crawling is allowed by robots.txt"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            return rp.can_fetch(user_agent, url)
        except Exception as e:
            self.logger.warning(f"Could not check robots.txt for {url}: {str(e)}")
            # If we can't check robots.txt, assume it's allowed
            return True
    
    def _get_crawl_delay(self, url: str, user_agent: str = "*", default_delay: int = 1) -> int:
        """Get crawl delay from robots.txt"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            delay = rp.crawl_delay(user_agent)
            return int(delay) if delay else default_delay
        except Exception as e:
            self.logger.warning(f"Could not get crawl delay from robots.txt for {url}: {str(e)}")
            return default_delay
    
    async def start_advanced_crawl(self, config: Dict[str, Any]) -> str:
        """Start an advanced crawl job and return job ID"""
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        job_status = CrawlJobStatus(
            job_id=job_id,
            status='running',
            message='Initializing crawl...',
            completed=0,
            total=0,
            results=[],
            started_at=time.time()
        )
        self.active_jobs[job_id] = job_status
        
        # Start crawling in background
        asyncio.create_task(self._execute_crawl(job_id, config))
        
        return job_id
    
    async def _execute_crawl(self, job_id: str, config: Dict[str, Any]):
        """Execute the actual crawling process"""
        job_status = self.active_jobs[job_id]
        
        try:
            url = config['url']
            crawl_mode = config.get('crawl_mode', 'single')
            
            if not self.is_valid_url(url):
                raise ValueError(f"Invalid URL: {url}")
            
            if crawl_mode == 'single':
                await self._crawl_single_page(job_id, config)
            elif crawl_mode == 'deep':
                await self._crawl_deep(job_id, config)
            elif crawl_mode == 'sitemap':
                await self._crawl_sitemap(job_id, config)
            elif crawl_mode == 'pattern':
                await self._crawl_pattern(job_id, config)
            else:
                raise ValueError(f"Unknown crawl mode: {crawl_mode}")
                
            job_status.status = 'completed'
            job_status.message = f'Crawl completed successfully! Scraped {len(job_status.results)} pages.'
            job_status.completed_at = time.time()
            
            # Send final completion update via WebSocket
            await self._send_progress_update(job_id)
            self.logger.info(f"Crawl job {job_id} completed successfully with {len(job_status.results)} pages")
            
        except Exception as e:
            self.logger.error(f"Error in crawl job {job_id}: {str(e)}")
            job_status.status = 'failed'
            job_status.error = str(e)
            job_status.message = f'Crawl failed: {str(e)}'
            job_status.completed_at = time.time()
            
            # Send final error update via WebSocket
            await self._send_progress_update(job_id)
            self.logger.error(f"Crawl job {job_id} failed: {str(e)}")
    
    async def _crawl_single_page(self, job_id: str, config: Dict[str, Any]):
        """Crawl a single page with optional infinite scroll"""
        job_status = self.active_jobs[job_id]
        job_status.total = 1
        
        # Check if infinite scroll is enabled
        enable_infinite_scroll = config.get('enable_infinite_scroll') == 'on'
        
        if enable_infinite_scroll:
            job_status.message = 'Scraping single page with infinite scroll...'
        else:
            job_status.message = 'Scraping single page...'
        
        await self._send_progress_update(job_id)
        
        # Use the existing single-page scraping logic
        from scraper import WebScraper
        scraper = WebScraper()
        
        # Prepare options for the scraper
        options = {
            "enable_infinite_scroll": enable_infinite_scroll,
            "max_scrolls": int(config.get('max_scrolls', 10)),
            "scroll_delay": int(config.get('scroll_delay', 2000)),
            "scroll_step": int(config.get('scroll_step', 1000)),
            "content_stability_checks": int(config.get('content_stability_checks', 3)),
            "youtube_optimized": config.get('youtube_optimized') == 'on',
            "human_behavior_simulation": config.get('human_behavior_simulation') == 'on'
        }
        
        result = await scraper.scrape_url(config['url'], options)
        job_status.results.append(result)
        job_status.completed = 1
        await self._send_progress_update(job_id)
    
    async def _crawl_deep(self, job_id: str, config: Dict[str, Any]):
        """Deep crawl with configurable depth and filters"""
        job_status = self.active_jobs[job_id]
        
        url = config['url']
        max_depth = int(config.get('max_depth', 2))
        max_pages = int(config.get('max_pages', 20))
        crawl_delay = int(config.get('crawl_delay', 2))
        same_domain = config.get('same_domain') == 'on'
        include_external = config.get('include_external') == 'on'
        
        # Check robots.txt compliance
        if not self._check_robots_txt(url):
            raise Exception(f"Crawling of {url} is not allowed by robots.txt")
        
        # Get robots.txt crawl delay (use the larger of user setting or robots.txt)
        robots_delay = self._get_crawl_delay(url, default_delay=crawl_delay)
        crawl_delay = max(crawl_delay, robots_delay)
        
        job_status.message = f'Starting deep crawl (depth: {max_depth}, max pages: {max_pages})...'
        job_status.total = max_pages  # This will be updated as we discover more pages
        await self._send_progress_update(job_id)
        
        try:
            browser_config = BrowserConfig(
                headless=True,
                browser_type="chromium"
            )
            
            # Simplified deep crawl using basic crawler with manual link following
            crawler_config = CrawlerRunConfig(
                word_count_threshold=10,
                excluded_tags=["nav", "footer", "aside", "script", "style"],
                remove_overlay_elements=True,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(threshold=0.48)
                ),
                wait_for="3000",
                page_timeout=30000
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                results = []
                visited_urls = set()
                urls_by_depth = {0: [url]}  # Organize URLs by depth for breadth-first crawling
                current_depth = 0
                
                while current_depth <= max_depth and len(results) < max_pages:
                    if job_status.status == 'cancelled':
                        break
                    
                    if current_depth not in urls_by_depth or not urls_by_depth[current_depth]:
                        current_depth += 1
                        continue
                    
                    # Process all URLs at current depth
                    urls_at_current_depth = urls_by_depth[current_depth].copy()
                    urls_by_depth[current_depth] = []
                    
                    for current_url in urls_at_current_depth:
                        if job_status.status == 'cancelled' or len(results) >= max_pages:
                            break
                            
                        if current_url in visited_urls:
                            continue
                            
                        visited_urls.add(current_url)
                        
                        # Check if we should crawl this URL
                        if same_domain:
                            if urlparse(current_url).netloc != urlparse(url).netloc:
                                continue
                        
                        try:
                            result = await crawler.arun(url=current_url, config=crawler_config)
                            
                            if result.success:
                                page_result = {
                                    "success": True,
                                    "url": current_url,
                                    "title": result.metadata.get("title", ""),
                                    "description": result.metadata.get("description", ""),
                                    "markdown": result.markdown,
                                    "links": result.links.get("external", []) + result.links.get("internal", []) if hasattr(result, 'links') and result.links else [],
                                    "images": [],
                                    "metadata": result.metadata,
                                    "word_count": len(result.markdown.split()) if result.markdown else 0,
                                    "depth": current_depth
                                }
                                results.append(page_result)
                                job_status.results.append(page_result)
                                
                                # Extract links for next depth level
                                if current_depth < max_depth and result.links:
                                    next_depth = current_depth + 1
                                    if next_depth not in urls_by_depth:
                                        urls_by_depth[next_depth] = []
                                    
                                    for link_type in ['internal', 'external']:
                                        if link_type in result.links:
                                            for link in result.links[link_type]:
                                                link_url = link.get('href', '')
                                                if link_url and self.is_valid_url(link_url):
                                                    if not include_external and link_type == 'external':
                                                        continue
                                                    if link_url not in visited_urls and link_url not in urls_by_depth[next_depth]:
                                                        urls_by_depth[next_depth].append(link_url)
                            
                            # Update progress with current depth info
                            total_remaining_urls = sum(len(urls) for depth_urls in urls_by_depth.values() for urls in [depth_urls])
                            estimated_total = min(max_pages, len(results) + total_remaining_urls)
                            job_status.total = max(job_status.total, estimated_total)
                            job_status.completed = len(results)
                            
                            # More informative progress message
                            depth_info = f"depth {current_depth}"
                            if current_depth < max_depth and any(urls_by_depth.get(d, []) for d in range(current_depth + 1, max_depth + 1)):
                                depth_info += f" (preparing depth {current_depth + 1})"
                            
                            job_status.message = f'Crawled {len(results)} pages at {depth_info}...'
                            await self._send_progress_update(job_id)
                            
                            # Respect crawl delay
                            await asyncio.sleep(crawl_delay)
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to crawl {current_url}: {str(e)}")
                            continue
                    
                    # Move to next depth
                    current_depth += 1
                    
        except Exception as e:
            raise Exception(f"Deep crawl failed: {str(e)}")
    
    async def _crawl_sitemap(self, job_id: str, config: Dict[str, Any]):
        """Crawl URLs from sitemap"""
        job_status = self.active_jobs[job_id]
        
        url = config['url']
        sitemap_url = config.get('sitemap_url') or urljoin(url, '/sitemap.xml')
        max_urls = int(config.get('sitemap_max', 50))
        
        try:
            # Fetch and parse sitemap
            import requests
            import xml.etree.ElementTree as ET
            
            job_status.message = f'Fetching sitemap from {sitemap_url}...'
            await self._send_progress_update(job_id)
            
            response = requests.get(sitemap_url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            # Extract URLs from sitemap (try different sitemap formats)
            urls = []
            
            # Standard sitemap format
            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc is not None and loc.text:
                    urls.append(loc.text)
                    if len(urls) >= max_urls:
                        break
            
            # Try sitemap index format (contains references to other sitemaps)
            if not urls:
                for sitemap_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                    loc = sitemap_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc is not None and loc.text:
                        try:
                            # Fetch the referenced sitemap
                            sub_response = requests.get(loc.text, timeout=30)
                            sub_response.raise_for_status()
                            sub_root = ET.fromstring(sub_response.content)
                            
                            for url_elem in sub_root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                                sub_loc = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                                if sub_loc is not None and sub_loc.text:
                                    urls.append(sub_loc.text)
                                    if len(urls) >= max_urls:
                                        break
                            
                            if len(urls) >= max_urls:
                                break
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to fetch sub-sitemap {loc.text}: {str(e)}")
                            continue
            
            # Try plain text sitemap format
            if not urls:
                lines = response.text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and self.is_valid_url(line):
                        urls.append(line)
                        if len(urls) >= max_urls:
                            break
            
            if not urls:
                # Fallback: Try common sitemap locations
                common_sitemap_paths = [
                    '/sitemap.xml',
                    '/sitemap_index.xml',
                    '/sitemap.txt',
                    '/robots.txt'  # Sometimes contains sitemap references
                ]
                
                base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                
                for path in common_sitemap_paths:
                    try:
                        fallback_url = urljoin(base_url, path)
                        if fallback_url == sitemap_url:  # Skip if we already tried this
                            continue
                            
                        job_status.message = f'Trying fallback sitemap at {fallback_url}...'
                        await self._send_progress_update(job_id)
                        
                        fallback_response = requests.get(fallback_url, timeout=30)
                        fallback_response.raise_for_status()
                        
                        if path == '/robots.txt':
                            # Look for sitemap references in robots.txt
                            for line in fallback_response.text.split('\n'):
                                if line.strip().lower().startswith('sitemap:'):
                                    sitemap_ref = line.split(':', 1)[1].strip()
                                    if self.is_valid_url(sitemap_ref):
                                        return await self._crawl_sitemap_url(job_id, config, sitemap_ref)
                        else:
                            # Try to parse as XML sitemap
                            try:
                                fallback_root = ET.fromstring(fallback_response.content)
                                for url_elem in fallback_root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                                    loc = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                                    if loc is not None and loc.text:
                                        urls.append(loc.text)
                                        if len(urls) >= max_urls:
                                            break
                                
                                if urls:
                                    break
                            except:
                                # Try as plain text
                                for line in fallback_response.text.strip().split('\n'):
                                    line = line.strip()
                                    if line and self.is_valid_url(line):
                                        urls.append(line)
                                        if len(urls) >= max_urls:
                                            break
                                
                                if urls:
                                    break
                    except Exception as e:
                        self.logger.debug(f"Fallback sitemap {fallback_url} failed: {str(e)}")
                        continue
            
            if not urls:
                raise Exception(f"No URLs found in sitemap at {sitemap_url} or common fallback locations. The website may not have a sitemap or it may be in an unsupported format.")
            
            job_status.total = len(urls)
            job_status.message = f'Found {len(urls)} URLs in sitemap, starting to crawl...'
            await self._send_progress_update(job_id)
            
            # Crawl each URL
            from scraper import WebScraper
            scraper = WebScraper()
            
            for i, page_url in enumerate(urls):
                if job_status.status == 'cancelled':
                    break
                    
                try:
                    result = await scraper.scrape_url(page_url)
                    job_status.results.append(result)
                    job_status.completed = i + 1
                    job_status.message = f'Scraped {i + 1}/{len(urls)} pages from sitemap...'
                    await self._send_progress_update(job_id)
                    
                    # Add delay between requests
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to scrape {page_url}: {str(e)}")
                    continue
                    
        except Exception as e:
            raise Exception(f"Sitemap crawl failed: {str(e)}")
    
    async def _crawl_pattern(self, job_id: str, config: Dict[str, Any]):
        """Crawl URLs matching specific patterns"""
        job_status = self.active_jobs[job_id]
        
        url = config['url']
        url_pattern = config.get('url_pattern', '.*')
        exclude_pattern = config.get('exclude_pattern', '')
        
        job_status.message = f'Starting pattern-based crawl...'
        
        try:
            # This is a simplified pattern crawl - in practice, you'd want to
            # discover URLs through crawling and then filter by pattern
            import re
            
            pattern_filter = URLPatternFilter(
                patterns=[url_pattern],
                exclude_patterns=[exclude_pattern] if exclude_pattern else []
            )
            
            # For now, just crawl the starting page and immediate links
            from scraper import WebScraper
            scraper = WebScraper()
            
            # Get the starting page
            result = await scraper.scrape_url(url)
            if result['success']:
                job_status.results.append(result)
                
                # Extract and filter links
                links_to_crawl = []
                for link in result.get('links', []):
                    link_url = link.get('href', '')
                    if link_url and self.is_valid_url(link_url):
                        if re.match(url_pattern, link_url):
                            if not exclude_pattern or not re.match(exclude_pattern, link_url):
                                links_to_crawl.append(link_url)
                
                job_status.total = len(links_to_crawl) + 1  # +1 for the starting page
                job_status.completed = 1
                job_status.message = f'Found {len(links_to_crawl)} matching links, crawling...'
                
                # Crawl matching links
                for i, link_url in enumerate(links_to_crawl[:20]):  # Limit to 20 links
                    if job_status.status == 'cancelled':
                        break
                        
                    try:
                        link_result = await scraper.scrape_url(link_url)
                        job_status.results.append(link_result)
                        job_status.completed = i + 2
                        job_status.message = f'Scraped {i + 2}/{job_status.total} matching pages...'
                        
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to scrape {link_url}: {str(e)}")
                        continue
            
        except Exception as e:
            raise Exception(f"Pattern crawl failed: {str(e)}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a crawl job"""
        if job_id not in self.active_jobs:
            return None
            
        job_status = self.active_jobs[job_id]
        return {
            "job_id": job_status.job_id,
            "status": job_status.status,
            "message": job_status.message,
            "completed": job_status.completed,
            "total": job_status.total,
            "error": job_status.error,
            "started_at": job_status.started_at,
            "completed_at": job_status.completed_at
        }
    
    def get_job_results(self, job_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get results of a completed crawl job"""
        if job_id not in self.active_jobs:
            return None
            
        return self.active_jobs[job_id].results
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running crawl job"""
        if job_id not in self.active_jobs:
            return False
            
        job_status = self.active_jobs[job_id]
        if job_status.status == 'running':
            job_status.status = 'cancelled'
            job_status.message = 'Crawl cancelled by user'
            job_status.completed_at = time.time()
            return True
            
        return False
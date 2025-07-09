import asyncio
import logging
import random
import re
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, PruningContentFilter, DefaultMarkdownGenerator
from crawl4ai.extraction_strategy import NoExtractionStrategy


class YouTubeScrollHandler:
    """YouTube-specific infinite scroll handler with anti-bot countermeasures"""
    
    def __init__(self, logger):
        self.logger = logger
        
    def detect_youtube_url(self, url: str) -> bool:
        """Detect if URL is a YouTube page that benefits from specialized handling"""
        youtube_patterns = [
            r'youtube\.com\/@[\w-]+/videos',
            r'youtube\.com\/c\/[\w-]+/videos', 
            r'youtube\.com\/channel\/[\w-]+/videos',
            r'youtube\.com\/user\/[\w-]+/videos',
            r'youtube\.com\/results\?search_query=',
            r'youtube\.com\/playlist\?list='
        ]
        return any(re.search(pattern, url) for pattern in youtube_patterns)
    
    def get_youtube_scroll_js(self, options: Dict[str, Any]) -> str:
        """Generate YouTube-optimized scroll JavaScript with human behavior simulation"""
        max_scrolls = options.get("max_scrolls", 20)  # Increased for YouTube
        base_delay = options.get("scroll_delay", 3000)  # Increased base delay
        scroll_step = options.get("scroll_step", 800)  # Smaller steps for more natural behavior
        content_stability_checks = options.get("content_stability_checks", 5)  # More checks
        youtube_optimized = options.get("youtube_optimized", True)
        human_behavior = options.get("human_behavior_simulation", True)
        
        return f"""
        (async function youtubeInfiniteScroll() {{
            console.log('YouTube Infinite Scroll Handler Started');
            
            let scrollCount = 0;
            let stableCount = 0;
            let totalVideosFound = 0;
            let lastVideoCount = 0;
            
            const config = {{
                maxScrolls: {max_scrolls},
                baseDelay: {base_delay},
                scrollStep: {scroll_step},
                stabilityChecks: {content_stability_checks},
                youtubeOptimized: {str(youtube_optimized).lower()},
                humanBehavior: {str(human_behavior).lower()}
            }};
            
            // YouTube-specific selectors
            const selectors = {{
                videoContainer: '#contents ytd-rich-grid-renderer',
                videoItems: 'ytd-rich-item-renderer, ytd-video-renderer',
                loadingSpinner: 'ytd-continuation-item-renderer',
                gridContainer: 'ytd-rich-grid-renderer #contents',
                pageManager: 'ytd-page-manager'
            }};
            
            function getRandomDelay(baseMs, variationPercent = 30) {{
                if (!config.humanBehavior) return baseMs;
                const variation = baseMs * (variationPercent / 100);
                return baseMs + (Math.random() * variation * 2 - variation);
            }}
            
            function getRandomScrollAmount() {{
                if (!config.humanBehavior) return config.scrollStep;
                return config.scrollStep + (Math.random() * 200 - 100);
            }}
            
            function sleep(ms) {{
                return new Promise(resolve => setTimeout(resolve, ms));
            }}
            
            function getYouTubeVideoCount() {{
                const containers = document.querySelectorAll(selectors.videoItems);
                return containers.length;
            }}
            
            function getContentHeight() {{
                const gridContainer = document.querySelector(selectors.gridContainer);
                if (gridContainer) {{
                    return gridContainer.scrollHeight;
                }}
                return Math.max(
                    document.body.scrollHeight,
                    document.documentElement.scrollHeight
                );
            }}
            
            function isLoadingVisible() {{
                const spinner = document.querySelector(selectors.loadingSpinner);
                return spinner && spinner.offsetParent !== null;
            }}
            
            function simulateHumanScroll(distance) {{
                if (!config.humanBehavior) {{
                    window.scrollBy(0, distance);
                    return;
                }}
                
                // Simulate gradual scrolling
                const chunks = 3 + Math.floor(Math.random() * 3);
                const chunkSize = distance / chunks;
                let currentChunk = 0;
                
                function scrollChunk() {{
                    if (currentChunk < chunks) {{
                        window.scrollBy(0, chunkSize);
                        currentChunk++;
                        setTimeout(scrollChunk, 50 + Math.random() * 100);
                    }}
                }}
                scrollChunk();
            }}
            
            function isNearBottom() {{
                const threshold = window.innerHeight * 1.5;
                return window.innerHeight + window.scrollY >= document.body.offsetHeight - threshold;
            }}
            
            // Wait for YouTube to initialize
            console.log('Waiting for YouTube page to initialize...');
            let initAttempts = 0;
            while (initAttempts < 10 && !document.querySelector(selectors.videoContainer)) {{
                await sleep(1000);
                initAttempts++;
            }}
            
            if (!document.querySelector(selectors.videoContainer)) {{
                console.log('YouTube container not found, trying alternative selectors...');
                
                // Try alternative YouTube selectors
                const altSelectors = [
                    'ytd-grid-renderer #contents',
                    '#contents ytd-video-renderer',
                    'ytd-section-list-renderer #contents',
                    '#primary ytd-rich-grid-renderer'
                ];
                
                let foundAltContainer = false;
                for (const altSelector of altSelectors) {{
                    if (document.querySelector(altSelector)) {{
                        console.log(`Found alternative container: ${{altSelector}}`);
                        selectors.videoContainer = altSelector;
                        selectors.gridContainer = altSelector;
                        foundAltContainer = true;
                        break;
                    }}
                }}
                
                if (!foundAltContainer) {{
                    console.log('No YouTube containers found, falling back to generic scroll');
                    return {{
                        success: false,
                        reason: 'YouTube container not detected',
                        videosFound: 0,
                        fallbackRecommended: true
                    }};
                }}
            }}
            
            console.log('YouTube container found, starting optimized scroll...');
            
            // Initial video count
            totalVideosFound = getYouTubeVideoCount();
            lastVideoCount = totalVideosFound;
            console.log(`Initial videos found: ${{totalVideosFound}}`);
            
            // Main scrolling loop
            while (scrollCount < config.maxScrolls) {{
                const currentHeight = getContentHeight();
                const currentVideoCount = getYouTubeVideoCount();
                
                // Log progress
                if (currentVideoCount > lastVideoCount) {{
                    console.log(`Videos loaded: ${{currentVideoCount}} (new: ${{currentVideoCount - lastVideoCount}})`);
                    totalVideosFound = currentVideoCount;
                    lastVideoCount = currentVideoCount;
                    stableCount = 0; // Reset stability counter when new content loads
                }}
                
                // Check if we've reached the end
                if (isNearBottom() && !isLoadingVisible()) {{
                    console.log('Reached bottom of content');
                    break;
                }}
                
                // Perform scroll with human-like behavior
                const scrollAmount = getRandomScrollAmount();
                simulateHumanScroll(scrollAmount);
                scrollCount++;
                
                console.log(`Scroll ${{scrollCount}}/${{config.maxScrolls}} - Videos: ${{currentVideoCount}}`);
                
                // Wait with randomized delay
                const delay = getRandomDelay(config.baseDelay);
                await sleep(delay);
                
                // Additional wait if loading spinner is visible
                if (isLoadingVisible()) {{
                    console.log('Loading spinner detected, waiting longer...');
                    await sleep(getRandomDelay(2000));
                }}
                
                // Check content stability
                const newHeight = getContentHeight();
                const newVideoCount = getYouTubeVideoCount();
                
                if (newHeight === currentHeight && newVideoCount === currentVideoCount) {{
                    stableCount++;
                    console.log(`Content stable count: ${{stableCount}}/${{config.stabilityChecks}}`);
                    
                    if (stableCount >= config.stabilityChecks) {{
                        console.log('Content appears to be fully loaded');
                        break;
                    }}
                }} else {{
                    stableCount = 0;
                }}
                
                // Random small pause to avoid detection
                if (config.humanBehavior && Math.random() < 0.3) {{
                    await sleep(getRandomDelay(500, 50));
                }}
            }}
            
            // Scroll back to top smoothly
            if (config.humanBehavior) {{
                console.log('Smoothly scrolling back to top...');
                const scrollTop = () => {{
                    const currentScroll = window.pageYOffset;
                    if (currentScroll > 0) {{
                        window.scrollTo(0, currentScroll - currentScroll * 0.1);
                        setTimeout(scrollTop, 16);
                    }}
                }};
                scrollTop();
            }} else {{
                window.scrollTo(0, 0);
            }}
            
            // Final wait for content to stabilize
            await sleep(2000);
            
            const finalVideoCount = getYouTubeVideoCount();
            console.log(`YouTube scroll completed. Final videos: ${{finalVideoCount}}, Total scrolls: ${{scrollCount}}`);
            
            return {{
                success: true,
                totalScrolls: scrollCount,
                videosFound: finalVideoCount,
                finalHeight: getContentHeight(),
                completed: true
            }};
        }})();
        """
        
    def get_generic_scroll_js(self, options: Dict[str, Any]) -> str:
        """Fallback generic scroll handler for non-YouTube sites"""
        max_scrolls = options.get("max_scrolls", 10)
        scroll_delay = options.get("scroll_delay", 2000)
        scroll_step = options.get("scroll_step", 1000)
        content_stability_checks = options.get("content_stability_checks", 3)
        human_behavior = options.get("human_behavior_simulation", True)
        
        return f"""
        (async function genericInfiniteScroll() {{
            let scrollCount = 0;
            let stableCount = 0;
            let maxScrolls = {max_scrolls};
            let scrollDelay = {scroll_delay};
            let scrollStep = {scroll_step};
            let stabilityChecks = {content_stability_checks};
            let humanBehavior = {str(human_behavior).lower()};
            
            function getRandomDelay(baseMs) {{
                if (!humanBehavior) return baseMs;
                return baseMs + (Math.random() * 1000 - 500);
            }}
            
            function getContentHeight() {{
                return Math.max(
                    document.body.scrollHeight,
                    document.body.offsetHeight,
                    document.documentElement.clientHeight,
                    document.documentElement.scrollHeight,
                    document.documentElement.offsetHeight
                );
            }}
            
            function isAtBottom() {{
                return window.innerHeight + window.scrollY >= document.body.offsetHeight - 100;
            }}
            
            function sleep(ms) {{
                return new Promise(resolve => setTimeout(resolve, ms));
            }}
            
            let initialHeight = getContentHeight();
            
            while (scrollCount < maxScrolls && !isAtBottom()) {{
                let currentHeight = getContentHeight();
                
                // Scroll down by step amount
                window.scrollBy(0, scrollStep);
                scrollCount++;
                
                // Wait for content to load with randomization
                const delay = getRandomDelay(scrollDelay);
                await sleep(delay);
                
                let newHeight = getContentHeight();
                
                // Check if content has changed
                if (newHeight > currentHeight) {{
                    stableCount = 0;
                }} else {{
                    stableCount++;
                    if (stableCount >= stabilityChecks) {{
                        break;
                    }}
                }}
            }}
            
            // Scroll back to top for final content extraction
            window.scrollTo(0, 0);
            
            // Wait a bit for any final content to stabilize
            await sleep(1000);
            
            return {{
                totalScrolls: scrollCount,
                finalHeight: getContentHeight(),
                completed: true
            }};
        }})();
        """


class WebScraper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.youtube_handler = YouTubeScrollHandler(self.logger)
        
    def is_valid_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    

    async def scrape_url(self, url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")
        
        options = options or {}
        enable_infinite_scroll = options.get("enable_infinite_scroll", False)
        
        try:
            browser_config = BrowserConfig(
                headless=True,
                browser_type="chromium"
            )
            
            # Detect if this is a YouTube URL that benefits from specialized handling
            is_youtube = self.youtube_handler.detect_youtube_url(url)
            
            # Configure crawler with extended timeout if infinite scroll is enabled
            base_timeout = 90000 if (enable_infinite_scroll and is_youtube) else 60000 if enable_infinite_scroll else 30000
            page_timeout = options.get("page_timeout", base_timeout)
            
            base_wait = "8000" if (enable_infinite_scroll and is_youtube) else "5000" if enable_infinite_scroll else "3000"
            wait_for = options.get("wait_for", base_wait)
            
            # Set YouTube-optimized defaults if detected
            if is_youtube and enable_infinite_scroll:
                options.setdefault("max_scrolls", 25)
                options.setdefault("scroll_delay", 3500)
                options.setdefault("scroll_step", 700)
                options.setdefault("content_stability_checks", 5)
                options.setdefault("youtube_optimized", True)
                options.setdefault("human_behavior_simulation", True)
                self.logger.info(f"🎯 YouTube URL detected: {url}")
                self.logger.info(f"📋 Optimized settings applied: max_scrolls={options['max_scrolls']}, delay={options['scroll_delay']}ms, step={options['scroll_step']}px")
                self.logger.info(f"🤖 Human behavior simulation: {options['human_behavior_simulation']}")
            
            # Prepare JavaScript code for infinite scroll
            js_code = []
            scroll_strategy = "none"
            
            if enable_infinite_scroll:
                if is_youtube:
                    scroll_js = self.youtube_handler.get_youtube_scroll_js(options)
                    scroll_strategy = "youtube_optimized"
                    self.logger.info("🎬 Using YouTube-optimized scroll handler with specialized DOM selectors")
                else:
                    scroll_js = self.youtube_handler.get_generic_scroll_js(options)
                    scroll_strategy = "generic"
                    self.logger.info("📜 Using generic scroll handler for non-YouTube site")
                
                js_code.append(scroll_js)
            
            crawler_config = CrawlerRunConfig(
                word_count_threshold=options.get("word_count_threshold", 10),
                excluded_tags=options.get("excluded_tags", ["nav", "footer", "aside", "script", "style"]),
                remove_overlay_elements=True,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=options.get("content_filter_threshold", 0.48)
                    )
                ),
                css_selector=options.get("css_selector"),
                wait_for=wait_for,
                page_timeout=page_timeout,
                js_code=js_code if js_code else options.get("js_code", [])
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=crawler_config)
                
                # Handle fallback for YouTube URLs if initial attempt fails
                if not result.success and is_youtube and enable_infinite_scroll:
                    self.logger.warning(f"⚠️  YouTube-optimized scrolling failed: {result.error_message}")
                    self.logger.info("🔄 Attempting fallback with generic scroll handler...")
                    
                    # Try with generic scroll handler as fallback
                    fallback_js = self.youtube_handler.get_generic_scroll_js(options)
                    fallback_config = CrawlerRunConfig(
                        word_count_threshold=options.get("word_count_threshold", 10),
                        excluded_tags=options.get("excluded_tags", ["nav", "footer", "aside", "script", "style"]),
                        remove_overlay_elements=True,
                        markdown_generator=DefaultMarkdownGenerator(
                            content_filter=PruningContentFilter(
                                threshold=options.get("content_filter_threshold", 0.48)
                            )
                        ),
                        css_selector=options.get("css_selector"),
                        wait_for=wait_for,
                        page_timeout=page_timeout,
                        js_code=[fallback_js]
                    )
                    
                    result = await crawler.arun(url=url, config=fallback_config)
                    scroll_strategy = "fallback_generic"
                    
                    if result.success:
                        self.logger.info("✅ Fallback generic scroll succeeded")
                    else:
                        self.logger.warning(f"❌ Fallback also failed: {result.error_message}")
                
                if not result.success:
                    raise Exception(f"Scraping failed: {result.error_message}")
                
                # Log successful completion
                word_count = len(result.markdown.split()) if result.markdown else 0
                if enable_infinite_scroll:
                    self.logger.info(f"🎉 Scraping completed successfully! Strategy: {scroll_strategy}, Words: {word_count}")
                    if is_youtube:
                        self.logger.info(f"📺 YouTube content extracted with enhanced scroll behavior")
                else:
                    self.logger.info(f"✅ Standard scraping completed, Words: {word_count}")
                
                return {
                    "success": True,
                    "url": url,
                    "title": result.metadata.get("title", ""),
                    "description": result.metadata.get("description", ""),
                    "markdown": result.markdown,
                    "links": result.links.get("external", []) + result.links.get("internal", []) if hasattr(result, 'links') and result.links else [],
                    "images": [],  # Images not available in this version
                    "metadata": result.metadata,
                    "word_count": len(result.markdown.split()) if result.markdown else 0,
                    "infinite_scroll_enabled": enable_infinite_scroll,
                    "youtube_optimized": is_youtube if enable_infinite_scroll else False,
                    "human_behavior_simulation": options.get("human_behavior_simulation", False) if enable_infinite_scroll else False,
                    "scroll_strategy": scroll_strategy if enable_infinite_scroll else "none"
                }
                
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            is_youtube = self.youtube_handler.detect_youtube_url(url)
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "markdown": None,
                "links": [],
                "images": [],
                "metadata": {},
                "word_count": 0,
                "infinite_scroll_enabled": enable_infinite_scroll,
                "youtube_optimized": is_youtube if enable_infinite_scroll else False,
                "human_behavior_simulation": options.get("human_behavior_simulation", False) if enable_infinite_scroll else False,
                "scroll_strategy": "error"
            }
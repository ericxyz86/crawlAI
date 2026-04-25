# Restore Advanced Web Crawler UI - Instructions

## What Happened

The advanced web crawler UI was accidentally overwritten when the repository was reset to `origin/main`.

**Evidence:**
- Git reflog shows commit `f7dadcc` with message "new" contained the advanced UI
- This commit was reset/overwritten: `HEAD@{0}: reset: moving to origin/main`
- The advanced UI had **1,111 lines** with extensive features
- Current UI has only **258 lines** with basic functionality

## Current State vs. Advanced UI

### Current UI (258 lines):
- Simple form with 3 fields: Company Name, Objective, LLM Model
- Basic submit button
- Simple results display

### Advanced UI (1,111 lines) - Lost but recoverable:
- **4 Crawl Modes:**
  - 🔹 Single Page (with infinite scroll support)
  - 🕷️ Deep Crawl (with depth & page limits)
  - 🗺️ Sitemap Crawl
  - 🎯 Pattern-Based crawling

- **Deep Crawl Features:**
  - Max Depth slider (1-5 levels, default: 2)
  - Max Pages slider (5-100 pages, default: 20)
  - Crawl Delay (1-10 seconds, default: 2)
  - Same domain restriction toggle

- **Infinite Scroll Features:**
  - Max Scrolls control (3-50)
  - Scroll Delay (500-5000ms)
  - Scroll Step pixels
  - Content Stability Checks
  - YouTube Optimization
  - Human Behavior Simulation

- **Professional Design:**
  - Purple gradient theme
  - Collapsible advanced options accordion
  - Real-time slider value displays
  - Progress tracking with live updates
  - Speed monitoring (pages/min)
  - Elegant animations and transitions

---

## Step 1: Recover the Advanced UI File

### In the crawlAI project directory:

```bash
# 1. Check that you're in the right directory
pwd  # Should show: /Users/ep/Desktop/crawlAI

# 2. Verify the commit exists
git reflog | head -10
# You should see: f7dadcc HEAD@{1}: commit: new

# 3. Extract the advanced UI from the lost commit
git show f7dadcc:templates/index.html > templates/index_advanced.html

# 4. Verify the file was extracted (should be 1,111 lines)
wc -l templates/index_advanced.html

# 5. Backup current simple UI (just in case)
cp templates/index.html templates/index_simple_backup.html

# 6. Replace with advanced UI
cp templates/index_advanced.html templates/index.html

# 7. Verify the replacement
wc -l templates/index.html  # Should show 1111
```

---

## Step 2: Update Backend to Support Advanced Features

The advanced UI sends these additional parameters that the backend needs to handle:

### Required Backend Parameters:

**Crawl Mode Parameters:**
- `crawl_mode` (string): "single", "deep", "sitemap", or "pattern"

**Deep Crawl Parameters:**
- `max_depth` (int): 1-5, default 2
- `max_pages` (int): 5-100, default 20
- `crawl_delay` (int): 1-10 seconds, default 2
- `same_domain` (bool): default true

**Single Page / Infinite Scroll Parameters:**
- `enable_infinite_scroll` (bool): default false
- `max_scrolls` (int): 3-50, default 10
- `scroll_delay` (int): 500-5000ms, default 2000
- `scroll_step` (int): 500-2000px, default 1000
- `content_stability_checks` (int): 1-10, default 3
- `youtube_optimized` (bool): default true
- `human_behavior_simulation` (bool): default true

**Sitemap Parameters:**
- `sitemap_url` (string): optional custom sitemap URL

**Pattern Parameters:**
- `url_pattern` (string): regex pattern for URLs
- `exclude_pattern` (string): regex pattern to exclude

### Backend Files to Update:

#### 1. Update `app.py` - Flask Route Handler

Location: `/crawl` endpoint

**Add parameter extraction:**

```python
@app.route("/crawl", methods=["POST"])
def crawl():
    """API endpoint to handle crawling requests."""
    try:
        # Lazy import to avoid blocking on app startup
        from improved_web_crawler import WebCrawler

        data = request.get_json()

        # Existing parameters
        if not data or "company_name" not in data:
            logger.error('Missing required field "company_name"')
            return jsonify({"error": 'Missing required field "company_name"'}), 400

        entity_name = data["company_name"]
        objective = data.get("objective", "")
        llm = data.get("llm", "R1")

        # NEW: Extract advanced crawl parameters
        crawl_config = {
            "crawl_mode": data.get("crawl_mode", "single"),
            "max_depth": int(data.get("max_depth", 2)),
            "max_pages": int(data.get("max_pages", 20)),
            "crawl_delay": int(data.get("crawl_delay", 2)),
            "same_domain": data.get("same_domain", True),
            "enable_infinite_scroll": data.get("enable_infinite_scroll", False),
            "max_scrolls": int(data.get("max_scrolls", 10)),
            "scroll_delay": int(data.get("scroll_delay", 2000)),
            "scroll_step": int(data.get("scroll_step", 1000)),
            "content_stability_checks": int(data.get("content_stability_checks", 3)),
            "youtube_optimized": data.get("youtube_optimized", True),
            "human_behavior_simulation": data.get("human_behavior_simulation", True),
            "sitemap_url": data.get("sitemap_url", ""),
            "url_pattern": data.get("url_pattern", ""),
            "exclude_pattern": data.get("exclude_pattern", "")
        }

        logger.info(f"Received request to crawl: {entity_name} with mode: {crawl_config['crawl_mode']}")
        logger.info(f"Crawl config: {crawl_config}")

        # Initialize and use the improved WebCrawler with config
        crawler = WebCrawler()
        result = crawler.crawl_website(entity_name, objective, llm, crawl_config=crawl_config)

        # ... rest of existing code
```

#### 2. Update `improved_web_crawler.py` - WebCrawler Class

**A. Update `crawl_website` method signature:**

```python
def crawl_website(self, entity_name: str, objective: str = None, llm: str = "R1", crawl_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Crawl a website based on company name or a specific URL with advanced configuration.

    Args:
        entity_name (str): The name of the company/entity or a specific URL to crawl.
        objective (str, optional): The objective for the crawl. Defaults to None.
        llm (str, optional): The language model to use. Defaults to "R1".
        crawl_config (dict, optional): Advanced crawl configuration options. Defaults to None.
            Available config options:
            - crawl_mode: "single", "deep", "sitemap", "pattern"
            - max_depth: int (1-5)
            - max_pages: int (5-100)
            - crawl_delay: int (1-10 seconds)
            - same_domain: bool
            - enable_infinite_scroll: bool
            - max_scrolls: int (3-50)
            - scroll_delay: int (500-5000ms)
            - scroll_step: int (500-2000px)
            - content_stability_checks: int (1-10)
            - youtube_optimized: bool
            - human_behavior_simulation: bool
            - sitemap_url: str
            - url_pattern: str (regex)
            - exclude_pattern: str (regex)

    Returns:
        dict: A dictionary containing crawled URLs and extracted data, or an error message.
    """
    try:
        # Set default config if not provided
        if crawl_config is None:
            crawl_config = {
                "crawl_mode": "single",
                "max_depth": 2,
                "max_pages": 20,
                "crawl_delay": 2,
                "same_domain": True,
                "enable_infinite_scroll": False,
                "max_scrolls": 10,
                "scroll_delay": 2000,
                "scroll_step": 1000,
                "content_stability_checks": 3,
                "youtube_optimized": True,
                "human_behavior_simulation": True,
                "sitemap_url": "",
                "url_pattern": "",
                "exclude_pattern": ""
            }

        crawl_mode = crawl_config.get("crawl_mode", "single")
        logger.info(f"Starting crawl with mode: {crawl_mode}")
        ColoredLogger.info(f"Crawl Mode: {crawl_mode}")

        # Route to appropriate crawl method based on mode
        if crawl_mode == "single":
            return self._crawl_single_page(entity_name, objective, llm, crawl_config)
        elif crawl_mode == "deep":
            return self._crawl_deep(entity_name, objective, llm, crawl_config)
        elif crawl_mode == "sitemap":
            return self._crawl_sitemap(entity_name, objective, llm, crawl_config)
        elif crawl_mode == "pattern":
            return self._crawl_pattern_based(entity_name, objective, llm, crawl_config)
        else:
            logger.error(f"Unknown crawl mode: {crawl_mode}")
            return {"error": f"Unknown crawl mode: {crawl_mode}"}

    except Exception as e:
        logger.error(f"Error in crawl_website: {str(e)}")
        return {"error": str(e)}
```

**B. Implement crawl mode methods:**

```python
def _crawl_single_page(self, url: str, objective: str, llm: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crawl a single page with optional infinite scroll support.
    """
    try:
        # Existing single-page logic, but add infinite scroll if enabled
        if config.get("enable_infinite_scroll", False):
            return self._crawl_with_infinite_scroll(url, objective, llm, config)
        else:
            # Use existing single page scraping logic
            # This is likely your current implementation
            pass

    except Exception as e:
        logger.error(f"Error in single page crawl: {str(e)}")
        return {"error": str(e)}

def _crawl_deep(self, url: str, objective: str, llm: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform deep crawl following links up to specified depth.
    """
    try:
        max_depth = config.get("max_depth", 2)
        max_pages = config.get("max_pages", 20)
        crawl_delay = config.get("crawl_delay", 2)
        same_domain = config.get("same_domain", True)

        logger.info(f"Deep crawl: depth={max_depth}, max_pages={max_pages}, delay={crawl_delay}s")

        # Implement breadth-first or depth-first search
        # Track visited URLs, current depth
        # Respect max_pages and max_depth limits
        # Add crawl_delay between requests
        # Filter by same_domain if enabled

        # Pseudo-code structure:
        # visited = set()
        # queue = [(url, 0)]  # (url, depth)
        # pages_scraped = 0
        #
        # while queue and pages_scraped < max_pages:
        #     current_url, depth = queue.pop(0)
        #     if depth > max_depth or current_url in visited:
        #         continue
        #
        #     # Scrape page
        #     result = self.scrape_page(current_url)
        #     visited.add(current_url)
        #     pages_scraped += 1
        #
        #     # Extract and queue links if not at max depth
        #     if depth < max_depth:
        #         links = self.extract_links(result, same_domain, current_url)
        #         queue.extend([(link, depth + 1) for link in links])
        #
        #     time.sleep(crawl_delay)

        pass

    except Exception as e:
        logger.error(f"Error in deep crawl: {str(e)}")
        return {"error": str(e)}

def _crawl_sitemap(self, url: str, objective: str, llm: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crawl using sitemap.xml to discover URLs.
    """
    try:
        sitemap_url = config.get("sitemap_url", "")

        # If custom sitemap URL provided, use it
        # Otherwise try common locations: /sitemap.xml, /sitemap_index.xml, etc.

        # Parse sitemap XML
        # Extract all <loc> URLs
        # Optionally filter by max_pages
        # Scrape each URL with crawl_delay

        pass

    except Exception as e:
        logger.error(f"Error in sitemap crawl: {str(e)}")
        return {"error": str(e)}

def _crawl_pattern_based(self, url: str, objective: str, llm: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crawl following URLs that match specified patterns.
    """
    try:
        url_pattern = config.get("url_pattern", "")
        exclude_pattern = config.get("exclude_pattern", "")

        # Similar to deep crawl but filter URLs by regex patterns
        # Include URLs matching url_pattern
        # Exclude URLs matching exclude_pattern

        pass

    except Exception as e:
        logger.error(f"Error in pattern-based crawl: {str(e)}")
        return {"error": str(e)}

def _crawl_with_infinite_scroll(self, url: str, objective: str, llm: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scrape single page with infinite scroll support using Playwright.
    """
    try:
        max_scrolls = config.get("max_scrolls", 10)
        scroll_delay = config.get("scroll_delay", 2000)
        scroll_step = config.get("scroll_step", 1000)
        stability_checks = config.get("content_stability_checks", 3)
        youtube_optimized = config.get("youtube_optimized", True)
        human_behavior = config.get("human_behavior_simulation", True)

        logger.info(f"Infinite scroll: max_scrolls={max_scrolls}, delay={scroll_delay}ms")

        # Use Playwright to scroll and load content
        # Check for content stability (no new content after N checks)
        # Apply YouTube-specific selectors if youtube_optimized
        # Randomize timing if human_behavior enabled

        # Pseudo-code:
        # page = await browser.new_page(url)
        # previous_height = await page.evaluate("document.body.scrollHeight")
        # no_change_count = 0
        #
        # for i in range(max_scrolls):
        #     if human_behavior:
        #         delay = scroll_delay + random.randint(-200, 200)
        #     else:
        #         delay = scroll_delay
        #
        #     await page.evaluate(f"window.scrollBy(0, {scroll_step})")
        #     await page.wait_for_timeout(delay)
        #
        #     current_height = await page.evaluate("document.body.scrollHeight")
        #     if current_height == previous_height:
        #         no_change_count += 1
        #         if no_change_count >= stability_checks:
        #             break
        #     else:
        #         no_change_count = 0
        #
        #     previous_height = current_height

        pass

    except Exception as e:
        logger.error(f"Error in infinite scroll: {str(e)}")
        return {"error": str(e)}
```

---

## Step 3: Testing the Restored UI

### 1. Start the Application

```bash
# In crawlAI directory
source venv/bin/activate
python app.py
```

### 2. Open Browser

Navigate to: `http://localhost:5002`

### 3. Test Each Mode

**Test Single Page Mode:**
- Enter URL: `https://example.com`
- Select "Single Page" mode
- Try with/without infinite scroll
- Verify it works

**Test Deep Crawl Mode:**
- Enter URL: `https://example.com`
- Select "Deep Crawl" mode
- Adjust max_depth slider (try 2)
- Adjust max_pages slider (try 20)
- Click "Start Crawling"
- Check backend logs for depth/pages parameters

**Test Sitemap Mode:**
- Enter URL: `https://example.com`
- Select "Sitemap Crawl" mode
- Should auto-discover sitemap.xml

**Test Pattern Mode:**
- Enter URL: `https://example.com`
- Select "Pattern-Based" mode
- Enter URL pattern (e.g., `/blog/.*`)

---

## Step 4: Optional Enhancements

### Add Real-Time Progress Tracking

The advanced UI has progress tracking features that require WebSocket or polling:

**Frontend expects these updates:**
- `progressPages`: "Pages: 5/20"
- `progressSpeed`: "Speed: 3.2 pages/min"
- Progress bar percentage

**Implementation Options:**

1. **WebSocket** (recommended for real-time updates)
2. **Server-Sent Events (SSE)**
3. **Polling** (simple but less efficient)

**Example with Flask-SocketIO:**

```python
from flask_socketio import SocketIO, emit

socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")

# In crawler, emit progress:
def update_progress(pages_done, total_pages, speed):
    socketio.emit('progress_update', {
        'pages_done': pages_done,
        'total_pages': total_pages,
        'speed': speed
    })
```

---

## Step 5: Commit the Restored UI

```bash
# After verifying everything works

git add templates/index.html
git commit -m "Restore advanced web crawler UI with multi-mode support

Features restored:
- 4 crawl modes: single, deep, sitemap, pattern
- Deep crawl with depth/page limits
- Infinite scroll with advanced options
- Professional gradient UI with sliders
- Real-time progress tracking
- 1,111 lines vs previous 258 lines

Backend support for advanced parameters added."

git push origin main
```

---

## Architecture Notes

### Current Backend Architecture

The project uses:
- **Framework**: Flask (not FastAPI) - see `app.py`
- **Crawler**: `improved_web_crawler.py` with WebCrawler class
- **Browser Automation**: Playwright for dynamic content
- **LLMs**: Deepseek R1 (primary), OpenAI GPT (fallback)
- **Search**: SerpAPI for Google search
- **Scraping**: Firecrawl API + Playwright

### Compatibility Note

The advanced UI was originally built for FastAPI but the current project uses Flask. The HTML/CSS/JavaScript will work fine with Flask, but ensure:

1. Route paths match (`/crawl`, `/download/<filename>`)
2. JSON response format is compatible
3. CORS is configured if needed (already done in current setup)

---

## Troubleshooting

### Issue: UI loads but sliders don't work
**Solution**: Check browser console for JavaScript errors. The slider update functions should be defined in the `<script>` section.

### Issue: Backend receives undefined parameters
**Solution**: Check that form submission in JavaScript includes all config parameters:
```javascript
const formData = {
    company_name: document.getElementById('company').value,
    crawl_mode: document.querySelector('input[name="crawl_mode"]:checked').value,
    max_depth: document.querySelector('input[name="max_depth"]').value,
    // ... etc
};
```

### Issue: Progress tracking doesn't update
**Solution**: Implement WebSocket or SSE as described in Step 4.

### Issue: Deep crawl doesn't respect limits
**Solution**: Verify `_crawl_deep` method implementation has proper loop controls and counters.

---

## Summary

1. ✅ **Recover** advanced UI from commit `f7dadcc`
2. ✅ **Update** backend to accept new parameters
3. ✅ **Implement** crawl mode routing logic
4. ✅ **Add** specific crawl methods for each mode
5. ✅ **Test** each mode thoroughly
6. ✅ **Commit** restored functionality

The advanced UI gives users much more control over crawling behavior and provides a significantly better user experience than the basic 3-field form.

---

## File Locations Reference

```
crawlAI/
├── app.py                      # Flask server - UPDATE THIS
├── improved_web_crawler.py     # WebCrawler class - UPDATE THIS
├── templates/
│   ├── index.html              # REPLACE with advanced UI
│   ├── index_advanced.html     # Recovered file (backup)
│   └── index_simple_backup.html # Current simple UI (backup)
├── requirements.txt
└── .env
```

---

**Created**: 2025-11-12
**Purpose**: Restore advanced web crawler UI accidentally overwritten by git reset
**Status**: Ready for implementation in crawlAI project

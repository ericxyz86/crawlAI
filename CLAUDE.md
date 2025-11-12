# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a Flask-based web crawler application with AI-powered content extraction.

### Core Architecture:
- **Flask Backend** (`app.py`): Main server on port 5001 with rate limiting and security hardening
- **AI-Powered Web Crawler** (`improved_web_crawler.py`): Advanced crawler with multiple LLM support (Deepseek R1 primary, OpenAI o4-mini fallback)
- **Flask Templates UI** (`templates/index.html`): Single-page web interface using Tailwind CSS

### Key Components:
- **WebCrawler**: Main orchestrator class with SearchEngine, URLProcessor, ContentExtractor
- **ConfigManager**: Handles API keys and configuration
- **Results Storage**: JSON files saved to `/results/` directory
- **Security Features**: Input validation, rate limiting, path traversal protection, sanitized error messages

## Development Commands

### Setup:
```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install --with-deps chromium

# Copy environment template and configure API keys
cp .env.example .env
# Edit .env and add your API keys

# Run main application
python app.py  # Main application on port 5001
```

### Testing:
```bash
# Backend testing
python test_flask.py                    # Test Flask server on port 5001
python test_enhanced_crawler.py         # Test enhanced crawler functionality
python test_apple_watch.py              # Test Apple Watch price extraction
python test_multi_site_comprehensive.py # Test multi-site functionality
python simple_app.py                    # Basic test server
python minimal.py                       # Simple HTTP server on port 9000

# Direct crawler testing
python improved_web_crawler.py  # Run crawler CLI interface
python demo_multi_site.py       # Demonstrate multi-site features
```

## Environment Configuration

Required API keys in `.env`:
- `DEEPSEEK_API_KEY`: Primary LLM for content extraction
- `SERP_API_KEY`: Google search functionality
- `FIRECRAWL_API_KEY`: Web scraping service
- `OPENAI_API_KEY`: Fallback LLM (optional)

## Key Architectural Notes

### AI Integration:
- **Primary LLM**: Deepseek R1 for URL selection and content extraction
- **Secondary LLM**: OpenAI GPT models as fallback (configured via OPENAI_API_KEY)
- **Structured Output**: AI generates JSON responses for consistent data extraction
- **Browser Automation**: Uses Playwright for dynamic content rendering

### Enhanced Multi-Site Web Crawling Flow:
1. User submits crawl request via `/crawl` endpoint
2. **Enhanced SearchEngine** performs comprehensive multi-query searches via SerpAPI
   - Generates retailer-specific queries based on region/category detection
   - Searches 3-5 targeted queries in parallel for broader coverage
3. URLProcessor uses AI to select and validate relevant URLs (15-20 sites)
4. **Site Categorization**: URLs classified as official, retailer, dealer, review, etc.
5. **Multi-Site ContentExtractor** scrapes and processes content with AI across categories
6. **Advanced Data Aggregation** with cross-retailer price comparison and analysis
7. Results saved as JSON files with site-categorized data and returned to frontend

### Multi-Site Features:
- **Company Name Searches**: Always trigger multi-site extraction for comprehensive coverage
- **Regional Retailer Databases**: Philippines, US, global retailer knowledge
- **Price Comparison Engine**: Multi-currency support with USD normalization
- **Site Categorization**: official, retailer, dealer, review, news, marketplace, other
- **Backward Compatibility**: Legacy mode available via `legacy_mode: true` parameter

### API Endpoints:
- **GET `/`**: Serves the main web interface
- **POST `/crawl`**: Initiates web crawling (rate limited: 10 requests/hour per IP)
  - Required: `company_name` (max 200 chars)
  - Optional: `objective` (max 500 chars), `llm` (R1 or o4-mini), `legacy_mode` (boolean)
- **GET `/download/<filename>`**: Download saved results (path-traversal protected)
- **GET `/health`**: Health check endpoint

### Security Features:
- **Input Validation**: Comprehensive validation on all user inputs
- **Rate Limiting**: 10 crawls/hour, 50 requests/hour, 200 requests/day per IP
- **Path Traversal Protection**: Filename sanitization and validation
- **Error Handling**: Generic error messages to prevent information disclosure
- **API Key Security**: No API keys logged or exposed

## Development Considerations

- **No Formal Testing Framework**: Current setup uses basic test files, consider adding pytest for Python
- **Production Deployment**: Uses Render.com with `render.yaml` configuration
- **Async Processing**: Crawler supports concurrent URL validation and processing using asyncio and concurrent.futures
- **Error Handling**: Comprehensive error management with retry logic using tenacity
- **Browser Automation**: Uses Playwright for JavaScript-heavy sites requiring dynamic rendering
- **Logging**: Dual logging to console (colored) and files (app.log, crawler.log)
- **Results Export**: Supports both JSON and Excel export formats

## Real-Time Progress Tracking

The application uses **Server-Sent Events (SSE)** to stream real-time progress updates from the backend to the frontend during crawling operations.

### SSE Implementation:
- **Backend** (`app.py`): Progress updates sent via `/crawl` endpoint using Flask's `Response` with `stream_with_context`
- **Frontend** (`templates/index.html`): EventSource API listens for progress events and updates UI
- **Progress Callback**: WebCrawler uses `progress_callback` parameter to emit status updates during:
  - Search phase (finding relevant URLs)
  - Validation phase (AI-powered URL selection)
  - Scraping phase (content extraction from each site)
  - Completion phase (final results)

### Progress Event Format:
```javascript
{
  "status": "progress",
  "message": "Scraping site 3 of 15...",
  "current": 3,
  "total": 15,
  "percentage": 20
}
```

### User Experience:
- Real-time progress bar with percentage
- Status messages showing current operation
- Animated spinner during processing
- Automatic UI updates without polling

## Smart Objective Suggestions

The frontend includes intelligent objective suggestions based on the entered URL pattern.

### Features:
- **URL Pattern Detection**: Analyzes entered URL to determine site type and likely use cases
- **Context-Aware Suggestions**: Different suggestions for e-commerce, corporate, news, and general sites
- **Common Patterns**:
  - E-commerce sites: "Extract product names, prices, and specifications"
  - Corporate sites: "Extract company information and contact details"
  - News sites: "Extract article titles, authors, and publication dates"
  - Default: "Extract key information and main content"

### Implementation:
Located in `templates/index.html`, the suggestion system uses JavaScript to:
1. Monitor URL input changes
2. Detect patterns (e.g., `/product`, `/shop`, `/about`)
3. Display relevant objective suggestions
4. Allow one-click selection of suggested objectives

## Production Deployment

### Live Application:
- **URL**: https://crawlai.onrender.com
- **Platform**: Render.com (Free tier)
- **Region**: Oregon
- **Auto-Deploy**: Enabled from GitHub `main` branch

### Deployment Configuration (`render.yaml`):
```yaml
services:
  - type: web
    name: crawlai
    env: python
    runtime: python-3.11.9
    region: oregon
    plan: free
    buildCommand: pip install -r requirements.txt && playwright install --with-deps chromium
    startCommand: gunicorn app:app
```

### Environment Variables (Required):
- `DEEPSEEK_API_KEY`: Primary LLM for content extraction
- `SERP_API_KEY`: Google search functionality
- `FIRECRAWL_API_KEY`: Web scraping service
- `OPENAI_API_KEY`: Fallback LLM (optional)
- `RENDER`: Set to `true` automatically by Render platform

### Deployment Process:
1. Code pushed to GitHub `main` branch
2. Render automatically detects changes
3. Triggers build: installs dependencies and Playwright
4. Runs start command: `gunicorn app:app`
5. Health checks verify service is responding
6. Traffic routed to new deployment

## Recent Deployment Fixes (November 2025)

### Issue 1: Missing nest-asyncio Dependency
**Date**: November 12, 2025
**Commit**: e8a0184

**Problem**: After upgrading Playwright from 1.40.0 to 1.48.0 and adding `greenlet>=3.1.1` for Python 3.13 compatibility, the build succeeded but the app crashed at runtime with:
```
ModuleNotFoundError: No module named 'nest_asyncio'
```

**Root Cause**: `nest_asyncio` was imported in `improved_web_crawler.py:20` but not listed in `requirements.txt`. The build phase didn't catch this because the import only executes at runtime.

**Fix**: Added `nest-asyncio==1.6.0` to `requirements.txt`

**Deployment**: Deploy ID `dep-d4a89ebuibrs73dkpuag` - Status: live at 2025-11-12T13:02:18Z

**Key Lesson**: Successful builds don't guarantee runtime success. Missing imports only fail when code executes.

---

### Issue 2: Form Submission 404 Error
**Date**: November 12, 2025
**Commit**: 427263c

**Problem**: When submitting the crawl form **without** expanding "Advanced Crawling Options" (i.e., using default single-page mode), users received a 404 Not Found error.

**Root Cause Analysis**:
1. Form had `action="/scrape-form"` which doesn't exist as a backend endpoint
2. JavaScript event handler only called `e.preventDefault()` for multi-page modes
3. In single mode (default), form submitted normally to the non-existent `/scrape-form` URL

**Code Investigation** (`templates/index.html:1061-1069`):
```javascript
// Problematic code
if (mode === 'single') {
    button.innerHTML = '<div class="spinner"></div> Scraping...';
    button.disabled = true;
    // No e.preventDefault() - form submits to /scrape-form!
} else {
    e.preventDefault(); // Only prevented for multi-page
    startAdvancedCrawl();
}
```

**Fix Applied**:
1. Changed form action from `/scrape-form` to `#` (line 483)
2. Always call `e.preventDefault()` regardless of mode
3. All modes now use `startAdvancedCrawl()` which calls `/crawl` API endpoint

**Fixed Code**:
```javascript
document.getElementById('scrapeForm').addEventListener('submit', function(e) {
    e.preventDefault(); // Always prevent default form submission
    const button = document.getElementById('submitBtn');
    const mode = document.querySelector('input[name="crawl_mode"]:checked').value;

    // All modes now use the API
    button.innerHTML = '<div class="spinner"></div> Scraping...';
    button.disabled = true;
    startAdvancedCrawl();
});
```

**Deployment**: Deploy ID `dep-d4a8giruibrs73dkron0` - Status: live at 2025-11-12T13:16:49Z

**Key Lesson**: When using API-based architecture, always prevent default form submissions to avoid accidental navigation to non-existent endpoints.

---

### Dependencies Timeline:
- **Playwright**: Upgraded from 1.40.0 → 1.48.0 (for greenlet 3.1.1 compatibility)
- **greenlet**: Added `>=3.1.1` (Python 3.13 support)
- **nest-asyncio**: Added `1.6.0` (runtime dependency fix)

### Current Production Status:
✅ All systems operational
✅ Real-time progress tracking working
✅ Smart objective suggestions active
✅ Form submission (all modes) functional
✅ Multi-site crawling operational
✅ API endpoints responding correctly
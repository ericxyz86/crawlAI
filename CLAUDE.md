# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based web scraping application with CrawlAI integration that provides both web interface and API access for scraping websites. The application specializes in handling dynamic content with advanced infinite scroll capabilities, particularly optimized for YouTube and other JavaScript-heavy sites.

## Development Commands

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
python app.py

# Run with custom host/port
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Testing
```bash
# Run infinite scroll tests
python test_infinite_scroll.py

# Run YouTube-specific tests
python test_youtube_scroll.py

# No formal test framework is configured - tests are standalone scripts
```

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables as needed
```

## Core Architecture

### Main Components

1. **FastAPI Web Application** (`app.py`)
   - Serves both HTML interface and JSON API
   - WebSocket support for real-time progress updates
   - Dual endpoints: `/scrape` (JSON API) and `/scrape-form` (HTML forms)

2. **Single Page Scraper** (`scraper.py`)
   - Handles individual page scraping with optional infinite scroll
   - YouTube-optimized scrolling with specialized DOM selectors
   - Human behavior simulation to avoid bot detection

3. **Advanced Multi-Page Crawler** (`advanced_scraper.py`)
   - Four crawling modes: Single, Deep, Sitemap, Pattern-based
   - Async job management with unique job IDs
   - Real-time progress tracking via WebSocket

4. **Frontend Interface** (`templates/`)
   - Modern responsive design with gradient UI
   - Real-time progress tracking with WebSocket integration
   - Export functionality (CSV, JSON, Markdown)

### Key Technologies
- **FastAPI**: Modern async web framework
- **CrawlAI**: AI-powered web crawling engine
- **Playwright**: Browser automation (via CrawlAI)
- **WebSockets**: Real-time progress updates
- **Jinja2**: Template engine for HTML rendering

## Special Features

### YouTube-Optimized Infinite Scroll
- Specialized DOM selectors for YouTube's video grid
- Anti-bot countermeasures with human behavior simulation
- Content stability detection specific to YouTube's loading patterns
- Fallback mechanisms for different YouTube page types

### Human Behavior Simulation
- Randomized scroll timing to mimic natural user behavior
- Variable scroll distances and patterns
- Gradual scrolling in chunks rather than instant jumps
- Random pauses to avoid detection

### Real-Time Progress Tracking
- WebSocket integration for live updates during advanced crawls
- Fallback polling if WebSocket fails
- Detailed progress metrics (pages/min, elapsed time)
- Cancellation support for long-running crawls

## Data Flow Patterns

### Single Page Scraping
```
User Request → FastAPI → WebScraper → CrawlAI → Browser → JavaScript Injection → Content Extraction → Response
```

### Advanced Crawling
```
User Request → FastAPI → AdvancedWebScraper → Job Creation → Async Processing → Progress Updates → WebSocket → Results
```

### WebSocket Progress Updates
```
Job Start → WebSocket Connection → Progress Updates → Real-time UI Updates → Completion/Error Handling
```

## API Endpoints

### Primary Endpoints
- `GET /` - Web interface
- `POST /scrape` - JSON API for single page scraping
- `POST /scrape-form` - HTML form submission
- `POST /crawl/advanced` - Advanced multi-page crawling
- `WebSocket /ws/progress/{job_id}` - Real-time progress updates

### Advanced Crawling Modes
- **Single**: Enhanced single-page with infinite scroll
- **Deep**: Breadth-first crawling with depth limits
- **Sitemap**: XML sitemap parsing with fallback locations
- **Pattern**: Regex-based URL filtering

## Error Handling and Testing

### Error Handling Strategy
- Graceful degradation with fallback mechanisms
- Comprehensive logging for debugging
- User-friendly error messages
- Automatic retry mechanisms for transient failures

### Testing Approach
- Comparative testing (with/without infinite scroll)
- Multiple URL testing for different site types
- Performance metrics (word count improvements)
- Error handling validation

## Security and Ethics

### Ethical Crawling Framework
- Robots.txt compliance checking
- Automatic crawl delays based on robots.txt
- Rate limiting and respectful request patterns
- Same-domain restrictions and external link filtering

### Security Measures
- URL validation to prevent malicious inputs
- Timeout configurations to prevent hanging
- Rate limiting and request throttling

## Configuration

### Environment Variables
Set up `.env` file based on `.env.example` for:
- Browser configuration (headless mode, browser type)
- Timeout and delay settings
- Logging level controls

### Key Settings
- **Headless browser mode** for production deployment
- **Browser type selection** (Chromium/Firefox/Webkit)
- **Configurable timeouts** and delays
- **Resource limits** for scalability

## Development Notes

### Code Organization
- Modular architecture with clear separation of concerns
- Async/await patterns throughout for performance
- Type hints with Pydantic models for data validation
- Consistent error handling with try/catch blocks

### Performance Considerations
- Intelligent scroll detection to avoid unnecessary operations
- Content stability checking to minimize redundant requests
- Parallel processing for multi-page crawls
- Efficient DOM selectors for fast content extraction

### Frontend Architecture
- Pure JavaScript (no external frameworks)
- CSS3 with modern features (gradients, grid, flexbox)
- WebSocket API for real-time updates
- Responsive design for all screen sizes
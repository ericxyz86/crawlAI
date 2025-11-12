# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a Flask backend + React frontend web crawler application with AI-powered content extraction.

### Core Architecture:
- **Flask Backend** (`app.py`): Main server on port 5001 with CORS enabled for frontend on port 5173 (Vite default)
- **AI-Powered Web Crawler** (`improved_web_crawler.py`): Advanced crawler with multiple LLM support (Deepseek R1 primary, OpenAI o4-mini fallback)
- **React Frontend** (`frontend/`): Modern React + TypeScript + Vite + Tailwind CSS setup
- **Dual Frontend Support**: Both React SPA and Flask templates (`templates/index.html`)

### Key Components:
- **WebCrawler**: Main orchestrator class with SearchEngine, URLProcessor, ContentExtractor
- **ConfigManager**: Handles API keys and configuration
- **Results Storage**: JSON files saved to `/results/` directory

## Development Commands

### Backend Setup:
```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run main application
python app.py  # Main application on port 5001
```

### Frontend Setup:
```bash
cd frontend
npm install
npm run dev      # Development server (port 5173)
npm run build    # Production build (includes TypeScript compilation)
npm run lint     # ESLint linting
npm run preview  # Preview production build
```

### Frontend Development Commands:
```bash
# TypeScript compilation only
tsc -b

# Build for production
tsc -b && vite build
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

### Frontend-Backend Communication:
- Backend serves both API endpoints and static templates
- React frontend communicates via REST API
- CORS configured for port 5173 (Vite dev server)

## Development Considerations

- **No Formal Testing Framework**: Current setup uses basic test files, consider adding pytest for Python and Jest for React
- **Multiple Crawler Versions**: `R1_web_crawler.py` is a simplified version, `improved_web_crawler.py` is the main implementation
- **Async Processing**: Crawler supports concurrent URL validation and processing using asyncio and concurrent.futures
- **Error Handling**: Comprehensive error management with retry logic using tenacity
- **Browser Automation**: Uses Playwright for JavaScript-heavy sites requiring dynamic rendering
- **Logging**: Dual logging to console (colored) and files (app.log, crawler.log)
- **Results Export**: Supports both JSON and Excel export formats
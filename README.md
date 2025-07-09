# Web Scraper App

A modern web scraping application built with FastAPI and Crawl4AI that allows users to scrape websites and extract content in markdown format.

## Features

- 🕷️ **Web Scraping**: Scrape any website URL using Crawl4AI
- 📝 **Markdown Output**: Clean, formatted markdown content
- 🎨 **Modern UI**: Beautiful, responsive web interface
- ⚡ **Fast Processing**: Optimized for speed with async operations
- 🛡️ **Error Handling**: Comprehensive error handling and validation
- 📊 **Content Stats**: Word count, links, and image statistics

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Crawl4AI:
```bash
crawl4ai-setup
```

3. Copy environment variables:
```bash
cp .env.example .env
```

## Usage

### Running the Application

Start the server:
```bash
python app.py
```

Or using uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Web Interface

1. Open your browser and go to `http://localhost:8000`
2. Enter a URL you want to scrape
3. Click "Scrape Website" 
4. View the extracted content in markdown format

### API Endpoints

#### POST `/scrape`
Scrape a website URL programmatically.

**Request body:**
```json
{
  "url": "https://example.com",
  "word_count_threshold": 10,
  "content_filter_threshold": 0.48,
  "css_selector": null,
  "wait_for": 3000,
  "page_timeout": 30000,
  "excluded_tags": ["nav", "footer", "aside", "script", "style"]
}
```

**Response:**
```json
{
  "success": true,
  "url": "https://example.com",
  "title": "Example Domain",
  "description": "This domain is for use in illustrative examples...",
  "markdown": "# Example Domain\n\nThis domain is for use in illustrative examples in documents...",
  "links": [...],
  "images": [...],
  "metadata": {...},
  "word_count": 45
}
```

#### GET `/health`
Health check endpoint.

## Configuration

The application can be configured through environment variables:

- `APP_ENV`: Application environment (development/production)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `CRAWL4AI_HEADLESS`: Run browser in headless mode (true/false)
- `CRAWL4AI_BROWSER_TYPE`: Browser type (chromium/firefox/webkit)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

## Technologies Used

- **FastAPI**: Modern, fast web framework for building APIs
- **Crawl4AI**: AI-powered web crawling and scraping
- **Jinja2**: Template engine for HTML rendering
- **Uvicorn**: ASGI server for running the application
- **Pydantic**: Data validation and settings management

## License

This project is open source and available under the [MIT License](LICENSE).
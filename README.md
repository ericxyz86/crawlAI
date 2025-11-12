# Improved Web Crawler

A comprehensive web crawler that extracts structured information from websites based on user objectives. This tool works with company names, domains, or specific URLs across any industry.

## Features

- **Generic Information Extraction**: Works for any industry or domain, not just automotive
- **Objective-Driven Crawling**: Focuses on retrieving information relevant to user queries
- **Smart URL Selection**: Uses AI to select the most relevant pages to crawl
- **Adaptive Content Extraction**: Tailors content extraction to the user's specific objective
- **Dual Input Modes**:
  - Company name search: Finds and crawls relevant websites via search API
  - Direct URL/domain input: Focuses on exploring that specific website

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd webcrawler
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following API keys:
```
DEEPSEEK_API_KEY=your_deepseek_api_key
SERP_API_KEY=your_serpapi_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

## Usage

### Command Line Interface

Run the crawler from the command line:

```bash
python improved_web_crawler.py
```

You'll be prompted to enter:
- Company name or URL to crawl
- Objective (what information you want to find)

### Web Interface

Start the Flask web server:

```bash
python app.py
```

Access the web interface at http://127.0.0.1:5001/

### API Usage

Make a POST request to the `/crawl` endpoint:

```bash
curl -X POST http://127.0.0.1:5001/crawl \
  -H "Content-Type: application/json" \
  -d '{"company_name": "example.com", "objective": "find pricing information"}'
```

Response format:
```json
{
  "success": true,
  "urls": ["https://example.com/pricing", "https://example.com/products"],
  "data": {
    "entity_overview": "...",
    "products_and_services": [...],
    "contact_information": {...},
    "objective_related_information": {...}
  },
  "metadata": {
    "crawl_time": "2025-05-05T12:00:00",
    "execution_time_seconds": 8.45,
    "objective": "find pricing information",
    "entity_name": "example.com",
    "input_type": "url"
  },
  "filename": "crawl_results_20250505_120000.json"
}
```

## Code Structure

- **WebCrawler**: Main class that orchestrates the crawling process
- **ConfigManager**: Handles configuration and API key management
- **SearchEngine**: Handles search operations via search APIs
- **URLProcessor**: Manages URL selection, validation, and discovery
- **ContentExtractor**: Extracts and structures content from web pages

## Enhancements Over Original Implementation

1. **Industry-Agnostic**: Removed automotive-specific code for true generic functionality
2. **Improved Architecture**: Modular, class-based structure with clear separation of concerns
3. **Better Error Handling**: Comprehensive logging and robust error recovery
4. **Enhanced URL Discovery**: Smarter internal link discovery focused on objective relevance
5. **Security Improvements**: Better API key management and safer request handling
6. **Performance Optimizations**: Parallel processing and efficient content extraction
7. **More Robust Content Extraction**: Pattern generation based on user objectives
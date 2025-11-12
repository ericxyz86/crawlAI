import os
import time
import json
import requests
import openai
import logging
import asyncio
import traceback
import re
import sys
import concurrent.futures
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Optional, Any, Tuple
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import validators
from datetime import datetime
from colorama import Fore, Style, init
from tenacity import retry, stop_after_attempt, wait_exponential
import nest_asyncio

# Try to import playwright, but make it optional
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None

# Apply nest_asyncio patch early
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crawler.log")
    ]
)
logger = logging.getLogger("WebCrawler")

# Load environment variables
load_dotenv(override=True)


class ColoredLogger:
    """Helper class to print colored log messages to console."""
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    @staticmethod
    def info(message: str) -> None:
        """Log an info message in green."""
        print(f"{ColoredLogger.GREEN}{message}{ColoredLogger.RESET}")

    @staticmethod
    def warning(message: str) -> None:
        """Log a warning message in yellow."""
        print(f"{ColoredLogger.YELLOW}{message}{ColoredLogger.RESET}")

    @staticmethod
    def error(message: str) -> None:
        """Log an error message in red."""
        print(f"{ColoredLogger.RED}{message}{ColoredLogger.RESET}")

    @staticmethod
    def debug(message: str) -> None:
        """Log a debug message in blue."""
        print(f"{ColoredLogger.BLUE}{message}{ColoredLogger.RESET}")


class ConfigManager:
    """Handles configuration and API keys."""
    
    # Define class-level constants for timeouts to avoid instance attribute issues
    TIMEOUT_SHORT = 10  # For quick operations like HEAD requests
    TIMEOUT_MEDIUM = 30  # For standard operations
    TIMEOUT_LONG = 120    # For potentially slow operations
    PLAYWRIGHT_TIMEOUT = 30000  # 30 seconds in milliseconds for Playwright
    MAX_CONTENT_LENGTH = 10000 # Max characters to feed to AI
    
    def __init__(self):
        """Initialize the configuration manager and load API keys."""
        # Load API keys from environment
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Validate required API keys
        self._validate_api_keys()
        
        # Initialize API clients
        self.deepseek_client = self._init_deepseek_client()
        self.openai_client = self._init_openai_client() if self.openai_api_key else None
        
        # Default request headers
        self.default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        
        # Define model names and default selection
        self.deepseek_model_name = "deepseek-chat"
        self.openai_model_name = "o4-mini"
        self.llm_model = "R1"
    
    @property
    def timeout_short(self):
        """Return the short timeout value."""
        return self.TIMEOUT_SHORT
    
    @property
    def timeout_medium(self):
        """Return the medium timeout value."""
        return self.TIMEOUT_MEDIUM
    
    @property
    def timeout_long(self):
        """Return the long timeout value."""
        return self.TIMEOUT_LONG
        
    def _validate_api_keys(self) -> None:
        """Validate that required API keys are present."""
        if not self.deepseek_api_key:
            logger.error("DEEPSEEK_API_KEY not found in .env file")
            ColoredLogger.error("DEEPSEEK_API_KEY not found in .env file")
            sys.exit(1)
        
        if not self.serp_api_key:
            logger.warning("SERP_API_KEY not found in .env file")
            ColoredLogger.warning("SERP_API_KEY not found in .env file")
            
        if not self.firecrawl_api_key:
            logger.warning("FIRECRAWL_API_KEY not found in .env file")
            ColoredLogger.warning("FIRECRAWL_API_KEY not found in .env file")
        
        if not getattr(self, 'openai_api_key', None):
            logger.warning("OPENAI_API_KEY not found in .env file; o4-mini model unavailable")
            ColoredLogger.warning("OPENAI_API_KEY not found in .env file; o4-mini model unavailable")

    def _init_deepseek_client(self) -> openai.Client:
        """Initialize and return the Deepseek client."""
        return openai.Client(
            api_key=self.deepseek_api_key, 
            base_url="https://api.deepseek.com/v1", 
            timeout=self.TIMEOUT_LONG  # Using class constant instead of instance attribute
        )
    
    def _init_openai_client(self) -> openai.Client:
        """Initialize and return the OpenAI client for o4-mini model."""
        return openai.Client(
            api_key=self.openai_api_key,
            timeout=self.TIMEOUT_LONG
        )


class SearchEngine:
    """Handles search operations using search APIs with multi-query strategy."""
    
    def __init__(self, config: ConfigManager):
        """Initialize with configuration."""
        self.config = config
        self.serp_api_key = config.serp_api_key
        
        # Regional retailer databases for enhanced company searches
        self.retailer_databases = {
            'philippines': {
                'electronics': ['powermaccenter.com', 'beyondthebox.ph', 'istore.ph', 'abenson.com', 'sm.com.ph/appliances'],
                'automotive': ['autodeal.com.ph', 'carmudi.com.ph', 'philkotse.com', 'suzuki.com.ph', 'toyota.com.ph'],
                'general': ['lazada.com.ph', 'shopee.ph', 'zalora.com.ph']
            },
            'us': {
                'electronics': ['bestbuy.com', 'amazon.com', 'apple.com', 'walmart.com', 'target.com'],
                'automotive': ['autotrader.com', 'cars.com', 'edmunds.com', 'kbb.com'],
                'general': ['amazon.com', 'walmart.com', 'target.com', 'costco.com']
            },
            'global': {
                'electronics': ['apple.com', 'samsung.com', 'sony.com', 'lg.com'],
                'price_comparison': ['pricewatch.com', 'shopping.google.com', 'nextag.com'],
                'reviews': ['gsmarena.com', 'techradar.com', 'cnet.com', 'engadget.com']
            }
        }
        
        # Pricing-focused search terms for different objectives
        self.objective_search_terms = {
            'pricing': ['price', 'buy', 'purchase', 'cost', 'deal', 'sale'],
            'products': ['specs', 'features', 'models', 'variants', 'review'],
            'services': ['support', 'warranty', 'service', 'repair'],
            'contact': ['contact', 'store', 'location', 'branch', 'dealer']
        }
    
    def clean_query(self, query: str) -> str:
        """Clean up the query by removing URLs and unnecessary characters."""
        # Remove URLs
        query = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            query,
        )
        # Remove extra whitespace
        query = " ".join(query.split())
        
        return query.strip()
    
    def search_google(self, query: str) -> List[Dict[str, Any]]:
        """Search Google using SerpAPI and return top results."""
        try:
            cleaned_query = self.clean_query(query)
            if not cleaned_query:
                logger.error("Invalid search query. Please enter a company name rather than a URL.")
                ColoredLogger.error("Invalid search query. Please enter a company name rather than a URL.")
                return []

            logger.info(f"Searching Google for '{cleaned_query}'...")
            ColoredLogger.info(f"Searching Google for '{cleaned_query}'...")

            # Construct the API URL with parameters
            params = {
                "api_key": self.serp_api_key,
                "q": cleaned_query,
                "engine": "google",
                "google_domain": "google.com",
                "gl": "us",  # Changed from "ph" to "us" for more generic results
                "hl": "en",
                "num": "10",
            }

            try:
                # Make the request
                logger.debug("Making request to SerpAPI...")
                ColoredLogger.debug("Making request to SerpAPI...")
                
                response = requests.get(
                    "https://serpapi.com/search.json",
                    params=params,
                    timeout=self.config.timeout_medium,
                )

                # Print debug information
                logger.debug(f"Response status code: {response.status_code}")
                ColoredLogger.debug(f"Response status code: {response.status_code}")

                if response.status_code == 200:
                    try:
                        logger.debug("Parsing JSON response...")
                        ColoredLogger.debug("Parsing JSON response...")
                        
                        data = response.json()
                        logger.debug(f"Response data keys: {list(data.keys())}")
                        ColoredLogger.debug(f"Response data keys: {list(data.keys())}")

                        if "error" in data:
                            logger.error(f"API Error: {data['error']}")
                            ColoredLogger.error(f"API Error: {data['error']}")
                            return []

                        logger.debug("Getting organic results...")
                        ColoredLogger.debug("Getting organic results...")
                        
                        results = data.get("organic_results", [])
                        logger.debug(f"Found {len(results)} results")
                        ColoredLogger.debug(f"Found {len(results)} results")
                        
                        if not results:
                            logger.warning("No results found. Try refining your search query.")
                            ColoredLogger.warning("No results found. Try refining your search query.")
                            
                        return results
                    except json.JSONDecodeError as e:
                        logger.error(f"Error: Invalid JSON response from API: {str(e)}")
                        ColoredLogger.error(f"Error: Invalid JSON response from API: {str(e)}")
                        logger.error(f"Response content: {response.text[:500]}")
                        ColoredLogger.error(f"Response content: {response.text[:500]}")
                        return []
                elif response.status_code == 401:
                    logger.error("Error: Invalid API key. Please check your SERP_API_KEY in the .env file.")
                    ColoredLogger.error("Error: Invalid API key. Please check your SERP_API_KEY in the .env file.")
                    return []
                else:
                    logger.error(f"Error: API request failed with status code {response.status_code}")
                    ColoredLogger.error(f"Error: API request failed with status code {response.status_code}")
                    try:
                        error_data = response.json()
                        logger.error(f"Error details: {error_data}")
                        ColoredLogger.error(f"Error details: {error_data}")
                    except:
                        logger.error(f"Response content: {response.text[:500]}")
                        ColoredLogger.error(f"Response content: {response.text[:500]}")
                    return []

            except requests.exceptions.RequestException as e:
                logger.error(f"Error making request: {str(e)}")
                ColoredLogger.error(f"Error making request: {str(e)}")
                return []

        except Exception as e:
            logger.error(f"Unexpected error in search_google: {str(e)}")
            ColoredLogger.error(f"Unexpected error in search_google: {str(e)}")
            return []
    
    def detect_region_and_category(self, query: str, objective: str) -> Tuple[str, str]:
        """Detect region and category from query and objective."""
        query_lower = query.lower()
        objective_lower = objective.lower()
        combined_text = f"{query_lower} {objective_lower}"
        
        # Detect region
        region = 'global'  # default
        if any(term in combined_text for term in ['philippines', 'ph', 'manila', 'cebu']):
            region = 'philippines'
        elif any(term in combined_text for term in ['usa', 'us', 'america', 'american']):
            region = 'us'
        
        # Detect category
        category = 'general'  # default
        if any(term in combined_text for term in ['apple', 'iphone', 'samsung', 'phone', 'laptop', 'watch', 'electronics']):
            category = 'electronics'
        elif any(term in combined_text for term in ['car', 'auto', 'vehicle', 'toyota', 'honda', 'bmw']):
            category = 'automotive'
        
        return region, category
    
    def generate_retailer_queries(self, base_query: str, objective: str, max_queries: int = 5) -> List[str]:
        """Generate retailer-specific search queries for comprehensive company searches."""
        region, category = self.detect_region_and_category(base_query, objective)
        queries = [base_query]  # Start with original query
        
        # Get relevant retailers for region and category
        retailers = []
        if region in self.retailer_databases:
            if category in self.retailer_databases[region]:
                retailers.extend(self.retailer_databases[region][category][:3])
            if 'general' in self.retailer_databases[region]:
                retailers.extend(self.retailer_databases[region]['general'][:2])
        
        # Add global retailers for certain objectives
        if 'pricing' in objective.lower() or 'price' in objective.lower():
            if 'global' in self.retailer_databases and 'price_comparison' in self.retailer_databases['global']:
                retailers.extend(self.retailer_databases['global']['price_comparison'][:2])
        
        # Generate retailer-specific queries
        objective_terms = []
        for obj_key, terms in self.objective_search_terms.items():
            if obj_key in objective.lower():
                objective_terms.extend(terms[:2])
        
        for retailer in retailers[:max_queries-1]:
            retailer_name = retailer.split('.')[0]  # Get main retailer name
            if objective_terms:
                query = f"{base_query} {retailer_name} {objective_terms[0]}"
            else:
                query = f"{base_query} {retailer_name}"
            queries.append(query)
            
            if len(queries) >= max_queries:
                break
        
        return queries[:max_queries]
    
    def search_comprehensive(self, query: str, objective: str = "") -> List[Dict[str, Any]]:
        """Perform comprehensive multi-query search for company names."""
        all_results = []
        seen_urls = set()
        
        # Generate multiple targeted queries
        search_queries = self.generate_retailer_queries(query, objective)
        
        logger.info(f"Performing comprehensive search with {len(search_queries)} queries...")
        ColoredLogger.info(f"Performing comprehensive search with {len(search_queries)} queries...")
        
        # Execute searches in parallel for efficiency
        def search_single_query(q):
            try:
                return self.search_google(q)
            except Exception as e:
                logger.error(f"Error searching query '{q}': {str(e)}")
                return []
        
        # Use ThreadPoolExecutor for parallel searches
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_query = {executor.submit(search_single_query, q): q for q in search_queries}
            
            for future in concurrent.futures.as_completed(future_to_query):
                query_used = future_to_query[future]
                try:
                    results = future.result()
                    logger.info(f"Query '{query_used}' returned {len(results)} results")
                    
                    for result in results:
                        url = result.get('link', '')
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            # Add metadata about which query found this result
                            result['found_by_query'] = query_used
                            all_results.append(result)
                            
                except Exception as e:
                    logger.error(f"Error processing results for query '{query_used}': {str(e)}")
        
        logger.info(f"Comprehensive search found {len(all_results)} unique results")
        ColoredLogger.info(f"Comprehensive search found {len(all_results)} unique results")
        
        # Sort by relevance (original query results first, then by position)
        all_results.sort(key=lambda x: (x.get('found_by_query') != query, x.get('position', 999)))
        
        return all_results[:20]  # Return top 20 results


class URLProcessor:
    """Handles URL selection, validation, and processing."""
    
    def __init__(self, config: ConfigManager):
        """Initialize with configuration."""
        self.config = config
        self.default_headers = config.default_headers
    
    def select_urls_with_ai(self, company: str, objective: str, serp_results: List[Dict[str, Any]], llm: str) -> List[str]:
        """
        Use AI to select the most relevant URLs from SERP results for the given company and objective.
        Returns a list of URLs.
        """
        logger.debug(f"select_urls_with_ai received llm: '{llm}'") # DIAGNOSTIC LOG
        try:
            logger.debug("Preparing data for AI URL selection...")
            ColoredLogger.debug("Preparing data for AI URL selection...")
            
            # Prepare the data for AI
            serp_data = [
                {
                    "title": r.get("title"),
                    "link": r.get("link"),
                    "snippet": r.get("snippet"),
                }
                for r in serp_results
                if r.get("link")
            ]

            # Generate keywords based on objective
            objective_keywords = []
            if objective:
                # Extract all words from objective
                objective_keywords = [w.lower() for w in re.findall(r"\w+", objective)]

            # Add specific keywords based on common information pages
            common_page_keywords = [
                "about", "contact", "pricing", "services", "products", 
                "overview", "models", "catalog", "portfolio", "features",
                "solutions", "specs", "specifications", "details", "options"
            ]
            
            # Combine all keywords for priority pages
            priority_keywords = common_page_keywords + objective_keywords
            
            for result in serp_data:
                link = result.get("link", "").lower()
                if any(keyword in link for keyword in priority_keywords):
                    logger.info(f"Found objective-specific page: {link}")
                    ColoredLogger.info(f"Found objective-specific page: {link}")

            logger.debug(f"Created serp_data with {len(serp_data)} entries")
            ColoredLogger.debug(f"Created serp_data with {len(serp_data)} entries")

            messages = [
                {
                    "role": "system",
                    "content": """You are a URL selector that always responds with valid JSON. You select URLs from the SERP results relevant to the company and objective. 
                    Prioritize official websites and pages containing information directly related to the objective. Your response must be a JSON object with a 'selected_urls' array property containing strings.
                    If the objective mentions specific information types (like pricing, contact info, products, etc.), prioritize pages that likely contain that information.""",
                },
                {
                    "role": "user",
                    "content": (
                        f"Company: {company}\n"
                        f"Objective: {objective}\n"
                        f"SERP Results: {json.dumps(serp_data)}\n\n"
                        "Return a JSON object with a property 'selected_urls' that contains an array "
                        "of URLs most likely to help meet the objective. Prioritize pages directly related to the objective. "
                        'For example: {"selected_urls": ["https://example.com/pricing", "https://example.com"]}'
                    ),
                },
            ]

            # Determine client and model based on llm setting
            if llm == "o4-mini":
                client_to_use = self.config.openai_client
                model_name = self.config.openai_model_name
            else: # Default to R1 (Deepseek)
                client_to_use = self.config.deepseek_client
                model_name = self.config.deepseek_model_name

            logger.info(f"Using model '{model_name}' for URL selection.")
            
            # Base parameters common to all models
            api_params = {
                "model": model_name,
                "messages": messages,
            }
            
            # Model-specific parameters
            if llm == "o4-mini":
                # o4-mini requires max_completion_tokens and doesn't support custom temperature
                api_params["max_completion_tokens"] = 500
                # No temperature parameter - o4-mini only supports default temperature (1.0)
            else:
                # Other models use max_tokens and support custom temperature
                api_params["max_tokens"] = 500
                api_params["temperature"] = 0.7
            
            try:
                response = client_to_use.chat.completions.create(**api_params)
                ai_response = response.choices[0].message.content

                try:
                    if ai_response.startswith("```json"):
                        ai_response = ai_response[7:-3]  # Remove markdown code block markers

                    parsed_result = json.loads(ai_response)
                    urls = parsed_result.get("selected_urls", [])

                    # Add objective-specific pages if not already included
                    additional_urls = []
                    for result in serp_data:
                        link = result.get("link", "").lower()
                        if any(keyword in link for keyword in priority_keywords) and link not in urls:
                            additional_urls.append(result.get("link"))

                    urls.extend(additional_urls)
                    # Remove duplicates while preserving order
                    return list(dict.fromkeys(urls))

                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing AI response: {str(e)}")
                    ColoredLogger.error(f"Error parsing AI response: {str(e)}")
                    return []

            except Exception as e:
                logger.error(f"Error selecting URLs with AI: {e}")
                ColoredLogger.error(f"Error selecting URLs with AI: {e}")
                return []
        except Exception as e:
            logger.error(f"Error selecting URLs with AI: {e}")
            ColoredLogger.error(f"Error selecting URLs with AI: {e}")
            return []
    
    def validate_url(self, url: str) -> Optional[str]:
        """Validate if a URL is accessible, trying GET if HEAD fails."""
        try:
            # Add scheme if missing
            if not urlparse(url).scheme:
                url = "https://" + url

            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                logger.error(f"Invalid URL structure: {url}")
                ColoredLogger.error(f"Invalid URL structure: {url}")
                return None

            session = requests.Session()
            session.headers.update(self.default_headers)

            response = None
            try:
                # Try HEAD first (more efficient)
                response = session.head(
                    url, 
                    timeout=self.config.timeout_short, 
                    allow_redirects=True
                )
                response.raise_for_status()
                logger.info(f"URL HEAD validation successful: {response.url} (Status: {response.status_code})")
                ColoredLogger.info(f"URL HEAD validation successful: {response.url} (Status: {response.status_code})")
                return response.url
            except requests.exceptions.RequestException:
                logger.warning(f"HEAD request failed for {url}. Trying GET...")
                ColoredLogger.warning(f"HEAD request failed for {url}. Trying GET...")
                try:
                    # Fallback to GET request (stream=True to avoid downloading full content)
                    response = session.get(
                        url, 
                        timeout=self.config.timeout_medium, 
                        allow_redirects=True, 
                        stream=True
                    )
                    response.raise_for_status()
                    # Close the connection after checking status
                    response.close()
                    logger.info(f"URL GET validation successful: {response.url} (Status: {response.status_code})")
                    ColoredLogger.info(f"URL GET validation successful: {response.url} (Status: {response.status_code})")
                    return response.url
                except requests.exceptions.RequestException as get_error:
                    logger.error(f"GET request also failed for {url}: {get_error}")
                    ColoredLogger.error(f"GET request also failed for {url}: {get_error}")
                    return None

        except Exception as e:
            logger.error(f"Error validating URL {url}: {str(e)}")
            ColoredLogger.error(f"Error validating URL {url}: {str(e)}")
            return None
    
    def validate_urls(self, urls: List[str]) -> List[str]:
        """Validate multiple URLs in parallel."""
        logger.info("Validating URLs...")
        ColoredLogger.info("Validating URLs...")
        
        valid_urls = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self.validate_url, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    valid_url = future.result()
                    if valid_url:
                        logger.info(f"URL validated: {valid_url}")
                        ColoredLogger.info(f"URL validated: {valid_url}")
                        valid_urls.append(valid_url)
                    else:
                        logger.error(f"Invalid URL: {url}")
                        ColoredLogger.error(f"Invalid URL: {url}")
                except Exception as e:
                    logger.error(f"Error validating {url}: {str(e)}")
                    ColoredLogger.error(f"Error validating {url}: {str(e)}")
        return valid_urls
    
    def discover_internal_links(self, root_url: str, objective: str, max_links: int = 50) -> List[Tuple[str, str]]:
        """
        Return a list of (url, anchor_text) tuples for same-domain links on the root page,
        filtered by relevance to the objective.
        """
        links = []
        try:
            response = requests.get(
                root_url, 
                headers=self.default_headers, 
                timeout=self.config.timeout_medium
            )
            if response.status_code != 200:
                return links

            soup = BeautifulSoup(response.text, "html.parser")
            root_domain = urlparse(root_url).netloc

            # Generate keywords based on objective
            objective_keywords = []
            if objective:
                objective_keywords = [w.lower() for w in re.findall(r"\w+", objective)]

            # Common information page identifiers
            common_page_keywords = [
                "about", "contact", "pricing", "services", "products", 
                "overview", "models", "catalog", "portfolio", "features",
                "solutions", "specs", "specifications", "details", "options"
            ]
            
            # Combined priority keywords
            priority_keywords = common_page_keywords + objective_keywords
            
            # First pass: collect all internal links
            all_links = []
            for a in soup.find_all("a", href=True):
                href = urljoin(root_url, a["href"].strip())
                parsed = urlparse(href)
                # Only include links from the same domain
                if parsed.netloc and parsed.netloc.endswith(root_domain):
                    clean_href = href.split("#")[0]  # Remove fragment identifiers
                    anchor_text = a.get_text(" ", strip=True)[:120]
                    all_links.append((clean_href, anchor_text))
            
            # Deduplicate links
            unique_links = list(dict.fromkeys(all_links))
            
            # Second pass: prioritize links relevant to objective
            prioritized_links = []
            other_links = []
            
            for href, text in unique_links:
                # Check if URL or anchor text contains any of the priority keywords
                if any(keyword in href.lower() for keyword in priority_keywords) or \
                   any(keyword in text.lower() for keyword in priority_keywords):
                    prioritized_links.append((href, text))
                else:
                    other_links.append((href, text))
            
            # Combine prioritized links first, then other links up to max_links
            result_links = prioritized_links + other_links
            links = result_links[:max_links]
            
            logger.info(f"Discovered {len(links)} internal links (prioritized {len(prioritized_links)})")
            ColoredLogger.info(f"Discovered {len(links)} internal links (prioritized {len(prioritized_links)})")
            
        except Exception as e:
            logger.warning(f"Error discovering internal links: {str(e)}")
            ColoredLogger.warning(f"Error discovering internal links: {str(e)}")
            
        return links


class ContentExtractor:
    """Handles content extraction from web pages."""
    
    def __init__(self, config: ConfigManager):
        """Initialize with configuration."""
        self.config = config
        self.firecrawl_api_key = config.firecrawl_api_key
        self.default_headers = config.default_headers

    def _generate_patterns_from_objective(self, objective: str) -> List[str]:
        """Generate regex patterns from the objective to find relevant content."""
        if not objective:
            return []

        patterns = []

        # Extract keywords from objective
        objective_words = re.findall(r'\b\w+\b', objective.lower())

        # Create patterns for each word
        for word in objective_words:
            patterns.append(rf'\b{re.escape(word)}\b')

        # Add specific patterns for common objectives
        if any(word in objective.lower() for word in ['price', 'pricing', 'cost']):
            patterns.extend([
                r'(?:PHP|₱|Php|\$|USD)\s*[\d,]+(?:\.\d{2})?',  # PHP/USD currency
                r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:PHP|₱|Php|\$|USD)',  # Numbers with currency
                r'SRP|MSRP|Price|Cost|Starting\s+at',  # Price indicators
                r'Starts\s+at|From\s+PHP|Price\s+range',
                r'₱[\d,]+|PHP\s*[\d,]+|\$[\d,]+',  # Direct currency patterns
                r'Series\s+\d+|Watch\s+SE|Watch\s+Ultra'  # Apple Watch specific models
            ])

        if any(word in objective.lower() for word in ['model', 'variant', 'car']):
            patterns.extend([
                r'\b(?:Model|Variant|Version|Trim)\b',
                r'\b(?:Standard|Premium|Advanced|Dynamic|Superior|Captain)\b',
                r'\b(?:MT|AT|CVT|Automatic|Manual)\b'  # Transmission types
            ])

        # Apple Watch specific patterns
        if any(word in objective.lower() for word in ['apple watch', 'watch']):
            patterns.extend([
                r'Apple\s+Watch\s+(?:Series\s+\d+|SE|Ultra)',  # Apple Watch models
                r'(?:41mm|45mm|40mm|44mm|49mm)',  # Watch sizes
                r'(?:GPS|Cellular|GPS\s*\+\s*Cellular)',  # Connectivity options
                r'(?:Aluminum|Stainless\s+Steel|Titanium)',  # Materials
                r'Sport\s+Band|Leather\s+Link|Milanese\s+Loop'  # Band types
            ])

        return patterns
    
    def scrape_url(self, url: str, objective: str = "") -> Optional[Dict[str, Any]]:
        """Scrape content from a URL with focus on information relevant to the objective."""
        try:
            response = requests.get(
                url,
                timeout=self.config.timeout_medium,
                headers=self.default_headers
            )
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "iframe"]):
                    element.decompose()

                # Generate patterns for objective-based extraction
                objective_patterns = self._generate_patterns_from_objective(objective)

                # Extract structured content
                content = {
                    "title": soup.title.string if soup.title else "",
                    "main_content": "",
                    "headings": [],
                    "paragraphs": [],
                    "lists": [],
                    "tables": [],  # Added dedicated field for tables
                    "contact_info": [],
                    "objective_related_info": [],  # Generic field for objective-related information
                    "url": url,  # Include the source URL
                }

                # Extract tables
                tables = soup.find_all("table")
                for table in tables:
                    table_data = []
                    rows = table.find_all("tr")
                    # Find table headers
                    headers = []
                    header_row = table.find("thead")
                    if header_row:
                        headers = [th.get_text(strip=True) for th in header_row.find_all("th")]

                    # Process each row
                    for row in rows:
                        cells = row.find_all(["td", "th"])
                        if cells:  # Skip empty rows
                            # Create a row representation
                            row_text = " | ".join(cell.get_text(strip=True) for cell in cells if cell.get_text(strip=True))
                            if row_text:  # Skip empty rows
                                table_data.append(row_text)
                                # Check for objective relevance using generated patterns
                                if objective_patterns and any(re.search(pattern, row_text, re.IGNORECASE) for pattern in objective_patterns):
                                    content["objective_related_info"].append(row_text)

                    # Add the whole table to the tables collection
                    if table_data:
                        # Try to create a more structured representation
                        if headers:
                            table_content = "TABLE HEADERS: " + " | ".join(headers) + "\n"
                        else:
                            table_content = "TABLE DATA:\n"
                        table_content += "\n".join(table_data)
                        content["tables"].append(table_content)

                # Extract lists
                info_lists = soup.find_all(["ul", "ol"])
                for lst in info_lists:
                    items = lst.find_all("li")
                    for item in items:
                        item_text = item.get_text(strip=True)
                        if objective_patterns and any(re.search(pattern, item_text, re.IGNORECASE) for pattern in objective_patterns):
                            content["objective_related_info"].append(item_text)

                # Extract paragraphs with relevant information
                paragraphs = soup.find_all("p")
                for p in paragraphs:
                    p_text = p.get_text(strip=True)
                    if objective_patterns and any(re.search(pattern, p_text, re.IGNORECASE) for pattern in objective_patterns):
                        content["objective_related_info"].append(p_text)

                # Get headings
                for heading in soup.find_all(["h1", "h2", "h3"]):
                    text = heading.get_text(strip=True)
                    if text:
                        content["headings"].append(text)
                        # Check if heading contains relevant information
                        if objective_patterns and any(re.search(pattern, text, re.IGNORECASE) for pattern in objective_patterns):
                            content["objective_related_info"].append(text)

                # Get paragraphs
                for p in soup.find_all("p"):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:  # Filter out short texts
                        content["paragraphs"].append(text)

                # Get lists
                for ul in soup.find_all(["ul", "ol"]):
                    items = [li.get_text(strip=True) for li in ul.find_all("li")]
                    if items:
                        content["lists"].append(items)

                # Look for contact information
                contact_patterns = [
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                    r"\b(?:\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",  # Phone
                    r"\b\d{1,5}\s+[A-Za-z\s,]+(?:Road|Street|Ave|Avenue|Blvd|Boulevard|Way|Drive|Lane|Place)\b",  # Address
                ]

                text_content = soup.get_text()
                for pattern in contact_patterns:
                    matches = re.findall(pattern, text_content)
                    if matches:
                        content["contact_info"].extend(matches)

                # Get all text content from the page for cases where structured extraction misses data
                all_text = soup.get_text(separator='\n', strip=True)
                
                # Filter out empty lines and very short lines
                meaningful_lines = [line.strip() for line in all_text.split('\n') 
                                  if line.strip() and len(line.strip()) > 2]
                
                # Combine all content, prioritizing structured content but falling back to all text
                structured_content = "\n".join(
                    content["headings"]
                    + content["paragraphs"]
                    + ["\n".join(lst) for lst in content["lists"]]
                    + content["tables"]  # Add table data to main_content
                    + content["objective_related_info"]
                )
                
                # If structured content is too short, use the full text content
                if len(structured_content) < 200:
                    content["main_content"] = "\n".join(meaningful_lines)
                else:
                    content["main_content"] = structured_content

                return content
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            ColoredLogger.error(f"Error scraping {url}: {str(e)}")
        return None
    
    def poll_firecrawl_result(self, extraction_id: str, interval: int = 10, max_attempts: int = 24) -> Optional[Dict[str, Any]]:
        """Poll Firecrawl API to get the extraction result."""
        url = f"https://api.firecrawl.dev/v1/extract/{extraction_id}"
        headers = {"Authorization": f"Bearer {self.firecrawl_api_key}"}

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Polling for extraction result (Attempt {attempt}/{max_attempts})...")
                ColoredLogger.info(f"Polling for extraction result (Attempt {attempt}/{max_attempts})...")
                
                response = requests.get(url, headers=headers, timeout=self.config.timeout_medium)
                response.raise_for_status()
                data = response.json()

                if data.get("success") and data.get("data"):
                    logger.info("Data successfully extracted")
                    ColoredLogger.info("Data successfully extracted")
                    return data["data"]
                elif data.get("success") and not data.get("data"):
                    logger.info(f"Still processing... (Attempt {attempt}/{max_attempts})")
                    ColoredLogger.info(f"Still processing... (Attempt {attempt}/{max_attempts})")
                    time.sleep(interval)
                else:
                    error_msg = data.get('error', 'No error message provided')
                    logger.error(f"API Error: {error_msg}")
                    ColoredLogger.error(f"API Error: {error_msg}")
                    return None

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {str(e)}")
                ColoredLogger.error(f"Request failed: {str(e)}")
                return None
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response: {str(e)}")
                ColoredLogger.error(f"Failed to parse response: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                ColoredLogger.error(f"Unexpected error: {str(e)}")
                return None

        logger.error("Max polling attempts reached. Extraction did not complete in time.")
        ColoredLogger.error("Max polling attempts reached. Extraction did not complete in time.")
        return None
    
    def extract_company_info(self, entity_name: str, url: str, objective: str, llm: str) -> Optional[Dict[str, Any]]:
        """Extract structured information from a single URL."""
        logger.debug(f"extract_company_info received llm: '{llm}', url: {url}") # DIAGNOSTIC LOG

        # Check if the single URL provided is valid before proceeding
        parsed_url_check = urlparse(url)
        if not all([parsed_url_check.scheme, parsed_url_check.netloc]):
            logger.error(f"Invalid URL provided for extraction: {url}")
            ColoredLogger.error(f"Invalid URL provided for extraction: {url}")
            return None

        logger.info(f"Attempting to extract info from: {url}")
        ColoredLogger.info(f"Attempting to extract info from: {url}")

        # Step 1: Scrape content from the URL with fallback strategy
        max_retries = 2
        scraped_content_dict = None

        for attempt in range(max_retries + 1):
            logger.info(f"Scraping attempt {attempt + 1}/{max_retries + 1} for {url}")
            ColoredLogger.info(f"Scraping attempt {attempt + 1}/{max_retries + 1} for {url}")

            scraped_content_dict = self.scrape_url_with_fallback(url, objective)

            if scraped_content_dict and scraped_content_dict.get("main_content"):
                content_length = len(scraped_content_dict["main_content"])
                logger.info(f"Successfully scraped content on attempt {attempt + 1}, length: {content_length}")
                break

            if attempt < max_retries:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                ColoredLogger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        if not scraped_content_dict or not scraped_content_dict.get("main_content"):
            logger.error(f"Failed to scrape any content from {url} after {max_retries + 1} attempts")
            ColoredLogger.error(f"Failed to scrape any content from {url} after {max_retries + 1} attempts")
            return None

        # Combine content for AI processing
        combined_content = scraped_content_dict["main_content"]

        # Include other structured elements found during scraping for better context
        # (Optional: Depending on how much context the LLM needs)
        # combined_content += "\n\nHeadings: " + ", ".join(scraped_content_dict['headings'])
        # combined_content += "\n\nLists: " + ", ".join(str(l) for l in scraped_content_dict['lists'])

        # Truncate content if necessary
        max_len = self.config.MAX_CONTENT_LENGTH
        if len(combined_content) > max_len:
            logger.warning(f"Content exceeds {max_len} characters. Truncating...")
            ColoredLogger.warning(f"Content exceeds {max_len} characters. Truncating...")
            combined_content = combined_content[:max_len]

        # Use AI to extract structured information
        try:
            logger.info("Processing scraped content with AI...")
            ColoredLogger.info("Processing scraped content with AI...")

            # Prepare a structured prompt with the scraped data
            content_summary = {
                "title": entity_name,
                "main_content": combined_content,
                "objective": objective,
                "tables": scraped_content_dict.get("tables", []),
                "objective_related_info": scraped_content_dict.get("objective_related_info", [])
            }

            # Debug: log payload size and timeout
            payload_size = len(json.dumps(content_summary))
            logger.info(f"Payload size: {payload_size} characters; timeout={self.config.timeout_long}s")
            ColoredLogger.info(f"Payload size: {payload_size} characters; timeout={self.config.timeout_long}s")

            messages = [
                {
                    "role": "system",
                    "content": f"""You are an information extraction expert. Extract structured information from the provided content with focus on the objective: "{objective}".

                    CRITICAL: The content includes HTML tables that contain pricing information. You MUST extract ALL pricing data from these tables.
                    Look for patterns like:
                    - TABLE HEADERS: MODEL | VARIANT | SRP
                    - TABLE DATA: BYD eMAX 7 | Superior Captain | Starts at Php 1,748,000

                    When you see table data, extract EVERY row that contains model names, variants, and prices.

                    Return it in the following JSON format:
                    
                    {{
                        "entity_overview": "Brief overview of the entity",
                        "products_and_services": [
                            {{
                                "name": "Product/Service name",
                                "description": "Brief description",
                                "price": "Price if available",
                                "features": ["Key features"],
                                "variants": [
                                    {{
                                        "variant_name": "Specific variant name",
                                        "variant_price": "Specific price for this variant"
                                    }}
                                ]
                            }}
                        ],
                        "contact_information": {{
                            "email": "Email if found",
                            "phone": "Phone if found",
                            "address": "Address if found"
                        }},
                        "key_features": ["List of key features/benefits"],
                        "unique_selling_points": ["List of unique selling points"],
                        "objective_related_information": {{
                            "summary": "Brief summary of information related to the user's objective",
                            "details": ["List of relevant details"],
                            "additional_notes": "Any additional important information",
                            "pricing_table": {{
                                "models": ["List of model names found in tables"],
                                "variants": ["List of variant options found in tables"],
                                "prices": ["List of prices found in tables"],
                                "model_variant_price_mapping": ["Example: 'Soluto L MT: 698,000'"]
                            }}
                        }}
                    }}
                    
                    Important:
                    1. Focus on extracting accurate information relevant to the objective: "{objective}"
                    2. MANDATORY: Extract ALL pricing data from tables - every model, variant, and price
                    3. Look for table patterns like "BYD eMAX 7 | Superior Captain | Starts at Php 1,748,000"
                    4. Parse table headers (MODEL, VARIANT, SRP) and map data correctly
                    5. For automotive pricing tables, extract: model names (e.g., "BYD eMAX 7"), variants (e.g., "Superior Captain"), and prices (e.g., "Php 1,748,000")
                    6. Include ALL table rows in the pricing_table section - do not skip any entries
                    7. Do not invent information not present in the source content
                    8. Be complete - missing pricing information is a critical error
                    """,
                },
                {
                    "role": "user",
                    "content": f"Extract information about {entity_name} from the following structured content, focusing on the objective: '{objective}':\n\n{json.dumps(content_summary, indent=2)}",
                },
            ]

            # Determine client and model based on llm setting
            if llm == "o4-mini":
                client_to_use = self.config.openai_client
                model_name = self.config.openai_model_name
            else: # Default to R1 (Deepseek)
                client_to_use = self.config.deepseek_client
                model_name = self.config.deepseek_model_name

            logger.info(f"Using model '{model_name}' for content extraction.")
            
            # Base parameters common to all models
            api_params = {
                "model": model_name,
                "messages": messages,
            }
            
            # Model-specific parameters
            if llm == "o4-mini":
                # o4-mini requires max_completion_tokens and doesn't support custom temperature
                api_params["max_completion_tokens"] = 3000  # Increased for comprehensive responses
                # No temperature parameter - o4-mini only supports default temperature (1.0)
            else:
                # Other models use max_tokens and support custom temperature
                api_params["max_tokens"] = 3000  # Increased for comprehensive responses
                api_params["temperature"] = 0.7
            
            try:
                response = client_to_use.chat.completions.create(**api_params)
                content = response.choices[0].message.content

                # Clean up the response - handle markdown code blocks more robustly
                logger.debug(f"Raw AI response length: {len(content)}")

                if "```json" in content:
                    # Extract JSON from markdown code block
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    if json_end != -1:
                        content = content[json_start:json_end]
                        logger.debug("Extracted JSON from ```json``` block")
                elif "```" in content:
                    # Handle generic code blocks
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    if json_end != -1:
                        content = content[json_start:json_end]
                        logger.debug("Extracted JSON from generic ``` block")

                # Remove any leading/trailing whitespace
                content = content.strip()
                logger.debug(f"Cleaned content length: {len(content)}")
                logger.debug(f"Content starts with: {content[:100]}...")

                # Additional cleanup for common issues
                if not content.startswith('{'):
                    # Try to find the first { character
                    first_brace = content.find('{')
                    if first_brace != -1:
                        content = content[first_brace:]
                        logger.debug("Trimmed content to start with {")

                if not content.endswith('}'):
                    # Try to find the last } character
                    last_brace = content.rfind('}')
                    if last_brace != -1:
                        content = content[:last_brace + 1]
                        logger.debug("Trimmed content to end with }")
                    else:
                        # If no closing brace found, try to fix incomplete JSON
                        logger.warning("No closing brace found, attempting to fix incomplete JSON")
                        # Count open braces and brackets and try to close them
                        open_braces = content.count('{') - content.count('}')
                        open_brackets = content.count('[') - content.count(']')

                        # Close any open arrays first, then objects
                        if open_brackets > 0:
                            content += ']' * open_brackets
                            logger.debug(f"Added {open_brackets} closing brackets")
                        if open_braces > 0:
                            content += '}' * open_braces
                            logger.debug(f"Added {open_braces} closing braces")

                try:
                    extracted_data = json.loads(content)
                    # Add the source URL to the result for this specific extraction
                    extracted_data['source_url'] = url
                    return extracted_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse AI response as JSON: {str(e)}")
                    logger.error(f"AI Response content (full): {content}")  # Log full content for debugging
                    ColoredLogger.error(f"Failed to parse AI response as JSON: {str(e)}")
                    ColoredLogger.error(f"AI Response content (full): {content}")
                    
                    # Try alternative parsing approaches
                    try:
                        # Try to find JSON more aggressively
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            json_only = json_match.group(0)
                            logger.info("Attempting alternative JSON extraction...")

                            # Try to fix common JSON issues
                            # Remove trailing commas before closing braces/brackets
                            json_only = re.sub(r',(\s*[}\]])', r'\1', json_only)

                            # Ensure proper closing
                            open_braces = json_only.count('{') - json_only.count('}')
                            open_brackets = json_only.count('[') - json_only.count(']')

                            if open_brackets > 0:
                                json_only += ']' * open_brackets
                                logger.debug(f"Added {open_brackets} closing brackets")
                            if open_braces > 0:
                                json_only += '}' * open_braces
                                logger.debug(f"Added {open_braces} closing braces")

                            extracted_data = json.loads(json_only)
                            extracted_data['source_url'] = url
                            logger.info("Successfully parsed JSON using enhanced alternative extraction")
                            return extracted_data
                    except Exception as alt_e:
                        logger.error(f"Alternative JSON parsing also failed: {str(alt_e)}")
                        logger.debug(f"Final JSON content: {json_only[:500] if 'json_only' in locals() else 'N/A'}...")
                    
                    return None

            except Exception as e:
                logger.error(f"Error processing content with AI: {str(e)}")
                ColoredLogger.error(f"Error processing content with AI: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Error extracting company information: {str(e)}")
            ColoredLogger.error(f"Error extracting company information: {str(e)}")
            return None

    async def scrape_url_with_playwright(self, url: str, objective: str = "") -> Optional[Dict[str, Any]]:
        """Scrape content using Playwright for JavaScript-heavy sites."""
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available, skipping JavaScript rendering")
            return None

        logger.info(f"Starting Playwright scrape for: {url}")
        ColoredLogger.info(f"Starting Playwright scrape for: {url}")

        try:
            async with async_playwright() as p:
                # Launch browser in headless mode
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-crash-reporter',
                        '--disable-breakpad',
                        '--disable-dev-shm-usage',
                        '--disable-gpu'
                    ]
                )

                # Create context with realistic settings
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
                )

                page = await context.new_page()

                # Set timeout
                page.set_default_timeout(self.config.PLAYWRIGHT_TIMEOUT)

                logger.debug(f"Navigating to {url}")
                # Navigate to the page
                await page.goto(url, wait_until='networkidle')

                # Wait for potential dynamic content
                await page.wait_for_timeout(3000)  # Wait 3 seconds for JS to load

                # Get page content
                html_content = await page.content()
                logger.debug(f"Retrieved HTML content, length: {len(html_content)}")

                await browser.close()

                # Parse with BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")

                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "iframe"]):
                    element.decompose()

                # Generate patterns for objective-based extraction
                objective_patterns = self._generate_patterns_from_objective(objective)

                # Extract structured content (same logic as scrape_url)
                content = {
                    "title": soup.title.string if soup.title else "",
                    "main_content": "",
                    "headings": [],
                    "paragraphs": [],
                    "lists": [],
                    "tables": [],
                    "contact_info": [],
                    "objective_related_info": [],
                    "url": url,
                    "scraped_with": "playwright"  # Mark as Playwright scraped
                }

                # Extract tables (enhanced for car pricing)
                tables = soup.find_all("table")
                for table in tables:
                    table_data = []
                    rows = table.find_all("tr")
                    headers = []
                    header_row = table.find("thead")
                    if header_row:
                        headers = [th.get_text(strip=True) for th in header_row.find_all("th")]

                    for row in rows:
                        cells = row.find_all(["td", "th"])
                        if cells:
                            row_text = " | ".join(cell.get_text(strip=True) for cell in cells if cell.get_text(strip=True))
                            if row_text:
                                table_data.append(row_text)
                                if objective_patterns and any(re.search(pattern, row_text, re.IGNORECASE) for pattern in objective_patterns):
                                    content["objective_related_info"].append(row_text)

                    if table_data:
                        if headers:
                            table_content = "TABLE HEADERS: " + " | ".join(headers) + "\n"
                        else:
                            table_content = "TABLE DATA:\n"
                        table_content += "\n".join(table_data)
                        content["tables"].append(table_content)

                # Extract other content similar to scrape_url method
                for heading in soup.find_all(["h1", "h2", "h3"]):
                    text = heading.get_text(strip=True)
                    if text:
                        content["headings"].append(text)
                        if objective_patterns and any(re.search(pattern, text, re.IGNORECASE) for pattern in objective_patterns):
                            content["objective_related_info"].append(text)

                for p in soup.find_all("p"):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:
                        content["paragraphs"].append(text)
                        if objective_patterns and any(re.search(pattern, text, re.IGNORECASE) for pattern in objective_patterns):
                            content["objective_related_info"].append(text)

                # Get all text content
                all_text = soup.get_text(separator='\n', strip=True)
                meaningful_lines = [line.strip() for line in all_text.split('\n')
                                  if line.strip() and len(line.strip()) > 2]

                # Combine content
                structured_content = "\n".join(
                    content["headings"]
                    + content["paragraphs"]
                    + ["\n".join(lst) for lst in content["lists"]]
                    + content["tables"]
                    + content["objective_related_info"]
                )

                if len(structured_content) < 200:
                    content["main_content"] = "\n".join(meaningful_lines)
                else:
                    content["main_content"] = structured_content

                logger.info(f"Successfully scraped with Playwright: {url}, content length: {len(content['main_content'])}")
                return content

        except Exception as e:
            logger.error(f"Playwright scraping error for {url}: {str(e)}")
            ColoredLogger.error(f"Playwright scraping error for {url}: {str(e)}")
            import traceback
            logger.debug(f"Playwright traceback: {traceback.format_exc()}")

        return None

    def scrape_url_with_fallback(self, url: str, objective: str = "") -> Optional[Dict[str, Any]]:
        """Scrape URL with requests first, fallback to Playwright if needed."""
        logger.info(f"Attempting to scrape {url} with fallback strategy")
        ColoredLogger.info(f"Attempting to scrape {url} with fallback strategy")

        # Try requests first (faster)
        content = self.scrape_url(url, objective)

        if content and content.get("main_content") and len(content["main_content"]) > 100:
            logger.info(f"Successfully scraped {url} with requests method")
            return content

        logger.warning(f"Requests method failed or returned insufficient content for {url}, trying Playwright...")
        ColoredLogger.warning(f"Requests method failed or returned insufficient content for {url}, trying Playwright...")

        # Fallback to Playwright
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.scrape_url_with_playwright(url, objective))
                    playwright_content = future.result(timeout=60)
            else:
                playwright_content = asyncio.run(self.scrape_url_with_playwright(url, objective))

            if playwright_content:
                logger.info(f"Successfully scraped {url} with Playwright fallback")
                return playwright_content
        except Exception as e:
            logger.error(f"Playwright fallback also failed for {url}: {str(e)}")
            ColoredLogger.error(f"Playwright fallback also failed for {url}: {str(e)}")

        logger.error(f"Both scraping methods failed for {url}")
        ColoredLogger.error(f"Both scraping methods failed for {url}")
        return None


class WebCrawler:
    """Main web crawler class that orchestrates the crawling process."""
    
    def __init__(self):
        """Initialize the web crawler."""
        self.config = ConfigManager()
        self.search_engine = SearchEngine(self.config)
        self.url_processor = URLProcessor(self.config)
        self.content_extractor = ContentExtractor(self.config)

    def _categorize_sites(self, urls: List[str], entity_name: str) -> Dict[str, List[str]]:
        """Categorize URLs by site type (official, retailer, dealer, review, etc.)."""
        categories = {
            'official': [],
            'retailer': [],
            'dealer': [],
            'review': [],
            'social': [],
            'news': [],
            'marketplace': [],
            'other': []
        }
        
        # Define patterns for different site types
        official_patterns = [
            'apple.com', 'microsoft.com', 'google.com', 'samsung.com', 'sony.com',
            'toyota.com', 'honda.com', 'nissan.com', 'hyundai.com', 'kia.com',
            'tesla.com', 'bmw.com', 'mercedes-benz.com', 'audi.com', 'volkswagen.com'
        ]
        
        retailer_patterns = [
            'beyondthebox.ph', 'istore.ph', 'powermaccenter.com', 'switch.com.ph',
            'lazada.com', 'shopee.ph', 'amazon.com', 'best-buy.com', 'newegg.com',
            'autodeal.com.ph', 'carmudi.com.ph', 'carousell.ph'
        ]
        
        dealer_patterns = [
            'dealer', 'dealership', 'showroom', 'branch', 'authorized',
            'service-center', 'repair-center'
        ]
        
        review_patterns = [
            'review', 'comparison', 'versus', 'vs', 'compare', 'rating',
            'cnet.com', 'techcrunch.com', 'the-verge.com', 'gsmarena.com',
            'topgear.ph', 'carguide.ph', 'auto-review'
        ]
        
        social_patterns = [
            'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'youtube.com', 'tiktok.com', 'pinterest.com', 'tumblr.com'
        ]
        
        news_patterns = [
            'news', 'press', 'media', 'cnn.com', 'bbc.com', 'reuters.com',
            'abs-cbn.com', 'gma.com', 'rappler.com', 'philstar.com'
        ]
        
        marketplace_patterns = [
            'marketplace', 'classifieds', 'buy-sell', 'second-hand',
            'olx.ph', 'facebook.com/marketplace'
        ]
        
        # Extract company keywords from entity name for official site detection
        entity_keywords = [word.lower() for word in re.findall(r'\w+', entity_name)]
        
        for url in urls:
            url_lower = url.lower()
            categorized = False
            
            # Check for official sites (company domain or entity keywords in domain)
            for pattern in official_patterns:
                if pattern in url_lower:
                    categories['official'].append(url)
                    categorized = True
                    break
            
            # Also check if entity keywords appear in domain (for company official sites)
            if not categorized:
                for keyword in entity_keywords:
                    if len(keyword) > 2:  # Skip short words
                        domain = urlparse(url).netloc.lower()
                        if keyword in domain:
                            categories['official'].append(url)
                            categorized = True
                            break
            
            if not categorized:
                # Check other categories
                if any(pattern in url_lower for pattern in social_patterns):
                    categories['social'].append(url)
                elif any(pattern in url_lower for pattern in retailer_patterns):
                    categories['retailer'].append(url)
                elif any(pattern in url_lower for pattern in dealer_patterns):
                    categories['dealer'].append(url)
                elif any(pattern in url_lower for pattern in review_patterns):
                    categories['review'].append(url)
                elif any(pattern in url_lower for pattern in news_patterns):
                    categories['news'].append(url)
                elif any(pattern in url_lower for pattern in marketplace_patterns):
                    categories['marketplace'].append(url)
                else:
                    categories['other'].append(url)
        
        # Log categorization results
        for category, urls_in_category in categories.items():
            if urls_in_category:
                logger.info(f"Categorized {len(urls_in_category)} URLs as '{category}': {urls_in_category[:2]}...")
                ColoredLogger.info(f"Categorized {len(urls_in_category)} URLs as '{category}': {urls_in_category[:2]}...")
        
        return categories

    def _extract_from_multiple_sites(self, categorized_urls: Dict[str, List[str]], 
                                   entity_name: str, objective: str, llm: str) -> Dict[str, Any]:
        """Extract data from multiple sites across different categories."""
        logger.info("Starting multi-site data extraction...")
        ColoredLogger.info("Starting multi-site data extraction...")
        
        all_extractions = {}
        extraction_summary = {
            'successful_extractions': 0,
            'failed_extractions': 0,
            'sites_by_category': {},
            'extraction_times': {}
        }
        
        # Define priority order for extraction
        category_priority = ['official', 'retailer', 'dealer', 'review', 'other', 'news', 'marketplace']
        
        # Extract from each category with limits
        category_limits = {
            'official': 3,    # Extract from up to 3 official sites
            'retailer': 5,    # Extract from up to 5 retailers (important for pricing comparison)
            'dealer': 3,      # Extract from up to 3 dealers
            'review': 2,      # Extract from up to 2 review sites
            'other': 2,       # Extract from up to 2 other sites
            'news': 1,        # Extract from 1 news site
            'marketplace': 2   # Extract from up to 2 marketplace sites
        }
        
        for category in category_priority:
            if category not in categorized_urls or not categorized_urls[category]:
                continue
                
            urls_in_category = categorized_urls[category]
            limit = category_limits.get(category, 2)
            urls_to_process = urls_in_category[:limit]
            
            logger.info(f"Processing {len(urls_to_process)} URLs from category '{category}'")
            ColoredLogger.info(f"Processing {len(urls_to_process)} URLs from category '{category}'")
            
            category_extractions = []
            
            # Process URLs in this category with concurrent extraction
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_url = {
                    executor.submit(
                        self.content_extractor.extract_company_info, 
                        entity_name, url, objective, llm
                    ): url for url in urls_to_process
                }
                
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    start_time = time.time()
                    
                    try:
                        extraction_result = future.result(timeout=120)  # 2-minute timeout per site
                        extraction_time = time.time() - start_time
                        
                        if extraction_result:
                            extraction_result['extraction_metadata'] = {
                                'site_category': category,
                                'extraction_time_seconds': round(extraction_time, 2),
                                'source_url': url
                            }
                            category_extractions.append(extraction_result)
                            extraction_summary['successful_extractions'] += 1
                            extraction_summary['extraction_times'][url] = round(extraction_time, 2)
                            
                            logger.info(f"Successfully extracted from {category} site: {url}")
                            ColoredLogger.info(f"Successfully extracted from {category} site: {url}")
                        else:
                            extraction_summary['failed_extractions'] += 1
                            logger.warning(f"Failed to extract from {category} site: {url}")
                            ColoredLogger.warning(f"Failed to extract from {category} site: {url}")
                            
                    except concurrent.futures.TimeoutError:
                        extraction_summary['failed_extractions'] += 1
                        logger.error(f"Timeout extracting from {category} site: {url}")
                        ColoredLogger.error(f"Timeout extracting from {category} site: {url}")
                    except Exception as e:
                        extraction_summary['failed_extractions'] += 1
                        logger.error(f"Error extracting from {category} site {url}: {str(e)}")
                        ColoredLogger.error(f"Error extracting from {category} site {url}: {str(e)}")
            
            if category_extractions:
                all_extractions[category] = category_extractions
                extraction_summary['sites_by_category'][category] = len(category_extractions)
        
        logger.info(f"Multi-site extraction completed. Success: {extraction_summary['successful_extractions']}, Failed: {extraction_summary['failed_extractions']}")
        ColoredLogger.info(f"Multi-site extraction completed. Success: {extraction_summary['successful_extractions']}, Failed: {extraction_summary['failed_extractions']}")
        
        return {
            'extractions_by_category': all_extractions,
            'extraction_summary': extraction_summary
        }
    
    def _parse_price_string(self, price_str: str) -> Optional[Tuple[float, str]]:
        """Parse a price string and return (numeric_value, currency)."""
        if not price_str or price_str.lower() in ['not specified', 'not available', 'n/a']:
            return None
        
        # Common currency symbols and codes
        currency_patterns = {
            r'[$]': 'USD',
            r'[₱]': 'PHP', 
            r'[€]': 'EUR',
            r'[£]': 'GBP',
            r'[¥]': 'JPY',
            r'\bUSD\b': 'USD',
            r'\bPHP\b': 'PHP',
            r'\bEUR\b': 'EUR'
        }
        
        # Extract currency
        currency = 'USD'  # default
        for pattern, curr in currency_patterns.items():
            if re.search(pattern, price_str, re.IGNORECASE):
                currency = curr
                break
        
        # Extract numeric value
        # Remove currency symbols and common text
        clean_price = re.sub(r'[₱$€£¥,\s]', '', price_str)
        clean_price = re.sub(r'\b(USD|PHP|EUR|GBP|JPY|from|starting|at)\b', '', clean_price, flags=re.IGNORECASE)
        
        # Find numeric value
        numbers = re.findall(r'\d+\.?\d*', clean_price)
        if numbers:
            try:
                return float(numbers[0]), currency
            except ValueError:
                pass
        
        return None
    
    def _normalize_to_usd(self, price: float, currency: str) -> float:
        """Convert price to USD for comparison (simplified rates)."""
        conversion_rates = {
            'USD': 1.0,
            'PHP': 0.018,  # Approximate PHP to USD
            'EUR': 1.1,    # Approximate EUR to USD
            'GBP': 1.25,   # Approximate GBP to USD
            'JPY': 0.0067  # Approximate JPY to USD
        }
        return price * conversion_rates.get(currency, 1.0)
    
    def _create_advanced_price_comparison(self, all_prices: List[Dict]) -> Dict[str, Any]:
        """Create comprehensive price comparison with advanced analysis."""
        price_comparison = {}
        
        for price_info in all_prices:
            product_name = price_info['product_name']
            if product_name not in price_comparison:
                price_comparison[product_name] = {
                    'prices_by_source': [],
                    'price_analysis': {
                        'lowest_price': None,
                        'highest_price': None,
                        'average_price': None,
                        'price_range_usd': None,
                        'currency_breakdown': {},
                        'retailer_count': 0,
                        'official_price': None
                    },
                    'price_recommendations': []
                }
            
            price_entry = {
                'price': price_info['price'],
                'source': price_info['source_domain'],
                'category': price_info['source_category'],
                'variants': price_info['variants']
            }
            
            # Parse and analyze price
            parsed_price = self._parse_price_string(price_info['price'])
            if parsed_price:
                numeric_price, currency = parsed_price
                price_entry['numeric_price'] = numeric_price
                price_entry['currency'] = currency
                price_entry['price_usd'] = self._normalize_to_usd(numeric_price, currency)
            
            price_comparison[product_name]['prices_by_source'].append(price_entry)
        
        # Perform price analysis for each product
        for product_name, data in price_comparison.items():
            valid_prices = [p for p in data['prices_by_source'] if 'price_usd' in p]
            
            if valid_prices:
                usd_prices = [p['price_usd'] for p in valid_prices]
                analysis = data['price_analysis']
                
                # Basic statistics
                analysis['lowest_price'] = min(usd_prices)
                analysis['highest_price'] = max(usd_prices)
                analysis['average_price'] = sum(usd_prices) / len(usd_prices)
                analysis['price_range_usd'] = analysis['highest_price'] - analysis['lowest_price']
                
                # Currency breakdown
                for price in valid_prices:
                    currency = price.get('currency', 'USD')
                    if currency not in analysis['currency_breakdown']:
                        analysis['currency_breakdown'][currency] = []
                    analysis['currency_breakdown'][currency].append(price['numeric_price'])
                
                # Count retailers
                retailers = [p for p in valid_prices if p.get('category') == 'retailer']
                analysis['retailer_count'] = len(retailers)
                
                # Find official price
                official_prices = [p for p in valid_prices if p.get('category') == 'official']
                if official_prices:
                    analysis['official_price'] = official_prices[0]['price_usd']
                
                # Generate recommendations
                recommendations = []
                
                # Best deal recommendation
                lowest_price_entry = min(valid_prices, key=lambda x: x['price_usd'])
                recommendations.append(f"Best price found at {lowest_price_entry['source']} ({lowest_price_entry['price']})")
                
                # Price range insight
                if analysis['price_range_usd'] > analysis['lowest_price'] * 0.2:  # 20% difference
                    recommendations.append(f"Significant price variation found (${analysis['price_range_usd']:.2f} range)")
                
                # Retailer comparison
                if analysis['retailer_count'] >= 2:
                    retailer_prices = [p['price_usd'] for p in retailers]
                    avg_retailer_price = sum(retailer_prices) / len(retailer_prices)
                    recommendations.append(f"Average retailer price: ${avg_retailer_price:.2f}")
                
                # Official vs retailer comparison
                if analysis['official_price'] and retailers:
                    min_retailer_price = min(p['price_usd'] for p in retailers)
                    if min_retailer_price < analysis['official_price']:
                        savings = analysis['official_price'] - min_retailer_price
                        recommendations.append(f"Save ${savings:.2f} by buying from retailers vs official")
                
                data['price_recommendations'] = recommendations
        
        return price_comparison

    def _aggregate_multi_site_data(self, multi_site_extractions: Dict[str, Any], 
                                 entity_name: str, objective: str) -> Dict[str, Any]:
        """Aggregate and analyze data from multiple sites."""
        logger.info("Aggregating multi-site data...")
        ColoredLogger.info("Aggregating multi-site data...")
        
        extractions_by_category = multi_site_extractions.get('extractions_by_category', {})
        
        aggregated_data = {
            'entity_name': entity_name,
            'objective': objective,
            'sites_analyzed': {},
            'consolidated_overview': '',
            'price_comparison': {},
            'feature_comparison': {},
            'contact_aggregation': {},
            'best_sources': {},
            'recommendations': []
        }
        
        all_products = []
        all_prices = []
        all_features = set()
        all_contacts = {}
        
        # Process each category
        for category, extractions in extractions_by_category.items():
            category_data = []
            
            for extraction in extractions:
                site_url = extraction.get('source_url', 'Unknown')
                domain = urlparse(site_url).netloc
                
                # Store site-specific data
                site_info = {
                    'url': site_url,
                    'domain': domain,
                    'category': category,
                    'extraction_time': extraction.get('extraction_metadata', {}).get('extraction_time_seconds', 0),
                    'data': extraction
                }
                category_data.append(site_info)
                
                # Collect products and pricing
                products = extraction.get('products_and_services', [])
                for product in products:
                    product_with_source = product.copy()
                    product_with_source['source_domain'] = domain
                    product_with_source['source_category'] = category
                    all_products.append(product_with_source)
                    
                    # Extract pricing information
                    if product.get('price'):
                        price_info = {
                            'product_name': product.get('name', ''),
                            'price': product.get('price', ''),
                            'source_domain': domain,
                            'source_category': category,
                            'variants': product.get('variants', [])
                        }
                        all_prices.append(price_info)
                
                # Collect features
                features = extraction.get('key_features', [])
                all_features.update(features)
                
                # Collect contact information
                contact_info = extraction.get('contact_information', {})
                if contact_info:
                    for key, value in contact_info.items():
                        if value and value != 'Not found' and value != 'null':
                            if key not in all_contacts:
                                all_contacts[key] = []
                            all_contacts[key].append({
                                'value': value,
                                'source_domain': domain,
                                'source_category': category
                            })
            
            aggregated_data['sites_analyzed'][category] = category_data
        
        # Create advanced price comparison with analysis
        if all_prices:
            price_comparison = self._create_advanced_price_comparison(all_prices)
            aggregated_data['price_comparison'] = price_comparison
        
        # Create feature comparison
        if all_features:
            aggregated_data['feature_comparison'] = {
                'common_features': list(all_features),
                'feature_frequency': {}
            }
        
        # Aggregate contact information
        aggregated_data['contact_aggregation'] = all_contacts
        
        # Determine best sources for different types of information
        best_sources = {}
        
        # Best for pricing (prioritize retailers)
        retailer_sites = aggregated_data['sites_analyzed'].get('retailer', [])
        if retailer_sites:
            best_sources['pricing'] = retailer_sites[0]['domain']
        
        # Best for official info (prioritize official sites)
        official_sites = aggregated_data['sites_analyzed'].get('official', [])
        if official_sites:
            best_sources['official_info'] = official_sites[0]['domain']
        
        # Best for reviews
        review_sites = aggregated_data['sites_analyzed'].get('review', [])
        if review_sites:
            best_sources['reviews'] = review_sites[0]['domain']
        
        aggregated_data['best_sources'] = best_sources
        
        # Generate recommendations
        recommendations = []
        
        if len(aggregated_data['sites_analyzed']) > 1:
            recommendations.append("Multi-site analysis completed - compare information across different source types")
        
        if retailer_sites and len(retailer_sites) > 1:
            recommendations.append("Multiple retailers found - compare pricing and availability")
        
        if official_sites:
            recommendations.append("Official information available - use for authoritative product details")
        
        if review_sites:
            recommendations.append("Review sites found - check for user feedback and comparisons")
        
        aggregated_data['recommendations'] = recommendations
        
        # Create consolidated overview
        total_sites = sum(len(sites) for sites in aggregated_data['sites_analyzed'].values())
        categories_found = list(aggregated_data['sites_analyzed'].keys())
        
        overview = f"Analyzed {total_sites} sites across {len(categories_found)} categories: {', '.join(categories_found)}. "
        if all_prices:
            overview += f"Found pricing information for {len(price_comparison)} products. "
        if all_contacts:
            overview += f"Collected contact information from multiple sources."
        
        aggregated_data['consolidated_overview'] = overview
        
        logger.info(f"Aggregation completed: {total_sites} sites, {len(categories_found)} categories")
        ColoredLogger.info(f"Aggregation completed: {total_sites} sites, {len(categories_found)} categories")
        
        return aggregated_data

    def _create_legacy_compatible_result(self, multi_site_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a legacy-compatible single result from multi-site data for backward compatibility."""
        if multi_site_result.get('extraction_mode') == 'single_site':
            return multi_site_result
        
        aggregated_data = multi_site_result.get('data', {})
        
        # Find the best official or retailer source for the primary data
        sites_analyzed = aggregated_data.get('sites_analyzed', {})
        
        primary_data = None
        primary_source = None
        
        # Priority: official > retailer > other
        for category in ['official', 'retailer', 'dealer', 'other']:
            if category in sites_analyzed and sites_analyzed[category]:
                primary_data = sites_analyzed[category][0]['data']
                primary_source = sites_analyzed[category][0]['domain']
                break
        
        if not primary_data:
            # Fallback: use any available data
            for category, sites in sites_analyzed.items():
                if sites:
                    primary_data = sites[0]['data']
                    primary_source = sites[0]['domain']
                    break
        
        # Create legacy format
        legacy_result = {
            "urls": multi_site_result.get('urls', []),
            "data": primary_data or {},
            "metadata": multi_site_result.get('metadata', {})
        }
        
        # Add multi-site summary to metadata
        if multi_site_result.get('metadata'):
            legacy_result['metadata']['multi_site_summary'] = {
                'total_sites_analyzed': aggregated_data.get('sites_analyzed', {}),
                'primary_source': primary_source,
                'extraction_mode': 'multi_site_legacy_compatible'
            }
        
        return legacy_result

    def _select_best_url_for_objective(self, urls: List[str], objective: str) -> str:
        """Select the best URL from the list based on the objective."""
        if not urls:
            return ""

        if not objective:
            return urls[0]

        objective_lower = objective.lower()

        # Priority keywords for different objectives
        priority_patterns = {
            'pricing': ['pricelist', 'price-list', 'pricing', 'prices', 'buy', 'shop'],
            'models': ['all-vehicles', 'vehicles', 'models', 'catalog', 'collections'],
            'compare': ['compare', 'comparison'],
            'contact': ['contact', 'dealership', 'dealers'],
            'about': ['about', 'company', 'overview']
        }

        # Special handling for Apple Watch - prioritize retailers for pricing
        if 'apple watch' in objective_lower or 'watch' in objective_lower:
            # If objective includes pricing, prioritize retailer sites first
            if any(word in objective_lower for word in ['price', 'pricing', 'cost']):
                # PowerMac Center often has the best pricing info, prioritize it
                for url in urls:
                    url_lower = url.lower()
                    if 'powermaccenter.com' in url_lower and 'watch' in url_lower:
                        logger.info(f"Selected PowerMac Center for Apple Watch pricing: {url}")
                        ColoredLogger.info(f"Selected PowerMac Center for Apple Watch pricing: {url}")
                        return url

                # Then try other retailers
                retailer_domains = ['beyondthebox.ph', 'istore.ph']
                for url in urls:
                    url_lower = url.lower()
                    if any(domain in url_lower for domain in retailer_domains) and '/watch' in url_lower:
                        logger.info(f"Selected Apple retailer URL for pricing objective '{objective}': {url}")
                        ColoredLogger.info(f"Selected Apple retailer URL for pricing objective '{objective}': {url}")
                        return url

            # Then prioritize specific Apple Watch URLs
            for url in urls:
                url_lower = url.lower()
                if '/watch' in url_lower and ('apple.com' in url_lower or 'collections/apple-watch' in url_lower):
                    logger.info(f"Selected Apple Watch specific URL for objective '{objective}': {url}")
                    ColoredLogger.info(f"Selected Apple Watch specific URL for objective '{objective}': {url}")
                    return url

        # Determine objective type
        objective_type = None
        if any(word in objective_lower for word in ['price', 'pricing', 'cost']):
            objective_type = 'pricing'
        elif any(word in objective_lower for word in ['model', 'vehicle', 'car']):
            objective_type = 'models'
        elif any(word in objective_lower for word in ['compare', 'comparison']):
            objective_type = 'compare'
        elif any(word in objective_lower for word in ['contact', 'dealer']):
            objective_type = 'contact'
        elif any(word in objective_lower for word in ['about', 'company']):
            objective_type = 'about'

        # Find URLs that match the objective type
        if objective_type and objective_type in priority_patterns:
            patterns = priority_patterns[objective_type]
            for pattern in patterns:
                for url in urls:
                    if pattern in url.lower():
                        logger.info(f"Selected URL for objective '{objective}': {url}")
                        ColoredLogger.info(f"Selected URL for objective '{objective}': {url}")
                        return url

        # If no specific match found, prioritize certain domains over social media
        # Prioritize official websites and authorized retailers over social media
        priority_domains = ['autodeal.com', 'tesla.com', 'kia.com', 'toyota.com', 'honda.com', 'nissan.com',
                           'beyondthebox.ph', 'istore.ph', 'powermaccenter.com', 'apple.com']
        social_media_domains = ['instagram.com', 'facebook.com', 'twitter.com', 'tiktok.com']

        # For Apple Watch specifically, prioritize retailers that show pricing
        if 'apple watch' in objective.lower() or 'watch' in objective.lower():
            retailer_domains = ['beyondthebox.ph', 'istore.ph', 'powermaccenter.com']
            for url in urls:
                if any(domain in url.lower() for domain in retailer_domains):
                    logger.info(f"Selected Apple retailer URL for pricing: {url}")
                    ColoredLogger.info(f"Selected Apple retailer URL for pricing: {url}")
                    return url

        # First try to find a priority domain
        for url in urls:
            if any(domain in url.lower() for domain in priority_domains):
                logger.info(f"Selected priority domain URL for objective '{objective}': {url}")
                ColoredLogger.info(f"Selected priority domain URL for objective '{objective}': {url}")
                return url

        # If still no URL found, use first non-social media URL
        for url in urls:
            if not any(domain in url.lower() for domain in social_media_domains):
                logger.info(f"Selected non-social media URL for objective '{objective}': {url}")
                ColoredLogger.info(f"Selected non-social media URL for objective '{objective}': {url}")
                return url

        # Last resort: use first available URL
        logger.info(f"No specific URL match found for objective '{objective}', using first URL: {urls[0]}")
        ColoredLogger.info(f"No specific URL match found for objective '{objective}', using first URL: {urls[0]}")
        return urls[0]

    def _crawl_single_page(self, url: str, objective: str, llm: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crawl a single page with optional infinite scroll support.
        """
        try:
            start_time = time.time()
            logger.info(f"Single page crawl mode for: {url}")
            ColoredLogger.info(f"Single page crawl mode for: {url}")

            # If infinite scroll is enabled, use the specialized method
            if config.get("enable_infinite_scroll", False):
                logger.info("Infinite scroll enabled")
                ColoredLogger.info("Infinite scroll enabled")
                return self._crawl_with_infinite_scroll(url, objective, llm, config)

            # Otherwise, use existing single-page logic by calling the original implementation
            # Set crawl_config to None to use default behavior
            result = self.crawl_website(url, objective, llm, crawl_config={"crawl_mode": "default"})
            return result

        except Exception as e:
            logger.error(f"Error in single page crawl: {str(e)}")
            ColoredLogger.error(f"Error in single page crawl: {str(e)}")
            return {"error": str(e)}

    def _crawl_deep(self, url: str, objective: str, llm: str, config: Dict[str, Any], progress_callback: callable = None) -> Dict[str, Any]:
        """
        Perform deep crawl following links up to specified depth.
        """
        try:
            start_time = time.time()
            max_depth = config.get("max_depth", 2)
            max_pages = config.get("max_pages", 20)
            crawl_delay = config.get("crawl_delay", 2)
            same_domain = config.get("same_domain", True)

            logger.info(f"Deep crawl: depth={max_depth}, max_pages={max_pages}, delay={crawl_delay}s")
            ColoredLogger.info(f"Deep crawl: depth={max_depth}, max_pages={max_pages}, delay={crawl_delay}s")

            # Ensure URL has protocol
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            from urllib.parse import urlparse, urljoin
            parsed_base = urlparse(url)
            base_domain = parsed_base.netloc

            # Track visited URLs and queue
            visited = set()
            queue = [(url, 0)]  # (url, depth)
            pages_scraped = 0
            all_extractions = []

            while queue and pages_scraped < max_pages:
                current_url, depth = queue.pop(0)

                if depth > max_depth or current_url in visited:
                    continue

                logger.info(f"Crawling [{pages_scraped + 1}/{max_pages}] depth {depth}: {current_url}")
                ColoredLogger.info(f"Crawling [{pages_scraped + 1}/{max_pages}] depth {depth}: {current_url}")

                # Progress callback
                if progress_callback:
                    progress_callback(current_url, f"Crawling page {pages_scraped + 1}/{max_pages}: {current_url}")

                try:
                    # Scrape the current page
                    page_data = self.content_extractor.extract_company_info(
                        url,  # entity name
                        current_url,
                        objective or "",
                        llm
                    )

                    if page_data:
                        all_extractions.append({
                            "url": current_url,
                            "depth": depth,
                            "data": page_data
                        })

                    visited.add(current_url)
                    pages_scraped += 1

                    # Extract links if not at max depth
                    if depth < max_depth:
                        links = self.url_processor.discover_internal_links(
                            current_url,
                            objective or "",
                            max_links=20
                        )

                        # Filter links by domain if same_domain is enabled
                        for link_url, link_title in links:
                            if link_url not in visited:
                                if same_domain:
                                    parsed_link = urlparse(link_url)
                                    if parsed_link.netloc == base_domain:
                                        queue.append((link_url, depth + 1))
                                else:
                                    queue.append((link_url, depth + 1))

                    # Respect crawl delay
                    time.sleep(crawl_delay)

                except Exception as e:
                    logger.error(f"Error scraping {current_url}: {str(e)}")
                    ColoredLogger.error(f"Error scraping {current_url}: {str(e)}")
                    continue

            end_time = time.time()
            logger.info(f"Deep crawl completed: {pages_scraped} pages in {end_time - start_time:.2f} seconds")
            ColoredLogger.info(f"Deep crawl completed: {pages_scraped} pages in {end_time - start_time:.2f} seconds")

            # Aggregate all extractions
            return {
                "urls": list(visited),
                "data": {
                    "pages_scraped": pages_scraped,
                    "max_depth_reached": max([e["depth"] for e in all_extractions]) if all_extractions else 0,
                    "extractions": all_extractions
                },
                "metadata": {
                    "crawl_time": datetime.now().isoformat(),
                    "execution_time_seconds": round(end_time - start_time, 2),
                    "crawl_mode": "deep",
                    "config": config
                }
            }

        except Exception as e:
            logger.error(f"Error in deep crawl: {str(e)}")
            ColoredLogger.error(f"Error in deep crawl: {str(e)}")
            return {"error": str(e)}

    def _crawl_sitemap(self, url: str, objective: str, llm: str, config: Dict[str, Any], progress_callback: callable = None) -> Dict[str, Any]:
        """
        Crawl using sitemap.xml to discover URLs.
        """
        try:
            start_time = time.time()
            sitemap_url = config.get("sitemap_url", "")
            max_pages = config.get("max_pages", 20)
            crawl_delay = config.get("crawl_delay", 2)

            logger.info(f"Sitemap crawl mode")
            ColoredLogger.info(f"Sitemap crawl mode")

            # Ensure URL has protocol
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            from urllib.parse import urlparse, urljoin
            import xml.etree.ElementTree as ET
            import requests

            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"

            # Try to find sitemap
            sitemap_locations = []
            if sitemap_url:
                sitemap_locations.append(sitemap_url)
            else:
                # Try common sitemap locations
                sitemap_locations = [
                    urljoin(base_url, "/sitemap.xml"),
                    urljoin(base_url, "/sitemap_index.xml"),
                    urljoin(base_url, "/sitemap-index.xml"),
                ]

            sitemap_urls = []
            for sitemap_loc in sitemap_locations:
                try:
                    logger.info(f"Trying sitemap: {sitemap_loc}")
                    response = requests.get(sitemap_loc, timeout=10)
                    if response.status_code == 200:
                        # Parse XML
                        root = ET.fromstring(response.content)
                        # Handle namespace
                        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                        # Find all <loc> tags
                        for loc in root.findall('.//ns:loc', ns):
                            if loc.text:
                                sitemap_urls.append(loc.text)

                        if sitemap_urls:
                            logger.info(f"Found {len(sitemap_urls)} URLs in sitemap")
                            ColoredLogger.info(f"Found {len(sitemap_urls)} URLs in sitemap")
                            break
                except Exception as e:
                    logger.debug(f"Could not fetch sitemap from {sitemap_loc}: {str(e)}")
                    continue

            if not sitemap_urls:
                logger.warning("No sitemap found, falling back to regular crawl")
                ColoredLogger.warning("No sitemap found, falling back to regular crawl")
                return self._crawl_single_page(url, objective, llm, config)

            # Limit URLs to max_pages
            urls_to_scrape = sitemap_urls[:max_pages]

            # Scrape each URL
            all_extractions = []
            for idx, page_url in enumerate(urls_to_scrape):
                logger.info(f"Scraping [{idx + 1}/{len(urls_to_scrape)}]: {page_url}")
                ColoredLogger.info(f"Scraping [{idx + 1}/{len(urls_to_scrape)}]: {page_url}")

                # Progress callback
                if progress_callback:
                    progress_callback(page_url, f"Scraping page {idx + 1}/{len(urls_to_scrape)}: {page_url}")

                try:
                    page_data = self.content_extractor.extract_company_info(
                        url,  # entity name
                        page_url,
                        objective or "",
                        llm
                    )

                    if page_data:
                        all_extractions.append({
                            "url": page_url,
                            "data": page_data
                        })

                    time.sleep(crawl_delay)

                except Exception as e:
                    logger.error(f"Error scraping {page_url}: {str(e)}")
                    continue

            end_time = time.time()
            logger.info(f"Sitemap crawl completed: {len(all_extractions)} pages in {end_time - start_time:.2f} seconds")
            ColoredLogger.info(f"Sitemap crawl completed: {len(all_extractions)} pages in {end_time - start_time:.2f} seconds")

            return {
                "urls": urls_to_scrape,
                "data": {
                    "pages_scraped": len(all_extractions),
                    "extractions": all_extractions
                },
                "metadata": {
                    "crawl_time": datetime.now().isoformat(),
                    "execution_time_seconds": round(end_time - start_time, 2),
                    "crawl_mode": "sitemap",
                    "config": config
                }
            }

        except Exception as e:
            logger.error(f"Error in sitemap crawl: {str(e)}")
            ColoredLogger.error(f"Error in sitemap crawl: {str(e)}")
            return {"error": str(e)}

    def _crawl_pattern_based(self, url: str, objective: str, llm: str, config: Dict[str, Any], progress_callback: callable = None) -> Dict[str, Any]:
        """
        Crawl following URLs that match specified patterns.
        """
        try:
            start_time = time.time()
            url_pattern = config.get("url_pattern", "")
            exclude_pattern = config.get("exclude_pattern", "")
            max_pages = config.get("max_pages", 20)
            crawl_delay = config.get("crawl_delay", 2)

            logger.info(f"Pattern-based crawl: include='{url_pattern}', exclude='{exclude_pattern}'")
            ColoredLogger.info(f"Pattern-based crawl: include='{url_pattern}', exclude='{exclude_pattern}'")

            # Ensure URL has protocol
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            # Discover links
            discovered_links = self.url_processor.discover_internal_links(
                url,
                objective or "",
                max_links=100
            )

            # Filter by patterns
            filtered_urls = []
            for link_url, link_title in discovered_links:
                # Check include pattern
                if url_pattern and not re.search(url_pattern, link_url):
                    continue
                # Check exclude pattern
                if exclude_pattern and re.search(exclude_pattern, link_url):
                    continue
                filtered_urls.append(link_url)

            logger.info(f"Found {len(filtered_urls)} URLs matching patterns")
            ColoredLogger.info(f"Found {len(filtered_urls)} URLs matching patterns")

            if not filtered_urls:
                logger.warning("No URLs matched the patterns")
                ColoredLogger.warning("No URLs matched the patterns")
                return {"error": "No URLs matched the specified patterns"}

            # Limit to max_pages
            urls_to_scrape = filtered_urls[:max_pages]

            # Scrape each URL
            all_extractions = []
            for idx, page_url in enumerate(urls_to_scrape):
                logger.info(f"Scraping [{idx + 1}/{len(urls_to_scrape)}]: {page_url}")
                ColoredLogger.info(f"Scraping [{idx + 1}/{len(urls_to_scrape)}]: {page_url}")

                # Progress callback
                if progress_callback:
                    progress_callback(page_url, f"Scraping page {idx + 1}/{len(urls_to_scrape)}: {page_url}")

                try:
                    page_data = self.content_extractor.extract_company_info(
                        url,  # entity name
                        page_url,
                        objective or "",
                        llm
                    )

                    if page_data:
                        all_extractions.append({
                            "url": page_url,
                            "data": page_data
                        })

                    time.sleep(crawl_delay)

                except Exception as e:
                    logger.error(f"Error scraping {page_url}: {str(e)}")
                    continue

            end_time = time.time()
            logger.info(f"Pattern-based crawl completed: {len(all_extractions)} pages in {end_time - start_time:.2f} seconds")
            ColoredLogger.info(f"Pattern-based crawl completed: {len(all_extractions)} pages in {end_time - start_time:.2f} seconds")

            return {
                "urls": urls_to_scrape,
                "data": {
                    "pages_scraped": len(all_extractions),
                    "extractions": all_extractions
                },
                "metadata": {
                    "crawl_time": datetime.now().isoformat(),
                    "execution_time_seconds": round(end_time - start_time, 2),
                    "crawl_mode": "pattern",
                    "config": config
                }
            }

        except Exception as e:
            logger.error(f"Error in pattern-based crawl: {str(e)}")
            ColoredLogger.error(f"Error in pattern-based crawl: {str(e)}")
            return {"error": str(e)}

    def _crawl_with_infinite_scroll(self, url: str, objective: str, llm: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrape single page with infinite scroll support using Playwright.
        """
        try:
            start_time = time.time()
            max_scrolls = config.get("max_scrolls", 10)
            scroll_delay = config.get("scroll_delay", 2000)
            scroll_step = config.get("scroll_step", 1000)
            stability_checks = config.get("content_stability_checks", 3)
            youtube_optimized = config.get("youtube_optimized", True)
            human_behavior = config.get("human_behavior_simulation", True)

            logger.info(f"Infinite scroll: max_scrolls={max_scrolls}, delay={scroll_delay}ms, step={scroll_step}px")
            ColoredLogger.info(f"Infinite scroll: max_scrolls={max_scrolls}, delay={scroll_delay}ms, step={scroll_step}px")

            # Check if playwright is available
            if not PLAYWRIGHT_AVAILABLE:
                logger.warning("Playwright not available, falling back to regular scrape")
                ColoredLogger.warning("Playwright not available, falling back to regular scrape")
                # Fall back to regular scrape
                return self.crawl_website(entity_name, objective, llm, crawl_config=None)

            # Ensure URL has protocol
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            from playwright.sync_api import sync_playwright
            import random

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="networkidle")

                previous_height = page.evaluate("document.body.scrollHeight")
                no_change_count = 0

                for i in range(max_scrolls):
                    # Apply human behavior randomization
                    if human_behavior:
                        actual_delay = scroll_delay + random.randint(-200, 200)
                        actual_step = scroll_step + random.randint(-100, 100)
                    else:
                        actual_delay = scroll_delay
                        actual_step = scroll_step

                    logger.info(f"Scroll {i + 1}/{max_scrolls}")

                    # Scroll down
                    page.evaluate(f"window.scrollBy(0, {actual_step})")
                    page.wait_for_timeout(actual_delay)

                    # Check if content has loaded
                    current_height = page.evaluate("document.body.scrollHeight")

                    if current_height == previous_height:
                        no_change_count += 1
                        logger.debug(f"No height change ({no_change_count}/{stability_checks})")
                        if no_change_count >= stability_checks:
                            logger.info("Content stable, stopping scroll")
                            ColoredLogger.info("Content stable, stopping scroll")
                            break
                    else:
                        no_change_count = 0

                    previous_height = current_height

                # Get final page content
                html_content = page.content()
                browser.close()

            logger.info("Scrolling complete, extracting content...")
            ColoredLogger.info("Scrolling complete, extracting content...")

            # Use the content extractor on the scrolled content
            # For now, we'll use the regular extraction method
            page_data = self.content_extractor.extract_company_info(
                url,  # entity name
                url,
                objective or "",
                llm
            )

            end_time = time.time()
            logger.info(f"Infinite scroll completed in {end_time - start_time:.2f} seconds")
            ColoredLogger.info(f"Infinite scroll completed in {end_time - start_time:.2f} seconds")

            return {
                "urls": [url],
                "data": page_data,
                "metadata": {
                    "crawl_time": datetime.now().isoformat(),
                    "execution_time_seconds": round(end_time - start_time, 2),
                    "crawl_mode": "single_with_infinite_scroll",
                    "scrolls_performed": i + 1,
                    "config": config
                }
            }

        except Exception as e:
            logger.error(f"Error in infinite scroll: {str(e)}")
            ColoredLogger.error(f"Error in infinite scroll: {str(e)}")
            # Fallback to regular single page scrape
            logger.info("Falling back to regular single page scrape")
            return self._crawl_single_page(url, objective, llm, {"crawl_mode": "default", "enable_infinite_scroll": False})

    def crawl_website(self, entity_name: str, objective: str = None, llm: str = "R1", crawl_config: Dict[str, Any] = None, progress_callback: callable = None) -> Dict[str, Any]:
        """
        Crawl a website based on company name or a specific URL with advanced configuration.

        Args:
            entity_name (str): The name of the company/entity or a specific URL to crawl.
            objective (str, optional): The objective for the crawl. Defaults to None.
            llm (str, optional): The language model to use. Defaults to "R1".
            crawl_config (dict, optional): Advanced crawl configuration options. Defaults to None.
            progress_callback (callable, optional): Callback function for progress updates. Should accept (url, progress_text). Defaults to None.
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
                return self._crawl_deep(entity_name, objective, llm, crawl_config, progress_callback)
            elif crawl_mode == "sitemap":
                return self._crawl_sitemap(entity_name, objective, llm, crawl_config, progress_callback)
            elif crawl_mode == "pattern":
                return self._crawl_pattern_based(entity_name, objective, llm, crawl_config, progress_callback)
            else:
                logger.error(f"Unknown crawl mode: {crawl_mode}, falling back to default")
                ColoredLogger.error(f"Unknown crawl mode: {crawl_mode}, falling back to default")
                # Fall through to original logic below

            start_time = time.time()
            urls_to_scrape = []
            entity_for_prompt = entity_name  # Keep original for prompt if it's a name
            is_input_url = False  # Flag to track if input is a URL
            specified_domain = None  # Store domain for filtering

            # Check if the input is a URL or domain
            domain_pattern = r"^[\w\-]+(\.[\w\-]+)+$"
            is_domain_input = re.match(domain_pattern, entity_name) and " " not in entity_name
            
            if validators.url(entity_name) or is_domain_input:
                # URL input mode
                url_input = entity_name if entity_name.startswith(("http://", "https://")) else "https://" + entity_name
                
                logger.info(f"Input is a URL: {url_input}. Crawling directly.")
                ColoredLogger.info(f"Input is a URL: {url_input}. Crawling directly.")
                
                is_input_url = True
                
                # Extract domain from URL for filtering and prompt
                parsed_url = urlparse(url_input)
                specified_domain = parsed_url.netloc
                domain_parts = specified_domain.split(".")
                
                # Get company name from domain
                if len(domain_parts) > 1:
                    entity_for_prompt = domain_parts[-2]
                else:
                    entity_for_prompt = entity_name
                
                # Discover and select relevant internal links
                logger.info(f"Discovering internal links for {url_input}...")
                ColoredLogger.info(f"Discovering internal links for {url_input}...")
                
                discovered_links = self.url_processor.discover_internal_links(
                    url_input, 
                    objective or "", 
                    max_links=50
                )
                
                # Create data structure similar to search results for AI selection
                serp_like = [
                    {"title": t or "", "link": u, "snippet": ""}
                    for u, t in discovered_links
                ]
                
                # Add the root URL if not already included
                root_entry = {"title": entity_for_prompt, "link": url_input, "snippet": ""}
                if root_entry not in serp_like:
                    serp_like.insert(0, root_entry)
                
                logger.info("Selecting most relevant URLs using AI...")
                ColoredLogger.info("Selecting most relevant URLs using AI...")
                
                selected_internal = self.url_processor.select_urls_with_ai(
                    entity_for_prompt, 
                    objective or "", 
                    serp_like,
                    llm
                )
                
                urls_to_scrape = selected_internal[:5] if selected_internal else [url_input]
                
            else:
                # Company/entity name mode
                logger.info(f"Input is an entity name: {entity_name}. Searching...")
                ColoredLogger.info(f"Input is an entity name: {entity_name}. Searching...")
                
                search_results = self.search_engine.search_comprehensive(entity_name, objective or "")
                if not search_results:
                    logger.error(f"No search results found for {entity_name}")
                    ColoredLogger.error(f"No search results found for {entity_name}")
                    return {"error": f"No search results found for {entity_name}"}
                
                logger.info("Search completed. Selecting relevant URLs...")
                ColoredLogger.info("Search completed. Selecting relevant URLs...")
                
                selected_urls = self.url_processor.select_urls_with_ai(
                    entity_name, 
                    objective or "", 
                    search_results,
                    llm
                )
                
                if not selected_urls:
                    logger.error("AI failed to select URLs.")
                    ColoredLogger.error("AI failed to select URLs.")
                    return {"error": "AI failed to select URLs"}
                
                logger.info(f"URLs selected: {selected_urls}")
                ColoredLogger.info(f"URLs selected: {selected_urls}")
                
                urls_to_scrape = selected_urls
                
                # For each official website found, look for more relevant internal pages
                official_roots = []
                social_domains = [
                    "facebook.com", "twitter.com", "instagram.com", "linkedin.com", 
                    "youtube.com", "tiktok.com", "pinterest.com", "tumblr.com"
                ]
                
                for url in selected_urls:
                    try:
                        parsed = urlparse(url)
                        if not parsed.netloc or any(sd in parsed.netloc for sd in social_domains):
                            continue
                        root_url = f"{parsed.scheme}://{parsed.netloc}"
                        if root_url not in official_roots:
                            official_roots.append(root_url)
                    except Exception:
                        continue
                
                # Find relevant internal pages for each official website
                for root in official_roots:
                    discovered = self.url_processor.discover_internal_links(
                        root, 
                        objective or "", 
                        max_links=30
                    )
                    
                    if discovered:
                        serp_like_internal = [
                            {"title": t or "", "link": u, "snippet": ""}
                            for u, t in discovered
                        ]
                        
                        # Select the most relevant internal pages using AI
                        selected_internal = self.url_processor.select_urls_with_ai(
                            entity_name,
                            objective or "",
                            serp_like_internal,
                            llm
                        )
                        
                        # Add selected internal pages to scrape list
                        urls_to_scrape.extend(selected_internal[:3])  # Limit to top 3 per root
                
                # Remove duplicates
                urls_to_scrape = list(dict.fromkeys(urls_to_scrape))
            
            # Validate all URLs before proceeding
            validated_urls = self.url_processor.validate_urls(urls_to_scrape)
            if not validated_urls:
                logger.error("No valid URLs found to scrape after filtering.")
                ColoredLogger.error("No valid URLs found to scrape after filtering.")
                return {"error": "No valid URLs found to scrape"}
            
            # Use validated URLs moving forward
            urls_to_scrape = validated_urls
            
            # Determine entity name for prompt (handle URL input case)
            entity_for_prompt = entity_name if not is_input_url else url_input
            
            logger.info(f"Starting extraction process for {entity_for_prompt}...")
            ColoredLogger.info(f"Starting extraction process for {entity_for_prompt}...")
            
            # Set LLM model based on user selection
            self.config.llm_model = llm
            logger.info(f"Using LLM: {llm}")
            if llm == "o4-mini" and self.config.openai_client:
                self.url_processor.client = self.config.openai_client
                self.content_extractor.client = self.config.openai_client
                logger.info("Assigned OpenAI client.")
            else:
                self.url_processor.client = self.config.deepseek_client
                self.content_extractor.client = self.config.deepseek_client
                logger.info("Assigned Deepseek client.")
            
            # Categorize URLs by site type
            categorized_urls = self._categorize_sites(urls_to_scrape, entity_name)
            
            # Determine extraction mode based on number of sites and categories
            total_urls = sum(len(urls) for urls in categorized_urls.values())
            categories_with_urls = [cat for cat, urls in categorized_urls.items() if urls]
            
            logger.info(f"Found {total_urls} URLs across {len(categories_with_urls)} categories")
            ColoredLogger.info(f"Found {total_urls} URLs across {len(categories_with_urls)} categories")
            
            # Multi-site extraction - Always trigger for company names, or when multiple sites/categories found
            if (not is_input_url and total_urls > 0) or (total_urls > 1 and len(categories_with_urls) > 1):
                logger.info("Performing multi-site extraction...")
                ColoredLogger.info("Performing multi-site extraction...")
                
                # Extract from multiple sites
                multi_site_extractions = self._extract_from_multiple_sites(
                    categorized_urls, entity_name, objective or "", llm
                )
                
                # Aggregate the results
                aggregated_data = self._aggregate_multi_site_data(
                    multi_site_extractions, entity_name, objective or ""
                )
                
                end_time = time.time()
                logger.info(f"Multi-site extraction completed in {end_time - start_time:.2f} seconds.")
                ColoredLogger.info(f"Multi-site extraction completed in {end_time - start_time:.2f} seconds.")
                
                # Structure the result with both legacy format and new multi-site format
                final_result = {
                    "urls": urls_to_scrape,
                    "data": aggregated_data,  # New multi-site aggregated data
                    "multi_site_data": multi_site_extractions,  # Raw extractions by category
                    "site_categories": categorized_urls,  # URL categorization
                    "extraction_mode": "multi_site",
                    "metadata": {
                        "crawl_time": datetime.now().isoformat(),
                        "execution_time_seconds": round(end_time - start_time, 2),
                        "objective": objective,
                        "entity_name": entity_name,
                        "input_type": "url" if is_input_url else "entity_name",
                        "sites_analyzed": total_urls,
                        "categories_found": categories_with_urls
                    }
                }
                
            else:
                # Single-site extraction (fallback for backward compatibility)
                logger.info("Performing single-site extraction (legacy mode)...")
                ColoredLogger.info("Performing single-site extraction (legacy mode)...")
                
                # Select the best URL for extraction based on objective
                best_url = self._select_best_url_for_objective(urls_to_scrape, objective or "")

                # Extract information
                extracted_data = self.content_extractor.extract_company_info(
                    entity_for_prompt,
                    best_url,
                    objective or "",
                    llm
                )
                
                end_time = time.time()
                logger.info(f"Single-site extraction completed in {end_time - start_time:.2f} seconds.")
                ColoredLogger.info(f"Single-site extraction completed in {end_time - start_time:.2f} seconds.")
                
                if not extracted_data:
                    logger.error("Failed to extract information.")
                    ColoredLogger.error("Failed to extract information.")
                    return {"error": "Failed to extract information"}
                
                # Legacy format for backward compatibility
                final_result = {
                    "urls": urls_to_scrape,
                    "data": extracted_data,
                    "site_categories": categorized_urls,  # Still include categorization
                    "extraction_mode": "single_site",
                    "metadata": {
                        "crawl_time": datetime.now().isoformat(),
                        "execution_time_seconds": round(end_time - start_time, 2),
                        "objective": objective,
                        "entity_name": entity_name,
                        "input_type": "url" if is_input_url else "entity_name",
                        "sites_analyzed": 1,
                        "categories_found": categories_with_urls
                    }
                }
            
            logger.info("Crawl successful.")
            ColoredLogger.info("Crawl successful.")
            
            return final_result
            
        except Exception as e:
            logger.error(f"An error occurred in crawl_website: {str(e)}")
            ColoredLogger.error(f"An error occurred in crawl_website: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"An internal error occurred: {str(e)}"}


def main():
    """Command-line interface for the web crawler."""
    print(f"{ColoredLogger.CYAN}Welcome to the Web Information Crawler{ColoredLogger.RESET}")
    print(f"{ColoredLogger.CYAN}You can enter a company name, website URL, or domain{ColoredLogger.RESET}")

    entity = input(f"{ColoredLogger.BLUE}Enter the company name or URL: {ColoredLogger.RESET}")
    if not entity:
        print(f"{ColoredLogger.RED}Input cannot be empty.{ColoredLogger.RESET}")
        return

    objective = input(f"{ColoredLogger.BLUE}Enter what information you want to find: {ColoredLogger.RESET}")
    if not objective:
        print(f"{ColoredLogger.RED}Search objective cannot be empty.{ColoredLogger.RESET}")
        return

    # Create and run the crawler
    crawler = WebCrawler()
    result = crawler.crawl_website(entity, objective)

    if result and "error" not in result:
        print(f"{ColoredLogger.GREEN}Extraction completed successfully.{ColoredLogger.RESET}")
        
        # Save the results to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/crawl_results_{timestamp}.json"
        
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)
            
        print(f"{ColoredLogger.GREEN}Results saved to {filename}{ColoredLogger.RESET}")
        
    else:
        error_message = result.get("error", "Unknown error occurred") if result else "Unknown error occurred"
        print(f"{ColoredLogger.RED}Failed to extract the requested information: {error_message}. Try refining your input or choosing a different entity/URL.{ColoredLogger.RESET}")


if __name__ == "__main__":
    nest_asyncio.apply()
    main()
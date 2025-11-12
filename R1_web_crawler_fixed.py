import os
import json
import time
import requests
from dotenv import load_dotenv
import openai
import re
import sys
from urllib.parse import urlparse, urljoin
import concurrent.futures
from bs4 import BeautifulSoup
import urllib3
import validators  # Add this import
from datetime import datetime
from serpapi import GoogleSearch
import random

# Global set to track URLs we've seen
urls_seen = set()

# Define a standard user agent header for requests
USER_AGENT_HEADER = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


# ANSI color codes
class Colors:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def clean_query(query):
    """
    Clean a search query by removing URLs and special characters.

    Args:
        query (str): The query to clean

    Returns:
        str: Cleaned query string
    """
    # If query looks like a URL, extract domain name
    if "." in query and "/" in query or query.startswith("www."):
        try:
            # Extract domain from URL
            parsed = urlparse(
                query
                if query.startswith(("http://", "https://"))
                else f"http://{query}"
            )
            domain = parsed.netloc or parsed.path
            # Remove www. if present
            domain = re.sub(r"^www\.", "", domain)
            # Remove TLD
            domain = domain.split(".")[0]
            return f"{domain} official website"
        except Exception:
            pass

    # Otherwise just clean the query
    cleaned = re.sub(r"[^\w\s]", " ", query)  # Replace special chars with spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()  # Normalize whitespace
    return cleaned


# Load environment variables
load_dotenv()

# Initialize clients
print(f"{Colors.YELLOW}Debug: Initializing Deepseek client...{Colors.RESET}")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    print(f"{Colors.RED}Error: DEEPSEEK_API_KEY not found in .env file{Colors.RESET}")
    sys.exit(1)
print(
    f"{Colors.YELLOW}Debug: Deepseek API key loaded: {'Yes' if deepseek_api_key else 'No'}{Colors.RESET}"
)

client = openai.Client(
    api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1", timeout=120.0
)

firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
serp_api_key = os.getenv("SERP_API_KEY")

# Debug: Print API key status (not the actual key)
print(
    f"{Colors.YELLOW}Debug: SERP API key loaded: {'Yes' if serp_api_key else 'No'}{Colors.RESET}"
)

# Disable SSL warnings for requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def search_google(query):
    """
    Search Google using the SerpAPI.

    Args:
        query (str): The search query

    Returns:
        list: List of search result dictionaries
    """
    api_key = os.getenv("SERPAPI_KEY")

    if not api_key:
        print(
            f"{Colors.RED}No SERPAPI_KEY found in environment variables{Colors.RESET}"
        )
        return []

    try:
        cleaned_query = clean_query(query)
        if not cleaned_query:
            print(
                f"{Colors.RED}Invalid search query. Please provide a valid company name or URL.{Colors.RESET}"
            )
            return []

        print(f"{Colors.YELLOW}Searching Google for '{cleaned_query}'...{Colors.RESET}")

        params = {"engine": "google", "q": cleaned_query, "api_key": api_key, "num": 10}

        search = GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            print(f"{Colors.RED}SerpAPI error: {results['error']}{Colors.RESET}")
            return []

        if "organic_results" not in results:
            print(
                f"{Colors.YELLOW}No organic results found for query: {cleaned_query}{Colors.RESET}"
            )
            return []

        return results["organic_results"]

    except Exception as e:
        print(f"{Colors.RED}Error searching Google: {str(e)}{Colors.RESET}")
        return []


def select_urls_with_r1(company, objective, serp_results):
    """
    Use R1 to select the most relevant URLs from SERP results for the given company and objective.
    Returns a list of URLs.
    """
    try:
        print(f"{Colors.YELLOW}Debug: Preparing data for R1...{Colors.RESET}")
        # Prepare the data for R1
        serp_data = [
            {
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
            }
            for r in serp_results
            if r.get("link")
        ]

        # Add specific pricing pages if they exist
        pricing_keywords = [
            "price",
            "pricing",
            "pricelist",
            "price-list",
            "models",
            "vehicles",
        ]
        for result in serp_data:
            link = result.get("link", "").lower()
            if any(keyword in link for keyword in pricing_keywords):
                print(
                    f"{Colors.GREEN}Found pricing-specific page: {link}{Colors.RESET}"
                )

        print(
            f"{Colors.YELLOW}Debug: Created serp_data with {len(serp_data)} entries{Colors.RESET}"
        )

        messages = [
            {
                "role": "system",
                "content": """You are a URL selector that always responds with valid JSON. You select URLs from the SERP results relevant to the company and objective. 
                Prioritize official websites and pages containing pricing information. Your response must be a JSON object with a 'selected_urls' array property containing strings.
                If you find pricing-specific pages (containing 'price', 'pricing', 'pricelist', etc.), ALWAYS include them.""",
            },
            {
                "role": "user",
                "content": (
                    f"Company: {company}\n"
                    f"Objective: {objective}\n"
                    f"SERP Results: {json.dumps(serp_data)}\n\n"
                    "Return a JSON object with a property 'selected_urls' that contains an array "
                    "of URLs most likely to help meet the objective. Prioritize pages with pricing information. "
                    'For example: {"selected_urls": ["https://example.com/pricing", "https://example.com"]}'
                ),
            },
        ]

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )

        result = response.choices[0].message.content

        try:
            if result.startswith("```json"):
                result = result[7:-3]

            parsed_result = json.loads(result)
            urls = parsed_result.get("selected_urls", [])

            # Add pricing-specific pages if not already included
            additional_urls = []
            for result in serp_data:
                link = result.get("link", "").lower()
                if (
                    any(keyword in link for keyword in pricing_keywords)
                    and link not in urls
                ):
                    additional_urls.append(result.get("link"))

            urls.extend(additional_urls)
            return list(dict.fromkeys(urls))  # Remove duplicates while preserving order

        except json.JSONDecodeError as e:
            print(f"{Colors.RED}Error parsing R1 response: {str(e)}{Colors.RESET}")
            return []

    except Exception as e:
        print(f"{Colors.RED}Error selecting URLs with R1: {e}{Colors.RESET}")
        return []


def validate_url(url):
    """Validate if a URL is accessible."""
    try:
        # Add http:// if no scheme is present
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Parse URL to ensure it's valid
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return None

        # First try with SSL verification
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                return response.url
        except requests.exceptions.SSLError as ssl_error:
            print(
                f"{Colors.YELLOW}Warning: SSL verification failed for {url}. Attempting without verification...{Colors.RESET}"
            )
            # If SSL verification fails, try without verification but with a warning
            try:
                response = requests.head(
                    url, timeout=10, allow_redirects=True, verify=False
                )
                if response.status_code == 200:
                    print(
                        f"{Colors.YELLOW}Warning: Accessed {url} without SSL verification. The connection is not secure.{Colors.RESET}"
                    )
                    return response.url
            except Exception as e:
                print(
                    f"{Colors.RED}Error accessing URL without SSL verification: {str(e)}{Colors.RESET}"
                )
                return None
        except Exception as e:
            print(f"{Colors.RED}Error accessing URL: {str(e)}{Colors.RESET}")
            return None

        return None
    except Exception as e:
        print(
            f"{Colors.YELLOW}Warning: Could not access URL {url}: {str(e)}{Colors.RESET}"
        )
        return None


def validate_urls(urls):
    """Validate multiple URLs in parallel."""
    print(f"{Colors.YELLOW}Validating URLs...{Colors.RESET}")
    valid_urls = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(validate_url, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                valid_url = future.result()
                if valid_url:
                    print(f"{Colors.GREEN}URL validated: {valid_url}{Colors.RESET}")
                    valid_urls.append(valid_url)
                else:
                    print(f"{Colors.RED}Invalid URL: {url}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}Error validating {url}: {str(e)}{Colors.RESET}")
    return valid_urls


def scrape_url(url):
    """Scrape content from a URL directly."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Try to resolve common URL redirects for Kia website
        if (
            "kia.com.ph" in url
            and not url.endswith("price-list.html")
            and "/shopping-tools/" not in url
        ):
            pricelist_url = (
                "https://www.kia.com/ph/shopping-tools/request-a-price-list.html"
            )
            print(
                f"{Colors.GREEN}Trying to access Kia price list page directly: {pricelist_url}{Colors.RESET}"
            )
            try:
                price_response = requests.get(
                    pricelist_url, verify=False, timeout=30, headers=headers
                )
                if price_response.status_code == 200:
                    urls_to_scrape = [url, pricelist_url]  # Add both URLs
                    all_content = []

                    for current_url in urls_to_scrape:
                        print(
                            f"{Colors.YELLOW}Scraping content from: {current_url}{Colors.RESET}"
                        )
                        current_response = requests.get(
                            current_url, verify=False, timeout=30, headers=headers
                        )
                        if current_response.status_code == 200:
                            content = process_page_content(
                                current_response, current_url
                            )
                            if content:
                                all_content.append(content)

                    # Combine all content
                    if all_content:
                        combined_content = {
                            "title": all_content[0].get("title", ""),
                            "main_content": "\n".join(
                                c.get("main_content", "") for c in all_content
                            ),
                            "headings": sum(
                                (c.get("headings", []) for c in all_content), []
                            ),
                            "paragraphs": sum(
                                (c.get("paragraphs", []) for c in all_content), []
                            ),
                            "lists": sum((c.get("lists", []) for c in all_content), []),
                            "contact_info": sum(
                                (c.get("contact_info", []) for c in all_content), []
                            ),
                            "pricing_info": sum(
                                (c.get("pricing_info", []) for c in all_content), []
                            ),
                        }
                        return combined_content
            except Exception as e:
                print(
                    f"{Colors.YELLOW}Error accessing Kia price list page: {str(e)}, continuing with original URL{Colors.RESET}"
                )

        # Proceed with original URL if price list page couldn't be accessed
        response = requests.get(url, verify=False, timeout=30, headers=headers)
        if response.status_code == 200:
            content = process_page_content(response, url)

            # Extract internal links for potential future crawling
            soup = BeautifulSoup(response.text, "html.parser")
            domain = urlparse(url).netloc

            for link in soup.find_all("a", href=True):
                href = link["href"]

                # Skip empty links, javascript, and anchors
                if not href or href.startswith(("javascript:", "#", "mailto:", "tel:")):
                    continue

                # Convert relative URLs to absolute
                if not href.startswith(("http://", "https://")):
                    if href.startswith("/"):
                        parsed_base = urlparse(url)
                        href = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
                    else:
                        href = urljoin(url, href)

                # Add the URL to a global set that could be used for further crawling
                if domain in urlparse(href).netloc and href not in urls_seen:
                    urls_seen.add(href)

            return content
    except Exception as e:
        print(f"{Colors.RED}Error scraping {url}: {str(e)}{Colors.RESET}")
    return None


def process_page_content(response, url):
    """Process the HTML content of a page and extract structured data"""
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "footer", "iframe"]):
        element.decompose()

    # Extract structured content
    content = {
        "title": soup.title.string if soup.title else "",
        "main_content": "",
        "headings": [],
        "paragraphs": [],
        "lists": [],
        "contact_info": [],
        "pricing_info": [],
    }

    # Special handling for Kia price list page
    if "kia.com.ph" in url and ("/shopping-tools/" in url or "price-list" in url):
        print(
            f"{Colors.GREEN}Detected Kia price list page, applying specialized extraction{Colors.RESET}"
        )
        # Look for table elements which likely contain the pricing information
        tables = soup.find_all("table")
        for table in tables:
            # Process each row in the table
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if (
                    len(cells) >= 2
                ):  # Ensure there are at least 2 columns (model and price)
                    model_cell = cells[0].get_text(strip=True) if len(cells) > 0 else ""
                    price_cell = (
                        cells[-1].get_text(strip=True) if len(cells) > 1 else ""
                    )

                    if (
                        model_cell
                        and price_cell
                        and (
                            "P" in price_cell
                            or "₱" in price_cell
                            or "SRP" in price_cell
                        )
                    ):
                        price_info = f"{model_cell} - {price_cell}"
                        content["pricing_info"].append(price_info)
                        print(
                            f"{Colors.GREEN}Found pricing info: {price_info}{Colors.RESET}"
                        )

    # Extract pricing information using various methods
    # Method 1: Look for price patterns in text
    price_patterns = [
        r"(?:PHP|₱|Php|P)\s*[\d,]+(?:\.\d{2})?",  # PHP currency with optional decimals
        r"(?:PHP|₱|Php|P)\s*[\d,]+K",  # Prices in thousands
        r"(?:PHP|₱|Php|P)\s*[\d,]+M",  # Prices in millions
        r"\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:PHP|₱|Php|P)",  # Numbers followed by currency
        r"Price[s]?\s+(?:starts?|begins?|from)?\s+(?:at|from)?\s+(?:PHP|₱|Php|P|P\s)\s*[\d,]+",  # Price starts at format
        r"(?:PHP|₱|Php|P|P\s)\s*[\d,]+(?:\.\d{2})?\s+[-–]\s+(?:PHP|₱|Php|P|P\s)\s*[\d,]+",  # Price range format
        r"(?:SRP|RRP):\s*(?:PHP|₱|Php|P|P\s)\s*[\d,]+",  # SRP format
    ]

    # Method 1.5: Look for car model patterns
    car_model_patterns = [
        r"(?:Kia\s+)?[A-Za-z0-9\s\-\+]+(?:\s+\d{1,2}\.\d|\s+(?:LX|EX|SX|GT))",  # Car model with variant
        r"(?:Kia\s+)?(?:Seltos|Sonet|Carnival|K2500|EV6|EV9|Sorento|Soluto|Sportage|Stinger|Picanto|Rio|Forte)",  # Known Kia models
        r"(?:Kia\s+)?[A-Za-z0-9]+\s+(?:1\.4|1\.5|1\.6|2\.0|2\.2|2\.5|3\.3)\s+(?:AT|MT|DCT|Diesel|Turbo)",  # Model with engine spec
    ]

    # Method 2: Look for pricing tables
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            row_text = " ".join(cell.get_text(strip=True) for cell in cells)
            # Check if row contains price information
            if any(re.search(pattern, row_text) for pattern in price_patterns):
                content["pricing_info"].append(row_text)
            # Also check if row contains car model information
            elif any(re.search(pattern, row_text) for pattern in car_model_patterns):
                content["pricing_info"].append(row_text)

    # Method 3: Look for price lists
    price_lists = soup.find_all(["ul", "ol"])
    for lst in price_lists:
        items = lst.find_all("li")
        for item in items:
            item_text = item.get_text(strip=True)
            # Check if item contains price information
            if any(re.search(pattern, item_text) for pattern in price_patterns):
                content["pricing_info"].append(item_text)
            # Also check if item contains car model information
            elif any(re.search(pattern, item_text) for pattern in car_model_patterns):
                content["pricing_info"].append(item_text)
            # If item contains "Kia" and any of price/model/variant keywords
            elif "kia" in item_text.lower() and any(
                x in item_text.lower() for x in ["price", "model", "variant"]
            ):
                content["pricing_info"].append(item_text)

    # Method 4: Look for price in paragraphs
    paragraphs = soup.find_all("p")
    for p in paragraphs:
        p_text = p.get_text(strip=True)
        # Check if paragraph contains price information
        if any(re.search(pattern, p_text) for pattern in price_patterns):
            content["pricing_info"].append(p_text)
        # Also check if paragraph contains car model information
        elif any(re.search(pattern, p_text) for pattern in car_model_patterns):
            content["pricing_info"].append(p_text)
        # If paragraph contains "Kia" and any of price/model/variant keywords
        elif "kia" in p_text.lower() and any(
            x in p_text.lower() for x in ["price", "model", "variant"]
        ):
            content["pricing_info"].append(p_text)

    # Method 5: Check for headings with pricing info (often used for model headlines with prices)
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        h_text = heading.get_text(strip=True)
        if h_text:
            content["headings"].append(h_text)
            # Check for price and model information in headings (common on Kia website)
            if any(re.search(pattern, h_text) for pattern in price_patterns) or any(
                re.search(pattern, h_text) for pattern in car_model_patterns
            ):
                content["pricing_info"].append(h_text)

            # Also check for specific Kia model and price combinations in headings
            if any(
                model.lower() in h_text.lower()
                for model in [
                    "seltos",
                    "sonet",
                    "carnival",
                    "ev6",
                    "ev9",
                    "sorento",
                    "soluto",
                ]
            ) and re.search(r"P\s*[\d,]+", h_text):
                content["pricing_info"].append(h_text)

    # Method 6: Look for div elements with specific pricing content
    pricing_divs = soup.find_all(
        "div",
        class_=lambda c: c
        and any(x in c.lower() for x in ["price", "srp", "cost", "variant"]),
    )
    for div in pricing_divs:
        div_text = div.get_text(strip=True)
        if div_text:
            content["pricing_info"].append(div_text)

    # Get regular content elements
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

    # Look for specific car model and price combinations
    model_price_pattern = r"((?:Kia\s+)?[A-Za-z0-9\s\-\+]+)(?:\s*[-–:]\s*|\s+)((?:PHP|₱|Php|P|Price:?)\s*[\d,\.]+(?:\s*[KkMmBb])?)"
    model_price_matches = re.findall(model_price_pattern, text_content)
    if model_price_matches:
        for match in model_price_matches:
            content["pricing_info"].append(f"{match[0]} - {match[1]}")

    # Combine all content
    content["main_content"] = "\n".join(
        content["headings"]
        + content["paragraphs"]
        + ["\n".join(lst) for lst in content["lists"]]
        + content["pricing_info"]  # Include pricing information in main content
    )

    return content


def extract_company_info(urls, prompt, company, api_key, is_input_url, objective=None):
    """
    Use requests to call Firecrawl's extract endpoint with selected URLs.

    Args:
        urls (list): List of URLs to extract information from
        prompt (str): Prompt for the extraction
        company (str): Company name or domain
        api_key (str): API key for Firecrawl
        is_input_url (bool): Whether the input was a URL
        objective (str, optional): The user's search objective

    Returns:
        dict: Extracted information
    """
    print(
        f"{Colors.YELLOW}Extracting structured data from the provided URLs using Firecrawl...{Colors.RESET}"
    )

    # Validate URLs before proceeding
    valid_urls = validate_urls(urls)
    if not valid_urls:
        print(
            f"{Colors.RED}No valid URLs found to extract information from.{Colors.RESET}"
        )
        return None

    # If we're dealing with kia.com.ph, try to add the price list page if it's not already there
    if any("kia.com.ph" in url for url in valid_urls):
        price_list_url = (
            "https://www.kia.com/ph/shopping-tools/request-a-price-list.html"
        )
        if price_list_url not in valid_urls:
            print(
                f"{Colors.GREEN}Adding Kia price list URL: {price_list_url}{Colors.RESET}"
            )
            valid_urls.append(price_list_url)

    # First try Firecrawl
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "urls": valid_urls,
            "prompt": prompt + " for " + company,
            "enableWebSearch": not is_input_url,
        }

        response = requests.post(
            "https://api.firecrawl.dev/v1/extract",
            headers=headers,
            json=payload,
            timeout=30,
        )

        data = response.json()
        if data.get("success"):
            extraction_id = data.get("id")
            if extraction_id:
                result = poll_firecrawl_result(extraction_id, api_key)
                if result:
                    # Check if we have car model data
                    if "carModels" in result and not result["carModels"]:
                        print(
                            f"{Colors.YELLOW}No car models found in Firecrawl result, will try direct scraping{Colors.RESET}"
                        )
                    else:
                        return result
    except Exception as e:
        print(
            f"{Colors.YELLOW}Firecrawl extraction failed, falling back to direct scraping: {str(e)}{Colors.RESET}"
        )

    # If Firecrawl fails or returns empty car models, fall back to direct scraping with the new generic system
    print(
        f"{Colors.YELLOW}Attempting direct web scraping with industry-specific extractors...{Colors.RESET}"
    )

    # Track extracted data by industry
    all_extracted_data = {}
    domain = urlparse(valid_urls[0]).netloc if valid_urls else company

    # Detect if we're dealing with an automotive domain
    is_automotive = False
    if objective and any(
        kw in objective.lower() for kw in ["car", "vehicle", "automotive", "model"]
    ):
        is_automotive = True

    # For each URL, use the appropriate extractor based on the industry
    for url in valid_urls:
        # Check if URL is likely to contain valuable information
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        # Skip URLs that are unlikely to contain car model information
        skip_paths = [
            "/blog/",
            "/news/",
            "/about/",
            "/contact/",
            "/terms/",
            "/privacy/",
            "/dealership/",
            "/careers/",
            "/press/",
            "/support/",
            "/login/",
            "/register/",
            "/account/",
            "/community/",
            "/forum/",
            "/faq/",
        ]

        if any(skip_path in path for skip_path in skip_paths) and not (
            "/pricelist" in path or "/price-list" in path
        ):
            print(
                f"{Colors.YELLOW}Skipping URL unlikely to contain valuable data: {url}{Colors.RESET}"
            )
            continue

        if is_automotive:
            # Use the traditional car model extraction for automotive sites
            # This ensures backward compatibility
            content = scrape_url(url)
            if content:
                if "all_content" not in all_extracted_data:
                    all_extracted_data["all_content"] = []
                    all_extracted_data["headings"] = []
                    all_extracted_data["paragraphs"] = []
                    all_extracted_data["lists"] = []
                    all_extracted_data["contact_info"] = set()
                    all_extracted_data["pricing_info"] = []
                    all_extracted_data["car_models"] = []

                all_extracted_data["all_content"].append(content["main_content"])
                all_extracted_data["headings"].extend(content["headings"])
                all_extracted_data["paragraphs"].extend(content["paragraphs"])
                all_extracted_data["lists"].extend(content["lists"])
                all_extracted_data["contact_info"].update(content["contact_info"])
                all_extracted_data["pricing_info"].extend(content["pricing_info"])

                # Also try to extract car models directly
                models = extract_car_models_with_api(url, domain)
                if models:
                    print(
                        f"{Colors.GREEN}Found {len(models)} car models from {url}{Colors.RESET}"
                    )
                    if "car_models" not in all_extracted_data:
                        all_extracted_data["car_models"] = []
                    all_extracted_data["car_models"].extend(models)
        else:
            # For non-automotive sites, use the new generic extractor
            extracted_data = extract_generic_data(url, domain, objective or "")
            if extracted_data:
                data_type = extracted_data.get("type", "generic")
                if data_type not in all_extracted_data:
                    all_extracted_data[data_type] = extracted_data
                else:
                    # Merge data of the same type
                    for key, value in extracted_data.items():
                        if key == "type":
                            continue
                        elif isinstance(value, list):
                            if key not in all_extracted_data[data_type]:
                                all_extracted_data[data_type][key] = value
                            else:
                                all_extracted_data[data_type][key].extend(value)
                        elif isinstance(value, dict):
                            if key not in all_extracted_data[data_type]:
                                all_extracted_data[data_type][key] = value
                            else:
                                all_extracted_data[data_type][key].update(value)

    if not all_extracted_data:
        print(f"{Colors.RED}Failed to extract any data from the URLs{Colors.RESET}")
        return None

    # If we're dealing with automotive data, process it using the existing logic
    if is_automotive and (
        "all_content" in all_extracted_data or "car_models" in all_extracted_data
    ):
        car_models = []

        # Process car models that were directly extracted
        if "car_models" in all_extracted_data and all_extracted_data["car_models"]:
            car_models.extend(all_extracted_data["car_models"])

        # Process pricing info if we have it
        if "pricing_info" in all_extracted_data:
            price_pattern = (
                r"(?:PHP|₱|Php|P|Price:?)\s*([\d,]+(?:\.\d{2})?(?:\s*[KkMmBb])?)"
            )
            model_price_pattern = r"((?:Kia\s+)?[A-Za-z0-9\s\-\+]+)(?:\s*[-–:]\s*|\s+)((?:PHP|₱|Php|P|Price:?)\s*[\d,\.]+(?:\s*[KkMmBb])?)"
            model_variant_pattern = r"(Kia\s+)?([A-Za-z0-9]+)(?:\s+([A-Za-z0-9\.\s]+))?"

            for info in all_extracted_data["pricing_info"]:
                # For price list page format: "MODEL - VARIANT - SRP"
                if " - " in info and ("P " in info or "₱" in info):
                    parts = info.split(" - ")
                    if len(parts) >= 2:
                        model_name = parts[0].strip()
                        price = parts[-1].strip()
                        if (
                            model_name
                            and price
                            and not any(
                                m.get("modelName") == model_name for m in car_models
                            )
                        ):
                            car_models.append({"modelName": model_name, "price": price})
                            continue

                # Try to extract model and price together
                model_price_matches = re.findall(model_price_pattern, info)
                if model_price_matches:
                    for match in model_price_matches:
                        model_name = match[0].strip()
                        price = match[1].strip()
                        if (
                            model_name
                            and price
                            and not any(
                                m.get("modelName") == model_name for m in car_models
                            )
                        ):
                            car_models.append({"modelName": model_name, "price": price})
                            continue

                # If no model-price pairs found, look for prices
                price_matches = re.findall(price_pattern, info)
                if price_matches:
                    price = "₱" + price_matches[0].strip()
                    # Try to find a model name in the same string
                    model_match = re.search(
                        r"((?:Kia\s+)?[A-Za-z0-9\s\-\+]+)(?:\s+\d{1,2}\.\d|\s+(?:LX|EX|SX|GT))",
                        info,
                    )
                    if model_match:
                        model_name = model_match.group(1).strip()
                        if model_name and not any(
                            m.get("modelName") == model_name for m in car_models
                        ):
                            car_models.append({"modelName": model_name, "price": price})

        # Try to extract more information using Deepseek
        try:
            print(
                f"{Colors.YELLOW}Processing automotive data with Deepseek...{Colors.RESET}"
            )

            # Structure the extracted data for processing
            content_summary = {
                "title": company,
                "headings": all_extracted_data.get("headings", [])[:10],
                "main_content": "\n".join(all_extracted_data.get("paragraphs", [])[:5]),
                "lists": all_extracted_data.get("lists", [])[:5],
                "contact_info": list(all_extracted_data.get("contact_info", [])),
                "pricing_info": all_extracted_data.get("pricing_info", []),
            }

            messages = [
                {
                    "role": "system",
                    "content": """You are an information extraction expert. Extract structured information from the provided content and return it in the following JSON format:
                    {
                        "company_overview": "Brief overview of the company",
                        "products_and_services": [
                            {
                                "name": "Product/Service name",
                                "description": "Brief description",
                                "price": "Price if available",
                                "features": ["Key features"]
                            }
                        ],
                        "contact_information": {
                            "email": "Email if found",
                            "phone": "Phone if found",
                            "address": "Address if found"
                        },
                        "key_features": ["List of key features/benefits"],
                        "unique_selling_points": ["List of unique selling points"],
                        "pricing_information": {
                            "currency": "PHP",
                            "products": [
                                {
                                    "name": "Product name",
                                    "model": "Model information if available",
                                    "base_price": "Base price",
                                    "variants": [
                                        {
                                            "name": "Variant name",
                                            "price": "Variant price",
                                            "features": ["Variant specific features"]
                                        }
                                    ]
                                }
                            ],
                            "promotions": ["Any ongoing promotions or special offers"],
                            "payment_terms": ["Available payment and financing options"],
                            "additional_fees": ["Any additional charges or fees"]
                        },
                        "carModels": []
                    }
                    
                    Important:
                    1. Focus on extracting accurate pricing information
                    2. Include all prices found in the content
                    3. Organize variants and models clearly
                    4. Include any financing or payment options
                    5. Note any special offers or promotions
                    6. Only extract car models that are actual vehicle models, not menu items or section names
                    7. Exclude entries like "More", "View All", "Choose color", etc.
                    """,
                },
                {
                    "role": "user",
                    "content": f"Extract information about {company} from the following structured content, paying special attention to pricing information and car models:\n\n{json.dumps(content_summary, indent=2)}",
                },
            ]

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
            )

            result = response.choices[0].message.content

            # Clean up the response
            if result.startswith("```json"):
                result = result[7:-3]  # Remove ```json and ``` markers

            try:
                extracted_data = json.loads(result)

                # Add directly scraped car models to the result
                if car_models:
                    # If we already have carModels, merge them
                    if "carModels" in extracted_data and extracted_data["carModels"]:
                        # Merge the two lists
                        existing_models = extracted_data["carModels"]
                        model_names = {
                            model.get("modelName", "").lower()
                            for model in existing_models
                        }

                        # Add unique models from car_models
                        for model in car_models:
                            model_name = model.get("modelName", "").strip()
                            # Skip invalid model names
                            if not model_name or len(model_name) < 4:
                                continue

                            # Skip common non-car-model entries
                            blacklist_terms = [
                                "more",
                                "all",
                                "view",
                                "see",
                                "choose",
                                "select",
                                "get",
                                "learn",
                                "specifications",
                                "features",
                                "protection",
                            ]
                            if any(
                                term in model_name.lower() for term in blacklist_terms
                            ):
                                continue

                            # Add if unique
                            if model_name.lower() not in model_names:
                                existing_models.append(model)
                                model_names.add(model_name.lower())

                        extracted_data["carModels"] = existing_models
                    else:
                        # Just use the scraped models
                        # First, filter out invalid model names
                        filtered_models = []
                        for model in car_models:
                            model_name = model.get("modelName", "").strip()
                            if not model_name or len(model_name) < 4:
                                continue

                            # Skip common non-car-model entries
                            blacklist_terms = [
                                "more",
                                "all",
                                "view",
                                "see",
                                "choose",
                                "select",
                                "get",
                                "learn",
                                "specifications",
                                "features",
                                "protection",
                            ]
                            if any(
                                term in model_name.lower() for term in blacklist_terms
                            ):
                                continue

                            filtered_models.append(model)

                        extracted_data["carModels"] = filtered_models

                # Deduplicate car models
                if "carModels" in extracted_data:
                    unique_models = {}
                    for model in extracted_data["carModels"]:
                        model_name = model.get("modelName", "").strip()
                        if model_name and model_name not in unique_models:
                            unique_models[model_name] = model
                    extracted_data["carModels"] = list(unique_models.values())

                print(
                    f"{Colors.GREEN}Final result has {len(extracted_data.get('carModels', []))} car models{Colors.RESET}"
                )
                return extracted_data

            except json.JSONDecodeError:
                print(
                    f"{Colors.RED}Failed to parse Deepseek response as JSON{Colors.RESET}"
                )
                # Return just the car models we scraped if JSON parsing fails
                if car_models:
                    # First, filter out invalid model names
                    filtered_models = []
                    for model in car_models:
                        model_name = model.get("modelName", "").strip()
                        if not model_name or len(model_name) < 4:
                            continue

                        # Skip common non-car-model entries
                        blacklist_terms = [
                            "more",
                            "all",
                            "view",
                            "see",
                            "choose",
                            "select",
                            "get",
                            "learn",
                            "specifications",
                            "features",
                            "protection",
                        ]
                        if any(term in model_name.lower() for term in blacklist_terms):
                            continue

                        filtered_models.append(model)

                    return {"carModels": filtered_models}
                return None

        except Exception as e:
            print(
                f"{Colors.RED}Error processing content with Deepseek: {str(e)}{Colors.RESET}"
            )
            # Return just the car models we scraped if Deepseek fails
            if car_models:
                # First, filter out invalid model names
                filtered_models = []
                for model in car_models:
                    model_name = model.get("modelName", "").strip()
                    if not model_name or len(model_name) < 4:
                        continue

                    # Skip common non-car-model entries
                    blacklist_terms = [
                        "more",
                        "all",
                        "view",
                        "see",
                        "choose",
                        "select",
                        "get",
                        "learn",
                        "specifications",
                        "features",
                        "protection",
                    ]
                    if any(term in model_name.lower() for term in blacklist_terms):
                        continue

                    filtered_models.append(model)

                return {"carModels": filtered_models}
            return None
    else:
        # For non-automotive data, format it for the frontend
        final_data = {}

        # Process each industry type
        for industry_type, data in all_extracted_data.items():
            if industry_type == "automotive":
                final_data["carModels"] = data.get("items", [])
            elif industry_type == "real_estate":
                final_data["properties"] = data.get("properties", [])
            elif industry_type == "ecommerce":
                final_data["products"] = data.get("products", [])
            elif industry_type == "travel":
                final_data["listings"] = data.get("listings", [])
            elif industry_type == "job_listing":
                final_data["jobs"] = data.get("jobs", [])
            elif industry_type == "generic":
                if "items" in data and data["items"]:
                    final_data["items"] = data["items"]
                elif "sections" in data:
                    final_data["sections"] = data["sections"]
                elif "content" in data:
                    final_data["content"] = data["content"]

        # Add company info
        final_data["company"] = company

        return final_data


def extract_real_estate_data(soup, url, objective):
    """Extract structured data specific to real estate industry"""
    data = {"type": "real_estate", "properties": []}

    # Look for property listings
    property_elements = soup.find_all(
        ["div", "article", "section", "li"],
        class_=lambda c: c
        and any(
            kw in c.lower()
            for kw in ["property", "listing", "house", "apartment", "condo", "estate"]
        ),
    )

    for element in property_elements:
        property_item = {"source_url": url}

        # Extract property title/name
        title_elem = element.find(["h1", "h2", "h3", "h4", "h5"])
        if title_elem:
            property_item["title"] = title_elem.get_text().strip()

        # Extract price
        price_elem = element.find(
            text=re.compile(
                r"(\$|USD|€|EUR|£|GBP|₱|PHP|¥|JPY|CNY|₹|INR)\s*[\d,]+", re.IGNORECASE
            )
        )
        if price_elem:
            property_item["price"] = (
                price_elem.strip()
                if isinstance(price_elem, str)
                else price_elem.get_text().strip()
            )

        # Extract details (bedrooms, bathrooms, etc.)
        details = {}
        detail_patterns = {
            "bedrooms": r"(\d+)\s*(?:bed|bedroom|br)",
            "bathrooms": r"(\d+(?:\.\d+)?)\s*(?:bath|bathroom|ba)",
            "area": r"(\d+(?:,\d+)?)\s*(?:sq\s*ft|sqft|m2|square\s*feet|square\s*meter)",
        }

        for key, pattern in detail_patterns.items():
            match = re.search(pattern, element.get_text(), re.IGNORECASE)
            if match:
                details[key] = match.group(1)

        if details:
            property_item["details"] = details

        # Extract location information
        location_elem = element.find(
            ["div", "span", "p"],
            class_=lambda c: c
            and any(
                kw in c.lower() for kw in ["location", "address", "area", "region"]
            ),
        )
        if location_elem:
            property_item["location"] = location_elem.get_text().strip()

        # Add to results if we have at least title or price
        if "title" in property_item or "price" in property_item:
            data["properties"].append(property_item)

    return data


def extract_ecommerce_data(soup, url, objective):
    """Extract structured data specific to e-commerce industry"""
    data = {"type": "ecommerce", "products": []}

    # Look for product elements
    product_elements = soup.find_all(
        ["div", "article", "section", "li"],
        class_=lambda c: c
        and any(kw in c.lower() for kw in ["product", "item", "goods", "merchandise"]),
    )

    # If no product elements found with class, try with data-* attributes
    if not product_elements:
        product_elements = soup.find_all(
            ["div", "article", "section", "li"],
            attrs=lambda attrs: any(
                attr
                for attr in attrs
                if attr.startswith("data-")
                and any(
                    kw in attrs[attr].lower()
                    for kw in ["product", "item", "sku", "price"]
                )
            ),
        )

    # Look for product-like structures in grid/list layouts
    if not product_elements:
        grid_containers = soup.find_all(
            ["div", "ul"],
            class_=lambda c: c
            and any(
                pattern in c.lower()
                for pattern in ["grid", "products", "list", "catalog"]
            ),
        )
        for container in grid_containers:
            items = container.find_all(["div", "li", "article"], recursive=False)
            if items and len(items) >= 2:
                product_elements.extend(items)

    for element in product_elements:
        product = {"source_url": url}

        # Extract product name
        name_elem = element.find(["h1", "h2", "h3", "h4", "h5", "strong"])
        if name_elem:
            product["name"] = name_elem.get_text().strip()

        # Extract price
        price_pattern = (
            r"(\$|USD|€|EUR|£|GBP|₱|PHP|¥|JPY|CNY|₹|INR)\s*[\d,]+(?:\.\d{2})?"
        )
        price_elem = element.find(text=re.compile(price_pattern, re.IGNORECASE))
        if price_elem:
            price_text = (
                price_elem.strip()
                if isinstance(price_elem, str)
                else price_elem.get_text().strip()
            )
            price_match = re.search(price_pattern, price_text, re.IGNORECASE)
            if price_match:
                product["price"] = price_match.group(0)

        # Extract description
        desc_elem = element.find(
            ["p", "div", "span"],
            class_=lambda c: c
            and any(
                kw in c.lower()
                for kw in ["desc", "description", "info", "detail", "feature"]
            ),
        )
        if desc_elem:
            description = desc_elem.get_text().strip()
            if len(description) > 200:
                description = description[:197] + "..."
            product["description"] = description

        # Add to results if we have at least a name or a price
        if "name" in product or "price" in product:
            data["products"].append(product)

    return data


def extract_travel_data(soup, url, objective):
    """Extract structured data specific to travel industry"""
    data = {"type": "travel", "listings": []}

    # Look for travel/hotel listings
    listing_elements = soup.find_all(
        ["div", "article", "section", "li"],
        class_=lambda c: c
        and any(
            kw in c.lower()
            for kw in [
                "hotel",
                "accommodation",
                "room",
                "flight",
                "package",
                "tour",
                "destination",
            ]
        ),
    )

    for element in listing_elements:
        listing = {"source_url": url}

        # Extract listing name
        name_elem = element.find(["h1", "h2", "h3", "h4", "h5", "strong"])
        if name_elem:
            listing["name"] = name_elem.get_text().strip()

        # Extract price
        price_pattern = (
            r"(\$|USD|€|EUR|£|GBP|₱|PHP|¥|JPY|CNY|₹|INR)\s*[\d,]+(?:\.\d{2})?"
        )
        price_elem = element.find(text=re.compile(price_pattern, re.IGNORECASE))
        if price_elem:
            price_text = (
                price_elem.strip()
                if isinstance(price_elem, str)
                else price_elem.get_text().strip()
            )
            price_match = re.search(price_pattern, price_text, re.IGNORECASE)
            if price_match:
                listing["price"] = price_match.group(0)

        # Extract rating/stars
        rating_elem = element.find(
            ["div", "span"],
            class_=lambda c: c
            and any(kw in c.lower() for kw in ["rating", "star", "score", "review"]),
        )
        if rating_elem:
            listing["rating"] = rating_elem.get_text().strip()

        # Extract location
        location_elem = element.find(
            ["div", "span", "p"],
            class_=lambda c: c
            and any(
                kw in c.lower()
                for kw in ["location", "address", "place", "destination"]
            ),
        )
        if location_elem:
            listing["location"] = location_elem.get_text().strip()

        # Add to results if we have at least a name
        if "name" in listing:
            data["listings"].append(listing)

    return data


def extract_job_data(soup, url, objective):
    """Extract structured data specific to job listings"""
    data = {"type": "job_listing", "jobs": []}

    # Look for job listing elements
    job_elements = soup.find_all(
        ["div", "article", "section", "li"],
        class_=lambda c: c
        and any(
            kw in c.lower()
            for kw in ["job", "position", "vacancy", "career", "opening"]
        ),
    )

    for element in job_elements:
        job = {"source_url": url}

        # Extract job title
        title_elem = element.find(["h1", "h2", "h3", "h4", "h5", "strong"])
        if title_elem:
            job["title"] = title_elem.get_text().strip()

        # Extract company name
        company_elem = element.find(
            ["div", "span", "p"],
            class_=lambda c: c
            and any(kw in c.lower() for kw in ["company", "employer", "organization"]),
        )
        if company_elem:
            job["company"] = company_elem.get_text().strip()

        # Extract salary
        salary_pattern = r"(\$|USD|€|EUR|£|GBP|₱|PHP|¥|JPY|CNY|₹|INR)\s*[\d,]+(?:\.\d{2})?(?:\s*-\s*(\$|USD|€|EUR|£|GBP|₱|PHP|¥|JPY|CNY|₹|INR)?\s*[\d,]+(?:\.\d{2})?)?"
        salary_elem = element.find(text=re.compile(salary_pattern, re.IGNORECASE))
        if salary_elem:
            salary_text = (
                salary_elem.strip()
                if isinstance(salary_elem, str)
                else salary_elem.get_text().strip()
            )
            salary_match = re.search(salary_pattern, salary_text, re.IGNORECASE)
            if salary_match:
                job["salary"] = salary_match.group(0)

        # Extract location
        location_elem = element.find(
            ["div", "span", "p"],
            class_=lambda c: c
            and any(
                kw in c.lower() for kw in ["location", "address", "place", "region"]
            ),
        )
        if location_elem:
            job["location"] = location_elem.get_text().strip()

        # Add to results if we have at least a title
        if "title" in job:
            data["jobs"].append(job)

    return data


def extract_generic_content(soup, url, objective):
    """Extract generic structured content when no specific industry is detected"""
    data = {"type": "generic", "items": []}

    # Try to find items that match the objective
    objective_words = set(re.sub(r"[^\w\s]", " ", objective.lower()).split())

    # Look for sections that might contain relevant content
    relevant_sections = []

    # First check headers that match objective keywords
    headers = soup.find_all(["h1", "h2", "h3", "h4"])
    for header in headers:
        header_text = header.get_text().lower()
        if any(word in header_text for word in objective_words):
            # Find the next sibling elements until next header or significant change
            relevant_content = []
            current = header.next_sibling
            while current and not current.name in ["h1", "h2", "h3", "h4"]:
                if current.name in ["p", "div", "span", "ul", "ol", "table"]:
                    content_text = current.get_text().strip()
                    if len(content_text) > 20:  # Only consider substantial content
                        relevant_content.append(
                            {"type": current.name, "content": content_text}
                        )
                current = current.next_sibling

            if relevant_content:
                relevant_sections.append(
                    {"heading": header.get_text().strip(), "content": relevant_content}
                )

    # Look for structured items like cards, listings, etc.
    item_containers = soup.find_all(
        ["div", "article", "section", "li"],
        class_=lambda c: c
        and any(
            kw in c.lower()
            for kw in ["item", "card", "block", "listing", "content", "result"]
        ),
    )

    for container in item_containers:
        container_text = container.get_text().lower()
        if any(word in container_text for word in objective_words):
            item = {"source_url": url}

            # Extract title/heading
            title_elem = container.find(["h1", "h2", "h3", "h4", "h5", "strong"])
            if title_elem:
                item["title"] = title_elem.get_text().strip()

            # Extract description
            desc_elem = container.find(["p", "div", "span"])
            if desc_elem and desc_elem.get_text().strip():
                description = desc_elem.get_text().strip()
                if len(description) > 200:
                    description = description[:197] + "..."
                item["description"] = description

            # Look for any data points matching objective words
            for word in objective_words:
                # Find elements containing this keyword
                matches = container.find_all(
                    text=lambda text: text and word in text.lower()
                )
                for match in matches:
                    parent = match.parent
                    if parent and parent.name not in [
                        "h1",
                        "h2",
                        "h3",
                        "h4",
                        "h5",
                        "strong",
                        "p",
                    ]:
                        # This might be a specific data point
                        item[word] = (
                            match.strip()
                            if isinstance(match, str)
                            else match.get_text().strip()
                        )

            if "title" in item or "description" in item:
                data["items"].append(item)

    # If we found relevant sections but no items, use the sections
    if relevant_sections and not data["items"]:
        data["sections"] = relevant_sections

    # As a last resort, extract main text content matching objective
    if not data["items"] and not relevant_sections:
        paragraphs = soup.find_all(
            ["p", "div"],
            class_=lambda c: not c
            or not any(
                kw in (c or "").lower()
                for kw in ["nav", "menu", "footer", "header", "sidebar"]
            ),
        )

        relevant_paragraphs = []
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 50 and any(word in text.lower() for word in objective_words):
                relevant_paragraphs.append(text)

        if relevant_paragraphs:
            data["content"] = relevant_paragraphs[
                :5
            ]  # Limit to top 5 most relevant paragraphs

    return data


def crawl_website(input_url_or_company, objective=None):
    """
    Crawl a website to extract car model information.

    Args:
        input_url_or_company (str): Either a URL or a company name to crawl
        objective (str, optional): The search objective. Defaults to None.

    Returns:
        dict: A dictionary containing the extracted information
    """
    print(
        f"{Colors.YELLOW}Starting crawl for: {input_url_or_company} with objective: {objective}{Colors.RESET}"
    )

    # For consistency in returned structure
    final_result = {
        "crawled_urls": [],
        "data": {},
        "error": None,
        "timestamps": {"start": datetime.now().isoformat(), "end": None},
    }

    is_input_url = False
    domain = None

    # Check if input is URL or company name
    if input_url_or_company.startswith(("http://", "https://")):
        # Input is a URL
        url = input_url_or_company
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        print(f"{Colors.YELLOW}Input is a URL: {url}. Crawling directly.{Colors.RESET}")
        is_input_url = True
    elif "." in input_url_or_company and " " not in input_url_or_company:
        # Looks like a domain without scheme
        url = "https://" + input_url_or_company
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        print(
            f"{Colors.YELLOW}Input is a domain: {domain}. Adding https:// and crawling.{Colors.RESET}"
        )
        is_input_url = True
    else:
        # Input is a company name
        print(
            f"{Colors.YELLOW}Input is a company name: {input_url_or_company}. Searching Google...{Colors.RESET}"
        )

        # Perform a Google search for the company
        search_results = search_google(input_url_or_company)

        if not search_results:
            error_msg = f"No search results found for {input_url_or_company}"
            print(f"{Colors.RED}{error_msg}{Colors.RESET}")
            final_result["error"] = error_msg
            return final_result

        # Use R1 to select the most relevant URLs
        prompt = f"Based on the objective '{objective}', select the most relevant URLs for {input_url_or_company}"
        urls = select_urls_with_r1(
            input_url_or_company, objective or "company information", search_results
        )

        if not urls:
            error_msg = f"No relevant URLs found for {input_url_or_company}"
            print(f"{Colors.RED}{error_msg}{Colors.RESET}")
            final_result["error"] = error_msg
            return final_result

        # Validate the selected URLs
        valid_urls = validate_urls(urls)
        if not valid_urls:
            error_msg = f"No valid URLs found for {input_url_or_company}"
            print(f"{Colors.RED}{error_msg}{Colors.RESET}")
            final_result["error"] = error_msg
            return final_result

        # Filter URLs to keep only those relevant to the objective
        model_info_patterns = [
            r"/vehicles/",
            r"/models/",
            r"/cars/",
            r"/catalog/",
            r"/pricelist",
            r"/prices",
            r"/price-list",
            r"/compare-models",
        ]

        filtered_urls = []
        # Always include the home page
        for u in valid_urls:
            path = urlparse(u).path
            if path == "/" or path == "":
                filtered_urls.append(u)
                break

        # Then add model-related URLs
        for u in valid_urls:
            path = urlparse(u).path.lower()
            if any(re.search(pattern, path) for pattern in model_info_patterns):
                if u not in filtered_urls:
                    filtered_urls.append(u)

        # If we haven't found any model-related URLs, just use the valid URLs
        if not filtered_urls:
            filtered_urls = valid_urls

        final_result["crawled_urls"] = filtered_urls

        # Extract company info from the selected URLs
        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        if not firecrawl_api_key:
            error_msg = "FIRECRAWL_API_KEY not found in environment variables"
            print(f"{Colors.RED}{error_msg}{Colors.RESET}")
            final_result["error"] = error_msg
            return final_result

        prompt = f"Based on the objective '{objective}', extract relevant information about {input_url_or_company}. Prioritize details related to the objective."
        print(f"{Colors.YELLOW}Using R1 prompt: {prompt}{Colors.RESET}")
        print(
            f"{Colors.YELLOW}Starting extraction process for {input_url_or_company}...{Colors.RESET}"
        )

        extracted_data = extract_company_info(
            filtered_urls,
            prompt,
            input_url_or_company,
            firecrawl_api_key,
            is_input_url=False,
            objective=objective,
        )

        if extracted_data:
            final_result["data"] = extracted_data
            print(f"{Colors.GREEN}Extraction completed successfully.{Colors.RESET}")
        else:
            error_msg = f"Failed to extract information from the URLs for {input_url_or_company}"
            print(f"{Colors.RED}{error_msg}{Colors.RESET}")
            final_result["error"] = error_msg

    # Handle URL/domain input
    if is_input_url:
        # Validate the URL
        valid_urls = validate_urls([url])
        if not valid_urls:
            error_msg = f"Could not validate URL: {url}"
            print(f"{Colors.RED}{error_msg}{Colors.RESET}")
            final_result["error"] = error_msg
            return final_result

        # Discover subpages to get more URLs to crawl
        print(f"{Colors.YELLOW}Discovering subpages for {domain}...{Colors.RESET}")
        discovered_urls = discover_subpages(
            url, domain, objective, max_pages=20, max_depth=2
        )

        # Add discovered URLs to valid_urls
        if discovered_urls:
            valid_urls.extend([u for u in discovered_urls if u not in valid_urls])
            print(
                f"{Colors.GREEN}Found {len(discovered_urls)} additional pages to crawl{Colors.RESET}"
            )

        # Filter URLs to ensure they match the input domain
        domain_valid_urls = [
            u for u in valid_urls if domain and domain in urlparse(u).netloc
        ]
        if not domain_valid_urls:
            domain_valid_urls = valid_urls  # Fallback to all valid URLs if domain filtering removed everything

        print(f"{Colors.YELLOW}Filtering URLs to match domain: {domain}{Colors.RESET}")
        print(
            f"{Colors.YELLOW}Total URLs to crawl: {len(domain_valid_urls)}{Colors.RESET}"
        )
        final_result["crawled_urls"] = domain_valid_urls

        # Extract company info from the URL
        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        if not firecrawl_api_key:
            error_msg = "FIRECRAWL_API_KEY not found in environment variables"
            print(f"{Colors.RED}{error_msg}{Colors.RESET}")
            final_result["error"] = error_msg
            return final_result

        prompt = f"Based on the objective '{objective}', extract relevant information about {domain}. Prioritize details related to the objective."
        print(f"{Colors.YELLOW}Using R1 prompt: {prompt}{Colors.RESET}")
        print(
            f"{Colors.YELLOW}Starting extraction process for {domain}...{Colors.RESET}"
        )

        start_time = time.time()
        extracted_data = extract_company_info(
            domain_valid_urls,
            prompt,
            domain,
            firecrawl_api_key,
            is_input_url=True,
            objective=objective,
        )
        elapsed_time = time.time() - start_time

        if extracted_data:
            final_result["data"] = extracted_data
            print(
                f"{Colors.GREEN}Extraction completed in {elapsed_time:.2f} seconds.{Colors.RESET}"
            )
        else:
            error_msg = f"Failed to extract information from the URL: {url}"
            print(f"{Colors.RED}{error_msg}{Colors.RESET}")
            final_result["error"] = error_msg

    # Record end timestamp
    final_result["timestamps"]["end"] = datetime.now().isoformat()

    # Clean up the result structure
    if final_result["error"]:
        print(f"{Colors.RED}Crawl failed. Error: {final_result['error']}{Colors.RESET}")
        return {"error": final_result["error"]}
    else:
        # Post-process the extracted data to make it more accurate and relevant

        # For car models, remove duplicates and filter out irrelevant entries
        if "carModels" in final_result["data"]:
            # Remove duplicates based on model name
            unique_models = {}
            for model in final_result["data"]["carModels"]:
                model_name = model.get("modelName", "").strip()

                # Skip entries that are likely not car models
                if (
                    not model_name
                    or len(model_name) < 4
                    or re.search(
                        r"^[0-9]+$", model_name
                    )  # Skip entries that are just numbers
                    or model_name.lower()
                    in ["more", "view all", "see more", "all vehicles", "get a quote"]
                ):
                    continue

                # Skip entries with unicode escape sequences
                if "\\u" in model_name:
                    continue

                # Skip marketing slogans and phrases that are not car models
                blacklist_terms = [
                    "home",
                    "vehicles",
                    "cars",
                    "models",
                    "inventory",
                    "search",
                    "all",
                    "new",
                    "used",
                    "certified",
                    "more details",
                    "view models",
                    "built to",
                    "designed for",
                    "engineered to",
                    "elevated",
                    "pioneering",
                    "introducing",
                    "explore",
                    "discover",
                    "introducing",
                    "experience",
                    "innovative",
                    "sustainable",
                    "luxury",
                    "performance",
                    "excellence",
                    "premium",
                    "efficiency",
                    "safe and",
                    "safety",
                    "learn more",
                    "contact us",
                    "find a dealer",
                    "build your",
                    "configure",
                    "coming soon",
                    "the new",
                    "the all-new",
                ]

                # More sophisticated model name validation
                def is_valid_model_name(name):
                    # Check if it's not in blacklist
                    if any(term in name.lower() for term in blacklist_terms):
                        return False

                    # Check if it contains marketing fluff words as standalone phrases
                    marketing_phrases = [
                        "built to be",
                        "designed for",
                        "elevated full",
                        "pioneering",
                    ]
                    if any(phrase in name.lower() for phrase in marketing_phrases):
                        return False

                    # Check if it contains a car brand - higher likelihood it's a real model
                    has_brand = any(
                        brand.lower() in name.lower() for brand in car_brands
                    )

                    # If no brand, apply stricter validation
                    if not has_brand:
                        # Most real car models are 1-3 words, sometimes with numbers
                        words = name.split()
                        if len(words) > 5:  # Too wordy to be a model name
                            return False

                        # Check if it looks like a sentence rather than a model name
                        if name[0].islower() and len(words) > 2:
                            return False

                    return True

                if (
                    model_name
                    and len(model_name.strip()) > 3
                    and is_valid_model_name(model_name)
                ):
                    car_models.append(
                        {
                            "modelName": model_name,
                            "price": price
                            if price
                            else "Price information not available",
                            "description": description
                            if description
                            else "No description available",
                        }
                    )

            # Only keep models that have at least a price or description
            filtered_models = []
            for model in unique_models.values():
                if "price" in model or "description" in model:
                    filtered_models.append(model)

            # If we don't have any models with both price and description, but have some with either,
            # keep those to provide at least some information
            if not any(
                ("price" in model and "description" in model)
                for model in filtered_models
            ):
                final_models = filtered_models
            else:
                # If we have complete models (with both price and description), prioritize those
                final_models = [
                    model
                    for model in filtered_models
                    if "price" in model and "description" in model
                ]

                # If we have very few complete models, include some with just price or description
                if len(final_models) < 3 and len(filtered_models) > len(final_models):
                    # Add up to 5 more models with at least a price
                    price_models = [
                        m
                        for m in filtered_models
                        if "price" in m and m not in final_models
                    ]
                    final_models.extend(price_models[:5])

            # Replace with the cleaned up models
            final_result["data"]["carModels"] = final_models

            # Sort models alphabetically for better presentation
            final_result["data"]["carModels"].sort(key=lambda x: x.get("modelName", ""))

            print(
                f"{Colors.GREEN}Cleaned up car models: found {len(final_result['data']['carModels'])} unique models with complete information{Colors.RESET}"
            )

        print(
            f"{Colors.GREEN}Crawl successful. Extracted data: {json.dumps(final_result['data'], indent=2)}{Colors.RESET}"
        )
        print(
            f"{Colors.YELLOW}Debug crawl_website: Returning final_result['urls']: {final_result['crawled_urls']}{Colors.RESET}"
        )
        return {
            "crawled_urls": final_result["crawled_urls"],
            "data": final_result["data"],
            "timestamps": final_result["timestamps"],
        }


def extract_car_models_with_api(url, domain=None):
    """
    Extract car models from a specific URL using API or direct HTML parsing
    """
    print(f"{Colors.YELLOW}Extracting car models from {url}{Colors.RESET}")
    car_models = []

    try:
        response = requests.get(url, headers=USER_AGENT_HEADER, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Check if the URL is likely a model details page or listing page
        is_model_detail_page = any(
            keyword in url.lower()
            for keyword in ["model", "vehicle", "car", "suv", "sedan"]
        )

        # Common car brands to help identify actual model names
        car_brands = [
            "toyota",
            "honda",
            "ford",
            "chevrolet",
            "bmw",
            "mercedes",
            "audi",
            "tesla",
            "hyundai",
            "kia",
            "nissan",
            "mazda",
            "lexus",
            "subaru",
            "volkswagen",
            "vw",
            "jeep",
            "cadillac",
            "buick",
            "chrysler",
            "dodge",
            "gmc",
            "lincoln",
            "acura",
            "infiniti",
            "volvo",
            "jaguar",
            "land rover",
            "mini",
            "mitsubishi",
            "genesis",
            "porsche",
            "fiat",
            "alfa romeo",
            "maserati",
            "bentley",
            "rolls-royce",
            "aston martin",
            "ferrari",
            "lamborghini",
            "mclaren",
            "bugatti",
            "byd",
            "nio",
            "rivian",
            "lucid",
        ]

        # Try to find model names and prices in various ways
        # Method 1: Look for dedicated model elements
        model_elements = soup.select(
            ".model, .car-model, .vehicle, .car-name, h1.title, .model-name, .model-title"
        )
        price_elements = soup.select(
            ".price, .msrp, .car-price, .vehicle-price, .model-price, .starting-price"
        )

        # Method 2: Look for headings that might contain model names
        headings = soup.find_all(["h1", "h2", "h3"])

        # Extract text from all paragraphs for descriptions
        paragraphs = soup.find_all("p")
        description_text = " ".join(
            [p.text.strip() for p in paragraphs if len(p.text.strip()) > 30]
        )

        # If we're on a model detail page, we're likely to have a single model with its details
        if is_model_detail_page:
            # Try to extract the model name from the title
            title = soup.find("title")
            title_text = title.text.strip() if title else ""

            # Try to extract model from H1 (often the main model name on detail pages)
            h1 = soup.find("h1")
            model_name = h1.text.strip() if h1 else ""

            # If no H1 or H1 too generic, try title
            if (
                not model_name
                or len(model_name) < 3
                or model_name.lower() in ["home", "vehicles", "models"]
            ):
                # Extract model from title - usually has format "Model Name | Brand Name"
                if " | " in title_text:
                    model_name = title_text.split(" | ")[0].strip()
                else:
                    # Try to extract just the model part from the title
                    # Remove common website suffixes
                    cleaned_title = re.sub(
                        r" [-|] .*(Motors|Cars|Automotive|Auto|Official Site).*$",
                        "",
                        title_text,
                    )

                    # If any car brand is in the title, try to extract the model name
                    for brand in car_brands:
                        if brand.lower() in cleaned_title.lower():
                            # Model name often follows the brand name
                            pattern = re.compile(
                                f"{brand}\\s+([\\w\\s-]+)", re.IGNORECASE
                            )
                            match = pattern.search(cleaned_title)
                            if match:
                                model_name = f"{brand} {match.group(1).strip()}"
                                break

            # Try to find price directly
            price = None
            for price_elem in price_elements:
                price_text = price_elem.text.strip()
                if re.search(
                    r"(\$[\d,]+|\$[\d,]+\s*-\s*\$[\d,]+|starting at \$[\d,]+|from \$[\d,]+)",
                    price_text,
                    re.IGNORECASE,
                ):
                    price = price_text
                    break

            # If we haven't found a price, look more broadly
            if not price:
                # Look for dollar amounts in the page
                price_patterns = [
                    r"(\$[\d,]+)",
                    r"(starting at|from|beginning at)\s*\$[\d,]+",
                    r"price[\s:]*\$[\d,]+",
                ]
                for pattern in price_patterns:
                    for element in soup.find_all(
                        text=re.compile(pattern, re.IGNORECASE)
                    ):
                        match = re.search(pattern, element, re.IGNORECASE)
                        if match:
                            # Get the parent to find more context
                            parent = element.parent
                            price = parent.text.strip()
                            break
                    if price:
                        break

            # Add the model to the results if we found a valid model name
            if model_name and len(model_name.strip()) > 3:
                # Look for a description related to this model
                description = ""

                # Try to extract description from meta description
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc and "content" in meta_desc.attrs:
                    description = meta_desc["content"].strip()

                # If no meta description, look for a suitable paragraph
                if not description or len(description) < 50:
                    for p in paragraphs:
                        p_text = p.text.strip()
                        # Look for paragraphs that mention the model or seem descriptive
                        if (
                            len(p_text) > 100
                            and not re.search(
                                r"(privacy|cookie|terms|policy)", p_text.lower()
                            )
                            and not re.search(
                                r"(copyright|all rights reserved)", p_text.lower()
                            )
                        ):
                            description = p_text
                            break

                # Ensure the model name is not just a generic term
                blacklist_terms = [
                    "home",
                    "vehicles",
                    "cars",
                    "models",
                    "inventory",
                    "search",
                    "all",
                    "new",
                    "used",
                    "certified",
                    "more details",
                    "view models",
                    "built to",
                    "designed for",
                    "engineered to",
                    "elevated",
                    "pioneering",
                    "introducing",
                    "explore",
                    "discover",
                    "introducing",
                    "experience",
                    "innovative",
                    "sustainable",
                    "luxury",
                    "performance",
                    "excellence",
                    "premium",
                    "efficiency",
                    "safe and",
                    "safety",
                    "learn more",
                    "contact us",
                    "find a dealer",
                    "build your",
                    "configure",
                    "coming soon",
                    "the new",
                    "the all-new",
                ]

                if not any(term == model_name.lower() for term in blacklist_terms):
                    car_models.append(
                        {
                            "modelName": model_name,
                            "price": price
                            if price
                            else "Price information not available",
                            "description": description
                            if description
                            else "No description available",
                        }
                    )

        # For listing pages, try to extract multiple models
        else:
            # Find potential model containers
            containers = soup.select(
                ".vehicle-card, .car-card, .model-card, .vehicle-item, .car-item, .model-item, .vehicle, .car, .model"
            )

            # If no specific containers, try to find divs that might contain model info
            if not containers:
                # Look for divs containing both model-like and price-like elements
                for div in soup.find_all("div"):
                    # Check if this div contains model and price info
                    has_model = any(
                        keyword in div.text.lower()
                        for keyword in ["model", "vehicle", "car"]
                    )
                    has_price = re.search(r"(\$[\d,]+|price|msrp)", div.text.lower())

                    if has_model and has_price:
                        containers.append(div)

            # Extract from containers if found
            if containers:
                for container in containers:
                    # Try to find model name
                    model_name_elem = container.select_one(
                        ".model-name, .car-name, .vehicle-name, h3, h4, .title, strong"
                    )
                    model_name = model_name_elem.text.strip() if model_name_elem else ""

                    # If no specific element found, try to infer from container text
                    if not model_name:
                        # Look for potential model name patterns
                        container_text = container.text.strip()

                        # Check if any known car brand is in the text
                        for brand in car_brands:
                            if brand.lower() in container_text.lower():
                                # Try to extract model name following the brand
                                pattern = re.compile(
                                    f"{brand}\\s+([\\w\\s-]+)", re.IGNORECASE
                                )
                                match = pattern.search(container_text)
                                if match:
                                    model_name = f"{brand} {match.group(1).strip()}"
                                    break

                    # Try to find price
                    price_elem = container.select_one(
                        ".price, .msrp, .car-price, .vehicle-price"
                    )
                    price = price_elem.text.strip() if price_elem else ""

                    # If no specific price element, look for price pattern
                    if not price:
                        price_match = re.search(
                            r"(\$[\d,]+|\$[\d,]+\s*-\s*\$[\d,]+|starting at \$[\d,]+)",
                            container.text,
                            re.IGNORECASE,
                        )
                        if price_match:
                            price = price_match.group(0)

                    # Try to find description
                    desc_elem = container.select_one("p, .description, .overview")
                    description = desc_elem.text.strip() if desc_elem else ""

                    # Only add if we have a valid model name
                    blacklist_terms = [
                        "home",
                        "vehicles",
                        "cars",
                        "models",
                        "inventory",
                        "search",
                        "all",
                        "new",
                        "used",
                        "certified",
                        "more details",
                        "view models",
                        "built to",
                        "designed for",
                        "engineered to",
                        "elevated",
                        "pioneering",
                        "introducing",
                        "explore",
                        "discover",
                        "introducing",
                        "experience",
                        "innovative",
                        "sustainable",
                        "luxury",
                        "performance",
                        "excellence",
                        "premium",
                        "efficiency",
                        "safe and",
                        "safety",
                        "learn more",
                        "contact us",
                        "find a dealer",
                        "build your",
                        "configure",
                        "coming soon",
                        "the new",
                        "the all-new",
                    ]

                    if (
                        model_name
                        and len(model_name) > 3
                        and not any(
                            term == model_name.lower() for term in blacklist_terms
                        )
                    ):
                        car_models.append(
                            {
                                "modelName": model_name,
                                "price": price
                                if price
                                else "Price information not available",
                                "description": description
                                if description
                                else "No description available",
                            }
                        )

            # If no models found from containers, try a more generic approach
            if not car_models:
                # Extract from headings that might be model names
                for heading in headings:
                    heading_text = heading.text.strip()

                    # Skip very short headings or obvious non-model headings
                    if len(heading_text) < 3 or heading_text.lower() in [
                        "home",
                        "vehicles",
                        "models",
                        "inventory",
                    ]:
                        continue

                    # Check if this heading might be a car model
                    is_potential_model = False

                    # If it contains a known car brand, it's likely a model
                    for brand in car_brands:
                        if brand.lower() in heading_text.lower():
                            is_potential_model = True
                            break

                    # If the heading has nearby price information, it's likely a model
                    if not is_potential_model:
                        next_sibling = heading.next_sibling
                        while (
                            next_sibling
                            and isinstance(next_sibling, str)
                            and not next_sibling.strip()
                        ):
                            next_sibling = next_sibling.next_sibling

                        if next_sibling:
                            sibling_text = (
                                next_sibling.text.strip()
                                if hasattr(next_sibling, "text")
                                else ""
                            )
                            if re.search(
                                r"(\$[\d,]+|price|msrp)", sibling_text.lower()
                            ):
                                is_potential_model = True
                                price = sibling_text

                    if is_potential_model:
                        # Look for nearby price if not found yet
                        if "price" not in locals() or not price:
                            # Look in siblings for price
                            siblings = list(heading.next_siblings)
                            for sibling in siblings[:5]:  # Check next 5 siblings
                                if hasattr(sibling, "text"):
                                    sibling_text = sibling.text.strip()
                                    price_match = re.search(
                                        r"(\$[\d,]+|\$[\d,]+\s*-\s*\$[\d,]+|starting at \$[\d,]+)",
                                        sibling_text,
                                        re.IGNORECASE,
                                    )
                                    if price_match:
                                        price = price_match.group(0)
                                        break

                        # Look for nearby description
                        description = ""
                        siblings = list(heading.next_siblings)
                        for sibling in siblings[:5]:  # Check next 5 siblings
                            if hasattr(sibling, "name") and sibling.name == "p":
                                sibling_text = sibling.text.strip()
                                if (
                                    len(sibling_text) > 50
                                ):  # Only use substantial paragraphs
                                    description = sibling_text
                                    break

                        # Only add if it looks like a real model (not just a heading)
                        blacklist_terms = [
                            "home",
                            "vehicles",
                            "cars",
                            "models",
                            "inventory",
                            "search",
                            "all",
                            "new",
                            "used",
                            "certified",
                            "more details",
                            "view models",
                            "built to",
                            "designed for",
                            "engineered to",
                            "elevated",
                            "pioneering",
                            "introducing",
                            "explore",
                            "discover",
                            "introducing",
                            "experience",
                            "innovative",
                            "sustainable",
                            "luxury",
                            "performance",
                            "excellence",
                            "premium",
                            "efficiency",
                            "starting",
                            "user-friendly",
                            "seats",
                            "space",
                            "safe and",
                            "elevated",
                        ]
                        if any(term in model_name for term in blacklist_terms):
#
#
            # If the model already exists, keep the one with more information
            if model_name in unique_models:
                existing_model = unique_models[model_name]

                # Compare which entry has more information
                current_info_score = (
                    1
                    if model.get("price")
                    and model["price"] != "Price information not available"
                    else 0
                ) + (
                    1
                    if model.get("description")
                    and model["description"] != "No description available"
                    else 0
                )

                existing_info_score = (
                    1
                    if existing_model.get("price")
                    and existing_model["price"] != "Price information not available"
                    else 0
                ) + (
                    1
                    if existing_model.get("description")
                    and existing_model["description"] != "No description available"
                    else 0
                )

                # Replace if current model has more information
                if current_info_score > existing_info_score:
                    unique_models[model_name] = model
            else:
                unique_models[model_name] = model

        # Convert back to list, only keeping entries that have actual information
        car_models = []
        for model in unique_models.values():
            has_price = (
                model.get("price")
                and model["price"] != "Price information not available"
            )
            has_description = (
                model.get("description")
                and model["description"] != "No description available"
            )

            # Only include models that have either a price or a description
            if has_price or has_description:
                car_models.append(model)

        print(
            f"{Colors.GREEN}Found {len(car_models)} potential car models{Colors.RESET}"
        )
        return car_models

    except requests.exceptions.RequestException as e:
        print(f"{Colors.RED}Failed to retrieve URL {url}: {str(e)}{Colors.RESET}")
        return []
    except Exception as e:
        print(
            f"{Colors.RED}Error extracting car models from {url}: {str(e)}{Colors.RESET}"
        )
        return []


def extract_automotive_data(soup, url, objective):
    """Extract structured data specific to automotive industry"""
    from urllib.parse import urlparse

    data = {"type": "automotive", "items": []}

    # Look for car model elements (similar to extract_car_models_with_api but simplified)
    # For the automotive industry, we'll reuse our existing car model extraction logic
    models = extract_car_models_with_api(url, urlparse(url).netloc)
    if models:
        data["items"] = models

    return data


def discover_subpages(base_url, domain, objective=None, max_pages=25, max_depth=2):
    """
    Discover subpages within a domain by recursively crawling the base URL and finding internal links.

    Args:
        base_url (str): The base URL to start crawling from
        domain (str): The domain to restrict crawling to
        objective (str): The user's search objective to guide page selection
        max_pages (int): Maximum number of pages to discover (default 25)
        max_depth (int): Maximum recursion depth for crawling (default 2)

    Returns:
        list: A list of discovered URLs within the domain, sorted by relevance
    """
    print(f"{Colors.YELLOW}Discovering subpages starting from {base_url}{Colors.RESET}")
    discovered_urls = set()
    visited_urls = set()

    # Track relevance scores for URLs
    url_scores = {}

    # Parse the objective to extract keywords
    objective_keywords = []
    if objective:
        # Clean and tokenize the objective
        cleaned_objective = re.sub(r"[^\w\s]", " ", objective.lower())
        objective_keywords = [
            word.strip() for word in cleaned_objective.split() if word.strip()
        ]
        print(
            f"{Colors.YELLOW}Extracted keywords from objective: {objective_keywords}{Colors.RESET}"
        )

    # Generate industry-specific keywords based on the objective
    industry_keywords = {}

    # Automotive industry keywords - prioritize model and pricing pages
    if any(
        kw in (objective or "").lower()
        for kw in [
            "car",
            "vehicle",
            "automotive",
            "model",
            "suv",
            "sedan",
            "price",
            "pricing",
        ]
    ):
        industry_keywords = {
            "high_priority": [
                "model",
                "price",
                "pricing",
                "pricelist",
                "vehicle",
                "car",
                "catalog",
                "vehicles",
            ],
            "medium_priority": ["specification", "specs", "feature", "compare"],
            "low_priority": ["dealer", "showroom", "service"],
            "avoid": [
                "blog",
                "news",
                "about",
                "contact",
                "career",
                "terms",
                "privacy",
                "signin",
                "login",
                "copyright",
                "search",
                "faq",
                "support",
                "events",
                "media",
            ],
        }
    # Default/generic keywords
    else:
        industry_keywords = {
            "high_priority": [
                "product",
                "service",
                "pricing",
                "price",
                "feature",
                "catalog",
                "detail",
            ],
            "medium_priority": [
                "overview",
                "info",
                "description",
                "specification",
                "comparison",
            ],
            "low_priority": ["news", "blog", "about", "contact", "career"],
            "avoid": ["terms", "privacy", "signin", "login", "copyright", "search"],
        }

    # URL path patterns that are likely to contain car model information
    model_info_patterns = [
        r"/vehicles/",
        r"/models/",
        r"/cars/",
        r"/catalog/",
        r"/pricelist",
        r"/prices",
        r"/price-list",
        r"/compare-models",
        r"/all-vehicles",
    ]

    # URL path patterns to explicitly avoid
    avoid_patterns = [
        r"/blog/",
        r"/news/",
        r"/press/",
        r"/media/",
        r"/events/",
        r"/about/",
        r"/contact/",
        r"/privacy/",
        r"/terms/",
        r"/jobs/",
        r"/careers/",
        r"/faq/",
        r"/support/",
        r"/customer-service/",
        r"/dealerships/",
        r"/locate/",
    ]

    urls_to_visit = [(base_url, 0, 10)]  # (url, depth, initial priority score)

    try:
        # Create a session for consistent headers
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

        while urls_to_visit and len(discovered_urls) < max_pages:
            # Sort by score (highest first) and get the highest scoring URL
            urls_to_visit.sort(key=lambda x: x[2], reverse=True)
            current_url, current_depth, current_score = urls_to_visit.pop(0)

            if current_url in visited_urls:
                continue

            visited_urls.add(current_url)

            # Ensure URL has a proper scheme
            if not current_url.startswith(("http://", "https://")):
                current_url = "https://" + current_url

            # Check if URL matches any avoid patterns - if so, skip it
            if any(
                re.search(pattern, current_url.lower()) for pattern in avoid_patterns
            ):
                print(
                    f"{Colors.YELLOW}Skipping URL that matches avoid pattern: {current_url}{Colors.RESET}"
                )
                continue

            print(
                f"{Colors.CYAN}Exploring: {current_url} (depth {current_depth}, score {current_score}){Colors.RESET}"
            )

            try:
                # Get the page content
                response = session.get(current_url, verify=False, timeout=20)
                if response.status_code != 200:
                    print(
                        f"{Colors.RED}Failed to access {current_url}, status code: {response.status_code}{Colors.RESET}"
                    )
                    continue

                # Add this URL to discovered URLs, but only if it's likely to contain relevant information
                # Check if URL path matches known model info patterns or has keywords in path
                is_model_info_url = any(
                    re.search(pattern, current_url.lower())
                    for pattern in model_info_patterns
                )

                # Also check for keywords in the path
                path = urlparse(current_url).path.lower()
                has_high_priority_keyword = any(
                    keyword in path for keyword in industry_keywords["high_priority"]
                )

                # Only add URL to discovered set if it matches pattern or has keyword
                if is_model_info_url or has_high_priority_keyword:
                    discovered_urls.add(current_url)
                    url_scores[current_url] = current_score
                    print(
                        f"{Colors.GREEN}Added to discovered pages: {current_url} (score: {current_score}){Colors.RESET}"
                    )

                # Home page should always be included
                elif path == "" or path == "/" or len(path) <= 1:
                    discovered_urls.add(current_url)
                    url_scores[current_url] = current_score
                    print(
                        f"{Colors.GREEN}Added home page to discovered pages: {current_url} (score: {current_score}){Colors.RESET}"
                    )
                else:
                    print(
                        f"{Colors.YELLOW}URL not added to discovered pages (not relevant to objective): {current_url}{Colors.RESET}"
                    )

                # If we've reached the max depth, don't explore further from this page
                if current_depth >= max_depth:
                    continue

                # Parse the content
                soup = BeautifulSoup(response.text, "html.parser")
                page_title = soup.title.string.lower() if soup.title else ""
                page_text = soup.get_text().lower()[
                    :1000
                ]  # Just check the first 1000 chars for keywords

                # Look for all links
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    link_text = link.get_text().strip().lower()

                    # Skip empty links, javascript, and anchors
                    if not href or href.startswith(
                        ("javascript:", "#", "mailto:", "tel:")
                    ):
                        continue

                    # Convert relative URLs to absolute
                    if not href.startswith(("http://", "https://")):
                        if href.startswith("/"):
                            # Absolute path within the domain
                            parsed_base = urlparse(current_url)
                            href = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
                        else:
                            # Relative path to current location
                            href = urljoin(current_url, href)

                    # Check if the URL is within the same domain and not already visited
                    parsed_href = urlparse(href)
                    if (
                        domain.lower() in parsed_href.netloc.lower()
                        and href not in visited_urls
                    ):
                        # Skip URLs with known file extensions that don't contain useful content
                        if any(
                            href.lower().endswith(ext)
                            for ext in [
                                ".jpg",
                                ".jpeg",
                                ".png",
                                ".gif",
                                ".pdf",
                                ".mp4",
                                ".zip",
                                ".css",
                                ".js",
                                ".ico",
                                ".svg",
                            ]
                        ):
                            continue

                        # Skip URLs with excessively long query parameters
                        if len(parsed_href.query) > 100:
                            continue

                        # Skip URLs that are likely to be avoided based on avoid patterns
                        if any(
                            re.search(pattern, href.lower())
                            for pattern in avoid_patterns
                        ):
                            continue

                        # Skip URLs that are likely to be avoided based on industry keywords
                        if any(
                            avoid_kw in parsed_href.path.lower()
                            for avoid_kw in industry_keywords["avoid"]
                        ):
                            continue

                        # Calculate a relevance score for this URL
                        score = 0  # Start with zero score and only increase for relevant URLs

                        # Check URL path against model info patterns - high boost for these
                        if any(
                            re.search(pattern, href.lower())
                            for pattern in model_info_patterns
                        ):
                            score += 30

                        # Check URL path against industry and objective keywords
                        path = parsed_href.path.lower()

                        # Boost score based on URL path matching high priority keywords
                        for keyword in industry_keywords["high_priority"]:
                            if keyword in path:
                                score += 10

                        # Additional boost if keyword is in link text and path
                        for keyword in industry_keywords["high_priority"]:
                            if keyword in path and keyword in link_text:
                                score += 5

                        # Medium boost for medium priority keywords
                        for keyword in industry_keywords["medium_priority"]:
                            if keyword in path:
                                score += 5

                        # Small boost for objective keywords
                        for keyword in objective_keywords:
                            if keyword in path or keyword in link_text:
                                score += 3

                        # Only add URLs with a positive score
                        if score > 0 and href not in discovered_urls:
                            urls_to_visit.append((href, current_depth + 1, score))

                        if len(discovered_urls) >= max_pages:
                            print(
                                f"{Colors.YELLOW}Reached limit of {max_pages} discovered URLs. Stopping discovery.{Colors.RESET}"
                            )
                            break

            except Exception as e:
                print(
                    f"{Colors.RED}Error exploring {current_url}: {str(e)}{Colors.RESET}"
                )
                continue

    except Exception as e:
        print(f"{Colors.RED}Error during subpage discovery: {str(e)}{Colors.RESET}")

    # Sort discovered URLs by their relevance score (highest first)
    sorted_urls = sorted(
        list(discovered_urls), key=lambda url: url_scores.get(url, 0), reverse=True
    )

    # Limit number of URLs to a more reasonable number (max_pages/2) to focus on most relevant ones
    if len(sorted_urls) > max_pages / 2:
        sorted_urls = sorted_urls[: int(max_pages / 2)]
        print(
            f"{Colors.YELLOW}Limited discovered URLs to top {len(sorted_urls)} most relevant.{Colors.RESET}"
        )

    print(f"{Colors.YELLOW}Total discovered URLs: {len(sorted_urls)}{Colors.RESET}")
    if sorted_urls:
        print(f"{Colors.GREEN}Top 3 most relevant URLs:{Colors.RESET}")
        for i, url in enumerate(sorted_urls[:3], 1):
            print(
                f"{Colors.GREEN}{i}. {url} (score: {url_scores.get(url, 0)}){Colors.RESET}"
            )

    return sorted_urls


def poll_firecrawl_result(extraction_id, api_key, interval=10, max_attempts=24):
    """Poll Firecrawl API to get the extraction result. Default timeout increased to 4 minutes."""
    url = f"https://api.firecrawl.dev/v1/extract/{extraction_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"{Colors.YELLOW}Polling for extraction result (Attempt {attempt}/{max_attempts})...{Colors.RESET}"
            )
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("success") and data.get("data"):
                print(f"{Colors.GREEN}Data successfully extracted:{Colors.RESET}")
                print(json.dumps(data["data"], indent=2))
                return data["data"]
            elif data.get("success") and not data.get("data"):
                print(
                    f"{Colors.YELLOW}Still processing... (Attempt {attempt}/{max_attempts}){Colors.RESET}"
                )
                time.sleep(interval)
            else:
                print(
                    f"{Colors.RED}API Error: {data.get('error', 'No error message provided')}{Colors.RESET}"
                )
                return None

        except requests.exceptions.RequestException as e:
            print(f"{Colors.RED}Request failed: {str(e)}{Colors.RESET}")
            return None
        except json.JSONDecodeError as e:
            print(f"{Colors.RED}Failed to parse response: {str(e)}{Colors.RESET}")
            return None
        except Exception as e:
            print(f"{Colors.RED}Unexpected error: {str(e)}{Colors.RESET}")
            return None

    print(
        f"{Colors.RED}Max polling attempts reached. Extraction did not complete in time (waited {max_attempts * interval} seconds).{Colors.RESET}"
    )
    return None


def extract_generic_data(url, domain, objective):
    """
    Extract generic structured data from a URL based on the objective.
    This function routes to the appropriate industry-specific extractor.

    Args:
        url (str): URL to extract data from
        domain (str): Domain name for context
        objective (str): User's objective to guide extraction

    Returns:
        dict: Extracted structured data
    """
    print(
        f"{Colors.YELLOW}Extracting generic data from {url} with objective: {objective}{Colors.RESET}"
    )

    try:
        # Request the page
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, verify=False, timeout=30, headers=headers)
        if response.status_code != 200:
            print(
                f"{Colors.RED}Failed to access {url}, status code: {response.status_code}{Colors.RESET}"
            )
            return None

        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Detect the industry type based on content and objective
        industry_type = "generic"

        # Check content indicators
        page_text = soup.get_text().lower()
        page_title = soup.title.string.lower() if soup.title else ""

        # Automotive industry indicators
        automotive_keywords = [
            "car",
            "vehicle",
            "suv",
            "sedan",
            "automotive",
            "model",
            "engine",
            "transmission",
        ]
        if (
            any(keyword in objective.lower() for keyword in automotive_keywords)
            or any(keyword in page_text[:2000] for keyword in automotive_keywords)
            or any(keyword in page_title for keyword in automotive_keywords)
        ):
            industry_type = "automotive"
            print(f"{Colors.GREEN}Detected automotive industry content{Colors.RESET}")
            return extract_automotive_data(soup, url, objective)

        # Real estate industry indicators
        real_estate_keywords = [
            "property",
            "house",
            "apartment",
            "real estate",
            "condo",
            "rent",
            "buy",
            "sqm",
            "sqft",
        ]
        if (
            any(keyword in objective.lower() for keyword in real_estate_keywords)
            or any(keyword in page_text[:2000] for keyword in real_estate_keywords)
            or any(keyword in page_title for keyword in real_estate_keywords)
        ):
            industry_type = "real_estate"
            print(f"{Colors.GREEN}Detected real estate industry content{Colors.RESET}")
            return extract_real_estate_data(soup, url, objective)

        # E-commerce industry indicators
        ecommerce_keywords = [
            "shop",
            "product",
            "store",
            "buy",
            "purchase",
            "cart",
            "price",
            "order",
        ]
        if (
            any(keyword in objective.lower() for keyword in ecommerce_keywords)
            or any(keyword in page_text[:2000] for keyword in ecommerce_keywords)
            or any(keyword in page_title for keyword in ecommerce_keywords)
        ):
            industry_type = "ecommerce"
            print(f"{Colors.GREEN}Detected e-commerce industry content{Colors.RESET}")
            return extract_ecommerce_data(soup, url, objective)

        # Travel industry indicators
        travel_keywords = [
            "travel",
            "hotel",
            "booking",
            "flight",
            "accommodation",
            "tour",
            "vacation",
        ]
        if (
            any(keyword in objective.lower() for keyword in travel_keywords)
            or any(keyword in page_text[:2000] for keyword in travel_keywords)
            or any(keyword in page_title for keyword in travel_keywords)
        ):
            industry_type = "travel"
            print(f"{Colors.GREEN}Detected travel industry content{Colors.RESET}")
            return extract_travel_data(soup, url, objective)

        # Job listing indicators
        job_keywords = [
            "job",
            "career",
            "position",
            "employment",
            "hire",
            "vacancy",
            "resume",
        ]
        if (
            any(keyword in objective.lower() for keyword in job_keywords)
            or any(keyword in page_text[:2000] for keyword in job_keywords)
            or any(keyword in page_title for keyword in job_keywords)
        ):
            industry_type = "job_listing"
            print(f"{Colors.GREEN}Detected job listing content{Colors.RESET}")
            return extract_job_data(soup, url, objective)

        # Default to generic content extraction
        print(
            f"{Colors.YELLOW}No specific industry detected, using generic extraction{Colors.RESET}"
        )
        return extract_generic_content(soup, url, objective)

    except Exception as e:
        print(
            f"{Colors.RED}Error extracting generic data from {url}: {str(e)}{Colors.RESET}"
        )
        return {"type": "generic", "error": str(e)}

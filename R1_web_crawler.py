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
    """Clean up the query by removing URLs and unnecessary characters."""
    # Remove URLs
    query = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        query,
    )
    # Remove extra whitespace
    query = " ".join(query.split())
    # Extract domain name if it was a URL
    if "cars" in query and "philippines" in query:
        return "BYD Cars Philippines"
    return query.strip()


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
    """Search Google using SerpAPI and return top results."""
    try:
        cleaned_query = clean_query(query)
        if not cleaned_query:
            print(
                f"{Colors.RED}Invalid search query. Please enter a company name rather than a URL.{Colors.RESET}"
            )
            return []

        print(f"{Colors.YELLOW}Searching Google for '{cleaned_query}'...{Colors.RESET}")

        # Construct the API URL with parameters
        params = {
            "api_key": serp_api_key,
            "q": cleaned_query,
            "engine": "google",
            "google_domain": "google.com",
            "gl": "ph",
            "hl": "en",
            "num": "10",
        }

        try:
            # Make the request
            print(f"{Colors.YELLOW}Debug: Making request to SerpAPI...{Colors.RESET}")
            response = requests.get(
                "https://serpapi.com/search.json",
                params=params,
                timeout=30,
            )

            # Print debug information
            print(
                f"{Colors.YELLOW}Debug: Response status code: {response.status_code}{Colors.RESET}"
            )

            if response.status_code == 200:
                try:
                    print(
                        f"{Colors.YELLOW}Debug: Parsing JSON response...{Colors.RESET}"
                    )
                    data = response.json()
                    print(
                        f"{Colors.YELLOW}Debug: Response data keys: {list(data.keys())}{Colors.RESET}"
                    )

                    if "error" in data:
                        print(f"{Colors.RED}API Error: {data['error']}{Colors.RESET}")
                        return []

                    print(
                        f"{Colors.YELLOW}Debug: Getting organic results...{Colors.RESET}"
                    )
                    results = data.get("organic_results", [])
                    print(
                        f"{Colors.YELLOW}Debug: Found {len(results)} results{Colors.RESET}"
                    )
                    if not results:
                        print(
                            f"{Colors.YELLOW}No results found. Try refining your search query.{Colors.RESET}"
                        )
                    return results
                except json.JSONDecodeError as e:
                    print(
                        f"{Colors.RED}Error: Invalid JSON response from API: {str(e)}{Colors.RESET}"
                    )
                    print(
                        f"{Colors.RED}Response content: {response.text[:500]}{Colors.RESET}"
                    )
                    return []
                except Exception as e:
                    print(
                        f"{Colors.RED}Unexpected error parsing response: {str(e)}{Colors.RESET}"
                    )
                    return []
            elif response.status_code == 401:
                print(
                    f"{Colors.RED}Error: Invalid API key. Please check your SERP_API_KEY in the .env file.{Colors.RESET}"
                )
                return []
            else:
                print(
                    f"{Colors.RED}Error: API request failed with status code {response.status_code}{Colors.RESET}"
                )
                try:
                    error_data = response.json()
                    print(f"{Colors.RED}Error details: {error_data}{Colors.RESET}")
                except:
                    print(
                        f"{Colors.RED}Response content: {response.text[:500]}{Colors.RESET}"
                    )
                return []

        except requests.exceptions.RequestException as e:
            print(f"{Colors.RED}Error making request: {str(e)}{Colors.RESET}")
            return []
        except Exception as e:
            print(
                f"{Colors.RED}Unexpected error during request: {str(e)}{Colors.RESET}"
            )
            return []

    except Exception as e:
        print(f"{Colors.RED}Unexpected error in search_google: {str(e)}{Colors.RESET}")
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
    """Validate if a URL is accessible, trying GET if HEAD fails."""
    try:
        # Add scheme if missing
        if not urlparse(url).scheme:
            url = "https://" + url

        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            print(f"{Colors.RED}Invalid URL structure: {url}{Colors.RESET}")
            return None

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }

        session = requests.Session()
        session.headers.update(headers)

        response = None
        try:
            # Try HEAD first (more efficient)
            response = session.head(url, timeout=10, allow_redirects=True, verify=False)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            print(
                f"{Colors.GREEN}URL HEAD validation successful: {response.url} (Status: {response.status_code}){Colors.RESET}"
            )
            return response.url
        except requests.exceptions.RequestException as head_error:
            print(
                f"{Colors.YELLOW}HEAD request failed for {url}: {head_error}. Trying GET...{Colors.RESET}"
            )
            try:
                # Fallback to GET request (stream=True to avoid downloading full content)
                response = session.get(
                    url, timeout=15, allow_redirects=True, verify=False, stream=True
                )
                response.raise_for_status()
                # Close the connection after checking status
                response.close()
                print(
                    f"{Colors.GREEN}URL GET validation successful: {response.url} (Status: {response.status_code}){Colors.RESET}"
                )
                return response.url
            except requests.exceptions.RequestException as get_error:
                print(
                    f"{Colors.RED}GET request also failed for {url}: {get_error}{Colors.RESET}"
                )
                return None
            except Exception as get_exc:
                print(
                    f"{Colors.RED}Unexpected error during GET validation for {url}: {get_exc}{Colors.RESET}"
                )
                return None
        except Exception as head_exc:
            print(
                f"{Colors.RED}Unexpected error during HEAD validation for {url}: {head_exc}{Colors.RESET}"
            )
            # Optionally try GET even on unexpected HEAD errors
            try:
                response = session.get(
                    url, timeout=15, allow_redirects=True, verify=False, stream=True
                )
                response.raise_for_status()
                response.close()
                print(
                    f"{Colors.GREEN}URL GET validation successful after unexpected HEAD error: {response.url} (Status: {response.status_code}){Colors.RESET}"
                )
                return response.url
            except Exception as fallback_get_error:
                print(
                    f"{Colors.RED}Fallback GET failed after unexpected HEAD error for {url}: {fallback_get_error}{Colors.RESET}"
                )
                return None

    except Exception as e:
        print(f"{Colors.RED}Error validating URL {url}: {str(e)}{Colors.RESET}")
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, verify=False, timeout=30, headers=headers)
        if response.status_code == 200:
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

            # Extract pricing information using various methods
            # Method 1: Look for price patterns in text
            price_patterns = [
                r"(?:PHP|₱|Php)\s*[\d,]+(?:\.\d{2})?",  # PHP currency with optional decimals
                r"(?:PHP|₱|Php)\s*[\d,]+K",  # Prices in thousands
                r"(?:PHP|₱|Php)\s*[\d,]+M",  # Prices in millions
                r"\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:PHP|₱|Php)",  # Numbers followed by currency
            ]

            # Method 2: Look for pricing tables
            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    row_text = " ".join(cell.get_text(strip=True) for cell in cells)
                    if any(re.search(pattern, row_text) for pattern in price_patterns):
                        content["pricing_info"].append(row_text)

            # Method 3: Look for price lists
            price_lists = soup.find_all(["ul", "ol"])
            for lst in price_lists:
                items = lst.find_all("li")
                for item in items:
                    item_text = item.get_text(strip=True)
                    if any(re.search(pattern, item_text) for pattern in price_patterns):
                        content["pricing_info"].append(item_text)

            # Method 4: Look for price in paragraphs
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                p_text = p.get_text(strip=True)
                if any(re.search(pattern, p_text) for pattern in price_patterns):
                    content["pricing_info"].append(p_text)

            # Get headings
            for heading in soup.find_all(["h1", "h2", "h3"]):
                text = heading.get_text(strip=True)
                if text:
                    content["headings"].append(text)
                    # Check if heading contains price information
                    if any(re.search(pattern, text) for pattern in price_patterns):
                        content["pricing_info"].append(text)

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

            # Combine all content
            content["main_content"] = "\n".join(
                content["headings"]
                + content["paragraphs"]
                + ["\n".join(lst) for lst in content["lists"]]
                + content["pricing_info"]  # Include pricing information in main content
            )

            return content
    except Exception as e:
        print(f"{Colors.RED}Error scraping {url}: {str(e)}{Colors.RESET}")
    return None


def extract_company_info(urls, prompt, company, api_key, is_input_url):
    """Use requests to call Firecrawl's extract endpoint with selected URLs."""
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
                    return result
    except Exception as e:
        print(
            f"{Colors.YELLOW}Firecrawl extraction failed, falling back to direct scraping: {str(e)}{Colors.RESET}"
        )

    # If Firecrawl fails, fall back to direct scraping
    print(f"{Colors.YELLOW}Attempting direct web scraping...{Colors.RESET}")

    # Scrape content from all valid URLs
    all_content = []
    structured_data = {
        "headings": [],
        "paragraphs": [],
        "lists": [],
        "contact_info": set(),
        "pricing_info": [],
    }

    for url in valid_urls:
        content = scrape_url(url)
        if content:
            all_content.append(content["main_content"])
            structured_data["headings"].extend(content["headings"])
            structured_data["paragraphs"].extend(content["paragraphs"])
            structured_data["lists"].extend(content["lists"])
            structured_data["contact_info"].update(content["contact_info"])
            structured_data["pricing_info"].extend(content["pricing_info"])

    if not all_content:
        print(f"{Colors.RED}Failed to scrape any content from the URLs{Colors.RESET}")
        return None

    # Use Deepseek to extract structured information
    try:
        print(
            f"{Colors.YELLOW}Processing scraped content with Deepseek...{Colors.RESET}"
        )

        # Prepare a structured prompt with the scraped data
        content_summary = {
            "title": company,
            "headings": structured_data["headings"][:10],
            "main_content": "\n".join(structured_data["paragraphs"][:5]),
            "lists": structured_data["lists"][:5],
            "contact_info": list(structured_data["contact_info"]),
            "pricing_info": structured_data["pricing_info"],
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
                    }
                }
                
                Important:
                1. Focus on extracting accurate pricing information
                2. Include all prices found in the content
                3. Organize variants and models clearly
                4. Include any financing or payment options
                5. Note any special offers or promotions
                """,
            },
            {
                "role": "user",
                "content": f"Extract information about {company} from the following structured content, paying special attention to pricing information:\n\n{json.dumps(content_summary, indent=2)}",
            },
        ]

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=1500,  # Increased token limit for more detailed response
        )

        result = response.choices[0].message.content

        # Clean up the response
        if result.startswith("```json"):
            result = result[7:-3]  # Remove ```json and ``` markers

        try:
            extracted_data = json.loads(result)
            return extracted_data
        except json.JSONDecodeError:
            print(
                f"{Colors.RED}Failed to parse Deepseek response as JSON{Colors.RESET}"
            )
            return None

    except Exception as e:
        print(
            f"{Colors.RED}Error processing content with Deepseek: {str(e)}{Colors.RESET}"
        )
        return None


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


def crawl_website(company_name, objective=None):
    """
    Crawl a website based on company name or a specific URL.

    Args:
        company_name (str): The name of the company or a specific URL to crawl.
        objective (str, optional): The objective for the crawl (e.g., find pricing). Defaults to None.

    Returns:
        dict: A dictionary containing crawled URLs and extracted data, or an error message.
    """
    try:
        urls_to_scrape = []
        company_for_prompt = company_name  # Keep original for prompt if it's a name
        is_input_url = False  # Flag to track if input is a URL
        specified_domain = None  # Store domain for filtering

        # Check if the input is a valid URL or a plain domain (e.g., example.com)
        domain_pattern = r"^[\w\-]+(\.[\w\-]+)+$"
        is_domain_input = (
            re.match(domain_pattern, company_name) and " " not in company_name
        )
        if validators.url(company_name) or is_domain_input:
            # Normalize input to full URL with scheme
            url_input = (
                company_name
                if company_name.startswith(("http://", "https://"))
                else "https://" + company_name
            )
            print(
                f"{Colors.CYAN}Input is a URL: {url_input}. Crawling directly.{Colors.RESET}"
            )
            is_input_url = True  # Set flag

            # Extract domain from URL for filtering and prompt
            parsed_url = urlparse(url_input)
            specified_domain = parsed_url.netloc
            domain_parts = specified_domain.split(".")

            # Simple heuristic: take the part before the TLD (e.g., 'google' from 'www.google.com')
            if len(domain_parts) > 1:
                company_for_prompt = domain_parts[-2]
            else:
                company_for_prompt = (
                    company_name  # Fallback to the full URL if extraction fails
                )

            # ------------------------------------------------------------------
            # Discover candidate internal links and let R1 pick the best ones
            # ------------------------------------------------------------------
            print(
                f"{Colors.MAGENTA}Debug [URL Mode]: Discovering internal links for {url_input}...{Colors.RESET}"
            )  # DEBUG
            discovered_links = discover_internal_links(url_input, max_links=50)
            print(
                f"{Colors.MAGENTA}Debug [URL Mode]: Found {len(discovered_links)} internal links.{Colors.RESET}"
            )  # DEBUG
            # print(f"{Colors.MAGENTA}Debug [URL Mode]: Discovered Links: {discovered_links[:10]}{Colors.RESET}") # Optional: Print sample links

            base_keywords = [
                "price",
                "pricing",
                "price-list",
                "pricelist",
                "plans",
                "products",
                "services",
                "models",
                "vehicles",
                "catalog",
                "showroom",
                "specs",
                "specifications",
                "brochure",
                "internet",
                "fiber",  # Added more keywords
            ]
            if objective:
                base_keywords.extend([w.lower() for w in re.findall(r"\w+", objective)])

            candidate_links = [
                (u, txt)
                for u, txt in discovered_links
                if any(
                    k in u.lower() or k in (txt or "").lower() for k in base_keywords
                )
            ]
            print(
                f"{Colors.MAGENTA}Debug [URL Mode]: Found {len(candidate_links)} candidate links after keyword filtering.{Colors.RESET}"
            )  # DEBUG
            # print(f"{Colors.MAGENTA}Debug [URL Mode]: Candidate Links: {candidate_links[:10]}{Colors.RESET}") # Optional: Print sample candidates

            serp_like = [
                {"title": t or "", "link": u, "snippet": ""}
                for u, t in (
                    candidate_links or discovered_links
                )  # Use candidates if found, else all discovered
            ]

            print(
                f"{Colors.MAGENTA}Debug [URL Mode]: Sending {len(serp_like)} links to R1 for selection...{Colors.RESET}"
            )  # DEBUG
            selected_internal = select_urls_with_r1(
                company_for_prompt, objective, serp_like
            )
            print(
                f"{Colors.MAGENTA}Debug [URL Mode]: R1 selected {len(selected_internal)} URLs: {selected_internal}{Colors.RESET}"
            )  # DEBUG

            urls_to_scrape = (
                selected_internal[:5] if selected_internal else [url_input]
            )  # Cap at 5, fallback to root
            print(
                f"{Colors.MAGENTA}Debug [URL Mode]: Final URLs to scrape before validation: {urls_to_scrape}{Colors.RESET}"
            )  # DEBUG
            # End of internal link discovery block

        else:
            # Existing logic: Input is treated as a company name
            print(
                f"{Colors.CYAN}Input is a company name: {company_name}. Searching Google...{Colors.RESET}"
            )
            search_results = search_google(company_name)
            if not search_results:
                print(
                    f"{Colors.RED}No search results found for {company_name}{Colors.RESET}"
                )
                return {"error": f"No search results found for {company_name}"}

            print(
                f"{Colors.GREEN}Google search completed. Selecting relevant URLs...{Colors.RESET}"
            )
            selected_urls = select_urls_with_r1(company_name, objective, search_results)
            if not selected_urls:
                print(f"{Colors.RED}R1 failed to select URLs.{Colors.RESET}")
                return {"error": "R1 failed to select URLs"}

            print(f"{Colors.GREEN}URLs selected by R1: {selected_urls}{Colors.RESET}")

            urls_to_scrape = selected_urls

            # >>>>> Move the company-name specific internal discovery here <<<<<
            # Identify official domains
            social_domains = [
                "facebook.com",
                "twitter.com",
                "instagram.com",
                "linkedin.com",
                "youtube.com",
                "tiktok.com",
            ]
            official_roots = []
            for u in selected_urls:
                try:
                    p = urlparse(u)
                    if not p.netloc:
                        continue
                    if any(sd in p.netloc for sd in social_domains):
                        continue
                    official_roots.append(f"{p.scheme}://{p.netloc}")
                except Exception:
                    continue
            official_roots = list(dict.fromkeys(official_roots))

            candidate_links_all = []
            base_keywords = [
                "price",
                "pricing",
                "price-list",
                "pricelist",
                "plans",
                "products",
                "services",
                "models",
                "vehicles",
                "catalog",
            ]
            if objective:
                base_keywords.extend([w.lower() for w in re.findall(r"\w+", objective)])

            for root in official_roots:
                discovered = discover_internal_links(root, max_links=50)
                filtered = [
                    (u, txt)
                    for u, txt in discovered
                    if any(
                        k in u.lower() or k in (txt or "").lower()
                        for k in base_keywords
                    )
                ]
                candidate_links_all.extend(filtered if filtered else discovered)

            if candidate_links_all:
                serp_like_internal = [
                    {"title": t or "", "link": u, "snippet": ""}
                    for u, t in candidate_links_all
                ]
                selected_internal = select_urls_with_r1(
                    company_name, objective, serp_like_internal
                )
                urls_to_scrape.extend(selected_internal)
                urls_to_scrape = list(dict.fromkeys(urls_to_scrape))

            # End of company name internal discovery block

        # ------------------------------------------------------------------
        # Validate URLs *after* final selection (applies to both branches)
        # ------------------------------------------------------------------
        validated_urls = validate_urls(urls_to_scrape)
        if not validated_urls:
            print(
                f"{Colors.RED}No valid URLs found to scrape after filtering.{Colors.RESET}"
            )
            return {"error": "No valid URLs found to scrape"}

        urls_to_scrape = validated_urls  # Use validated list moving forward

        # Determine the prompt (applies to both branches)
        prompt = f"Extract key information about {company_for_prompt}."
        if objective:
            prompt = f"Based on the objective '{objective}', extract relevant information about {company_for_prompt}. Prioritize details related to the objective."

        print(f"{Colors.MAGENTA}Using R1 prompt: {prompt}{Colors.RESET}")

        # Extract information using Firecrawl/R1
        start_time = time.time()
        print(
            f"{Colors.YELLOW}Starting extraction process for {company_for_prompt}...{Colors.RESET}"
        )
        # Note: Pass company_for_prompt to extract_company_info for better context
        # Also pass the is_input_url flag
        extracted_data = extract_company_info(
            urls_to_scrape, prompt, company_for_prompt, firecrawl_api_key, is_input_url
        )
        end_time = time.time()
        print(
            f"{Colors.GREEN}Extraction completed in {end_time - start_time:.2f} seconds.{Colors.RESET}"
        )

        if extracted_data and "error" in extracted_data:
            print(
                f"{Colors.RED}Error during extraction: {extracted_data['error']}{Colors.RESET}"
            )
            return {"error": extracted_data["error"]}

        if not extracted_data:
            print(f"{Colors.RED}Failed to extract information.{Colors.RESET}")
            return {"error": "Failed to extract information"}

        # Add crawled URLs to the result
        final_result = {"urls": urls_to_scrape, "data": extracted_data}

        # --- DEBUG PRINT ADDED ---
        print(
            f"{Colors.MAGENTA}Debug crawl_website: Returning final_result['urls']: {final_result.get('urls')}{Colors.RESET}"
        )
        # --- END DEBUG PRINT ---

        print(
            f"{Colors.BLUE}Crawl successful. Extracted data: {json.dumps(extracted_data, indent=2)}{Colors.RESET}"
        )
        return final_result

    except Exception as e:
        print(f"{Colors.RED}An error occurred in crawl_website: {str(e)}{Colors.RESET}")
        # Consider logging the full traceback here for debugging
        import traceback

        traceback.print_exc()
        return {"error": f"An internal error occurred: {str(e)}"}


def main():
    print(f"{Colors.CYAN}Welcome to the Company Information Crawler{Colors.RESET}")
    print(
        f"{Colors.CYAN}Please enter the company name (not a URL) when prompted{Colors.RESET}"
    )

    company = input(f"{Colors.BLUE}Enter the company name: {Colors.RESET}")
    if not company:
        print(f"{Colors.RED}Company name cannot be empty.{Colors.RESET}")
        return

    objective = input(
        f"{Colors.BLUE}Enter what information you want about the company: {Colors.RESET}"
    )
    if not objective:
        print(f"{Colors.RED}Search objective cannot be empty.{Colors.RESET}")
        return

    # Call the main crawling function which handles URL vs. Company Name logic
    result = crawl_website(company, objective)

    if result and "error" not in result:
        print(f"{Colors.GREEN}Extraction completed successfully.{Colors.RESET}")
        # You can optionally print the final result here if needed
        # print(f"{Colors.BLUE}Final data: {json.dumps(result.get('data'), indent=2)}{Colors.RESET}")
    else:
        error_message = (
            result.get("error", "Unknown error occurred")
            if result
            else "Unknown error occurred"
        )
        print(
            f"{Colors.RED}Failed to extract the requested information: {error_message}. Try refining your input or choosing a different company/URL.{Colors.RESET}"
        )


if __name__ == "__main__":
    main()

# -------------------------
# Helper: discover internal links for a given domain root (depth-1)
# -------------------------


def discover_internal_links(root_url, max_links=50):
    """Return a list of (url, anchor_text) tuples for same-domain links on the root page."""
    links = []
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        response = requests.get(root_url, headers=headers, timeout=15, verify=False)
        if response.status_code != 200:
            return links

        soup = BeautifulSoup(response.text, "html.parser")
        root_domain = urlparse(root_url).netloc

        for a in soup.find_all("a", href=True):
            href = urljoin(root_url, a["href"].strip())
            parsed = urlparse(href)
            if parsed.netloc and parsed.netloc.endswith(root_domain):
                clean_href = href.split("#")[0]
                anchor_text = a.get_text(" ", strip=True)[:120]
                links.append((clean_href, anchor_text))
                if len(links) >= max_links:
                    break
    except Exception as e:
        print(
            f"{Colors.YELLOW}Warning: discover_internal_links error: {e}{Colors.RESET}"
        )
    return list(dict.fromkeys(links))  # dedupe while preserving order

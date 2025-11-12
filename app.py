from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
from datetime import datetime
import json
import logging
from dotenv import load_dotenv
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from improved_web_crawler import WebCrawler
import threading
import queue
import uuid
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("WebCrawlerApp")

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for production
allowed_origins = [
    "https://crawlai.onrender.com"
]
CORS(app, origins=allowed_origins, supports_credentials=False)

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Configure port
PORT = int(os.getenv('PORT', 5002))

# Create results directory if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Validation constants
ALLOWED_LLM_MODELS = ["R1", "o4-mini"]
MAX_COMPANY_NAME_LENGTH = 200
MAX_OBJECTIVE_LENGTH = 500

# Progress tracking storage
# Format: {job_id: {"status": "running/completed/error", "current_url": "...", "progress": "1/20", "result": None, "error": None}}
progress_storage = {}
progress_queues = {}  # SSE queues for each job


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


def progress_callback(job_id, current_url, progress_text):
    """Callback function to update progress."""
    if job_id in progress_storage:
        progress_storage[job_id]["current_url"] = current_url
        progress_storage[job_id]["progress_text"] = progress_text

        # Send update to SSE queue if it exists
        if job_id in progress_queues:
            try:
                progress_queues[job_id].put({
                    "current_url": current_url,
                    "progress_text": progress_text
                }, block=False)
            except queue.Full:
                pass  # Queue full, skip this update


def run_crawl_background(job_id, entity_name, objective, llm, crawl_config, legacy_mode):
    """Run crawl in background thread."""
    try:
        logger.info(f"Starting background crawl for job {job_id}")

        # Create crawler with progress callback
        crawler = WebCrawler()

        # Create a wrapper callback
        def callback(url, progress):
            progress_callback(job_id, url, progress)

        # Run the crawl
        result = crawler.crawl_website(entity_name, objective, llm,
                                      crawl_config=crawl_config,
                                      progress_callback=callback)

        if result is None:
            logger.error(f"Crawler returned None for job {job_id}")
            progress_storage[job_id]["status"] = "error"
            progress_storage[job_id]["error"] = "Failed to crawl website"
            return

        if isinstance(result, dict) and "error" in result:
            logger.error(f"Crawler returned error for job {job_id}: {result['error']}")
            progress_storage[job_id]["status"] = "error"
            progress_storage[job_id]["error"] = result["error"]
            return

        # Handle legacy mode
        if legacy_mode and result.get('extraction_mode') == 'multi_site':
            logger.info(f"Converting multi-site result to legacy format for job {job_id}")
            result = crawler._create_legacy_compatible_result(result)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crawl_results_{timestamp}.json"
        filepath = os.path.join(RESULTS_DIR, filename)

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Successfully crawled {entity_name} for job {job_id}. Saved to {filepath}")

        # Update progress storage with result
        progress_storage[job_id]["status"] = "completed"
        progress_storage[job_id]["result"] = result
        progress_storage[job_id]["filename"] = filename

        # Send completion event to SSE
        if job_id in progress_queues:
            try:
                progress_queues[job_id].put({"status": "completed"}, block=False)
            except queue.Full:
                pass

    except Exception as e:
        logger.error(f"Error in background crawl for job {job_id}: {str(e)}", exc_info=True)
        progress_storage[job_id]["status"] = "error"
        progress_storage[job_id]["error"] = str(e)

        if job_id in progress_queues:
            try:
                progress_queues[job_id].put({"status": "error", "error": str(e)}, block=False)
            except queue.Full:
                pass


@app.route("/progress/<job_id>")
def progress_stream(job_id):
    """SSE endpoint for progress updates."""
    if job_id not in progress_storage:
        return jsonify({"error": "Invalid job ID"}), 404

    # Create a queue for this connection
    q = queue.Queue(maxsize=50)
    progress_queues[job_id] = q

    def generate():
        try:
            while True:
                # Check if job is done
                if progress_storage[job_id]["status"] in ["completed", "error"]:
                    # Send final update
                    final_data = {
                        "status": progress_storage[job_id]["status"],
                        "current_url": progress_storage[job_id].get("current_url", ""),
                        "progress_text": progress_storage[job_id].get("progress_text", "")
                    }
                    if progress_storage[job_id]["status"] == "error":
                        final_data["error"] = progress_storage[job_id].get("error", "Unknown error")
                    yield f"data: {json.dumps(final_data)}\n\n"
                    break

                # Get updates from queue with timeout
                try:
                    update = q.get(timeout=1)
                    yield f"data: {json.dumps(update)}\n\n"
                except queue.Empty:
                    # Send heartbeat
                    yield f": heartbeat\n\n"
        finally:
            # Clean up
            if job_id in progress_queues and progress_queues[job_id] is q:
                del progress_queues[job_id]

    return Response(generate(), mimetype='text/event-stream')


@app.route("/crawl", methods=["POST"])
@limiter.limit("10 per hour")
def crawl():
    """API endpoint to handle crawling requests."""
    try:
        data = request.get_json()

        # Validate input
        if not data or "company_name" not in data:
            logger.error('Missing required field "company_name"')
            return jsonify({"error": 'Missing required field "company_name"'}), 400

        # Validate and sanitize company_name
        entity_name = data["company_name"].strip()
        if not entity_name:
            return jsonify({"error": "Company name cannot be empty"}), 400
        if len(entity_name) > MAX_COMPANY_NAME_LENGTH:
            return jsonify({"error": f"Company name too long (max {MAX_COMPANY_NAME_LENGTH} characters)"}), 400

        # Validate objective
        objective = data.get("objective", "").strip()
        if len(objective) > MAX_OBJECTIVE_LENGTH:
            return jsonify({"error": f"Objective too long (max {MAX_OBJECTIVE_LENGTH} characters)"}), 400

        # Validate LLM model
        llm = data.get("llm", "R1")
        if llm not in ALLOWED_LLM_MODELS:
            return jsonify({"error": f"Invalid LLM model. Allowed values: {', '.join(ALLOWED_LLM_MODELS)}"}), 400

        legacy_mode = data.get("legacy_mode", False)  # Option for legacy compatibility
        logger.info(f"LLM model selected: {llm}, Legacy mode: {legacy_mode}")

        # Extract advanced crawl parameters
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

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Initialize progress storage
        progress_storage[job_id] = {
            "status": "running",
            "current_url": "",
            "progress_text": "Starting crawl...",
            "result": None,
            "error": None,
            "filename": None
        }

        # Start crawl in background thread
        thread = threading.Thread(
            target=run_crawl_background,
            args=(job_id, entity_name, objective, llm, crawl_config, legacy_mode),
            daemon=True
        )
        thread.start()

        # Return job ID immediately
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Crawl started. Connect to /progress/<job_id> for updates."
        })

    except Exception as e:
        logger.error(f"Error in /crawl endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred. Please try again later."}), 500


@app.route("/result/<job_id>")
def get_result(job_id):
    """Get the final result of a crawl job."""
    if job_id not in progress_storage:
        return jsonify({"error": "Invalid job ID"}), 404

    job_data = progress_storage[job_id]

    if job_data["status"] == "running":
        return jsonify({"status": "running", "message": "Crawl still in progress"}), 202

    if job_data["status"] == "error":
        return jsonify({"status": "error", "error": job_data.get("error", "Unknown error")}), 500

    # Return the complete result
    result = job_data["result"]
    if not result:
        return jsonify({"error": "No result available"}), 500

    # Prepare response similar to old format
    response_data = {
        "success": True,
        "urls": result.get("urls", []),
        "data": result.get("data", {}),
        "metadata": result.get("metadata", {}),
        "filename": job_data.get("filename"),
    }

    # Add multi-site specific information if available
    if result.get('extraction_mode') == 'multi_site':
        price_comparison = result.get("data", {}).get("price_comparison", {})
        price_summary = {}
        if price_comparison:
            for product, data in price_comparison.items():
                analysis = data.get("price_analysis", {})
                if analysis.get("lowest_price"):
                    price_summary[product] = {
                        "lowest_price_usd": analysis.get("lowest_price"),
                        "highest_price_usd": analysis.get("highest_price"),
                        "retailer_count": analysis.get("retailer_count", 0),
                        "price_recommendations": data.get("price_recommendations", [])[:2]
                    }

        response_data.update({
            "extraction_mode": "multi_site",
            "site_categories": result.get("site_categories", {}),
            "price_summary": price_summary,
            "multi_site_summary": {
                "sites_analyzed": result.get("metadata", {}).get("sites_analyzed", 0),
                "categories_found": result.get("metadata", {}).get("categories_found", []),
                "aggregated_overview": result.get("data", {}).get("consolidated_overview", "")
            }
        })
    elif result.get('extraction_mode') == 'single_site':
        response_data["extraction_mode"] = "single_site"

    return jsonify(response_data)


# Keep old synchronous endpoint logic below for reference/fallback
@app.route("/crawl_sync", methods=["POST"])
@limiter.limit("10 per hour")
def crawl_sync():
    """Synchronous crawl endpoint (old behavior, kept for compatibility)."""
    try:
        data = request.get_json()

        # Validate input
        if not data or "company_name" not in data:
            logger.error('Missing required field "company_name"')
            return jsonify({"error": 'Missing required field "company_name"'}), 400

        # Validate and sanitize company_name
        entity_name = data["company_name"].strip()
        if not entity_name:
            return jsonify({"error": "Company name cannot be empty"}), 400
        if len(entity_name) > MAX_COMPANY_NAME_LENGTH:
            return jsonify({"error": f"Company name too long (max {MAX_COMPANY_NAME_LENGTH} characters)"}), 400

        # Validate objective
        objective = data.get("objective", "").strip()
        if len(objective) > MAX_OBJECTIVE_LENGTH:
            return jsonify({"error": f"Objective too long (max {MAX_OBJECTIVE_LENGTH} characters)"}), 400

        # Validate LLM model
        llm = data.get("llm", "R1")
        if llm not in ALLOWED_LLM_MODELS:
            return jsonify({"error": f"Invalid LLM model. Allowed values: {', '.join(ALLOWED_LLM_MODELS)}"}), 400

        legacy_mode = data.get("legacy_mode", False)
        logger.info(f"LLM model selected: {llm}, Legacy mode: {legacy_mode}")

        # Extract advanced crawl parameters
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

        # Initialize and use the improved WebCrawler
        crawler = WebCrawler()
        result = crawler.crawl_website(entity_name, objective, llm, crawl_config=crawl_config)

        if result is None:
            logger.error("Crawler returned None")
            return jsonify({"error": "Failed to crawl website"}), 500

        if isinstance(result, dict) and "error" in result:
            logger.error(f"Crawler returned error: {result['error']}")
            return jsonify(result), 500

        # Handle legacy mode conversion if requested
        if legacy_mode and result.get('extraction_mode') == 'multi_site':
            logger.info("Converting multi-site result to legacy format")
            result = crawler._create_legacy_compatible_result(result)

        # Save results to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crawl_results_{timestamp}.json"
        filepath = os.path.join(RESULTS_DIR, filename)

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Successfully crawled {entity_name}. Saved results to {filepath}")

        # Prepare response with enhanced information
        response_data = {
            "success": True,
            "urls": result.get("urls", []),  # List of crawled URLs
            "data": result.get("data", {}),  # The extracted data
            "metadata": result.get("metadata", {}),  # Metadata about the crawl
            "filename": filename,  # For downloading results
        }
        
        # Add multi-site specific information if available
        if result.get('extraction_mode') == 'multi_site':
            price_comparison = result.get("data", {}).get("price_comparison", {})
            price_summary = {}
            if price_comparison:
                # Create price summary for quick overview
                for product, data in price_comparison.items():
                    analysis = data.get("price_analysis", {})
                    if analysis.get("lowest_price"):
                        price_summary[product] = {
                            "lowest_price_usd": analysis.get("lowest_price"),
                            "highest_price_usd": analysis.get("highest_price"),
                            "retailer_count": analysis.get("retailer_count", 0),
                            "price_recommendations": data.get("price_recommendations", [])[:2]  # Top 2 recommendations
                        }
            
            response_data.update({
                "extraction_mode": "multi_site",
                "site_categories": result.get("site_categories", {}),
                "price_summary": price_summary,
                "multi_site_summary": {
                    "sites_analyzed": result.get("metadata", {}).get("sites_analyzed", 0),
                    "categories_found": result.get("metadata", {}).get("categories_found", []),
                    "aggregated_overview": result.get("data", {}).get("consolidated_overview", "")
                }
            })
        elif result.get('extraction_mode') == 'single_site':
            response_data["extraction_mode"] = "single_site"
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in /crawl endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred. Please try again later."}), 500


@app.route("/download/<filename>")
def download(filename):
    """Download a saved crawl result file."""
    try:
        # Sanitize filename to prevent path traversal
        safe_filename = secure_filename(filename)

        # Validate filename format (should be crawl_results_*.json)
        if not safe_filename.startswith("crawl_results_") or not safe_filename.endswith(".json"):
            logger.warning(f"Invalid filename format requested: {filename}")
            return jsonify({"error": "Invalid filename format"}), 403

        # Construct full path
        filepath = os.path.join(RESULTS_DIR, safe_filename)

        # Ensure the resolved path is within RESULTS_DIR (prevent directory traversal)
        if not os.path.abspath(filepath).startswith(os.path.abspath(RESULTS_DIR)):
            logger.warning(f"Path traversal attempt detected: {filename}")
            return jsonify({"error": "Invalid file path"}), 403

        # Check if file exists
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {safe_filename}")
            return jsonify({"error": "File not found"}), 404

        logger.info(f"Download requested for {safe_filename}")
        return send_from_directory(RESULTS_DIR, safe_filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {str(e)}")
        return jsonify({"error": "Failed to download file"}), 500


@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Determine if we're in production
    is_production = os.getenv('RENDER', False)

    if is_production:
        logger.info(f"Starting Flask server in PRODUCTION mode on port {PORT}")
        app.run(host='0.0.0.0', port=PORT, debug=False)
    else:
        logger.info(f"Starting Flask server in DEVELOPMENT mode on http://127.0.0.1:{PORT}")
        app.run(debug=True, port=PORT)
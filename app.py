from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any, List
import logging
import json
from scraper import WebScraper
from advanced_scraper import AdvancedWebScraper

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        self.active_connections[job_id] = websocket

    def disconnect(self, job_id: str):
        if job_id in self.active_connections:
            del self.active_connections[job_id]

    async def send_progress_update(self, job_id: str, data: dict):
        if job_id in self.active_connections:
            try:
                await self.active_connections[job_id].send_text(json.dumps(data))
            except Exception as e:
                # Connection might be closed
                self.disconnect(job_id)

app = FastAPI(title="Web Scraper App", description="Scrape websites using Crawl4AI")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

scraper = WebScraper()
websocket_manager = WebSocketManager()
advanced_scraper = AdvancedWebScraper(websocket_manager)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScrapeRequest(BaseModel):
    url: HttpUrl
    word_count_threshold: Optional[int] = 10
    content_filter_threshold: Optional[float] = 0.48
    css_selector: Optional[str] = None
    wait_for: Optional[str] = "3000"
    page_timeout: Optional[int] = 30000
    excluded_tags: Optional[list] = None
    # Infinite scroll options
    enable_infinite_scroll: Optional[bool] = False
    max_scrolls: Optional[int] = 10
    scroll_delay: Optional[int] = 2000
    scroll_step: Optional[int] = 1000
    content_stability_checks: Optional[int] = 3
    youtube_optimized: Optional[bool] = False
    human_behavior_simulation: Optional[bool] = False

class ScrapeResponse(BaseModel):
    success: bool
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    markdown: Optional[str] = None
    links: Optional[list] = []
    images: Optional[list] = []
    metadata: Optional[dict] = {}
    word_count: int = 0
    error: Optional[str] = None
    infinite_scroll_enabled: Optional[bool] = False
    youtube_optimized: Optional[bool] = False
    human_behavior_simulation: Optional[bool] = False
    scroll_strategy: Optional[str] = "none"

class AdvancedCrawlRequest(BaseModel):
    url: str
    crawl_mode: str = "single"  # single, deep, sitemap, pattern
    # Infinite scroll options (for single page mode)
    enable_infinite_scroll: Optional[str] = None
    max_scrolls: Optional[int] = 10
    scroll_delay: Optional[int] = 2000
    scroll_step: Optional[int] = 1000
    content_stability_checks: Optional[int] = 3
    youtube_optimized: Optional[str] = None
    human_behavior_simulation: Optional[str] = None
    # Deep crawl options
    max_depth: Optional[int] = 2
    max_pages: Optional[int] = 20
    crawl_delay: Optional[int] = 2
    same_domain: Optional[str] = None
    include_external: Optional[str] = None
    # Sitemap options
    sitemap_url: Optional[str] = None
    sitemap_max: Optional[int] = 50
    # Pattern options
    url_pattern: Optional[str] = ".*"
    exclude_pattern: Optional[str] = None

class CrawlJobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class CrawlStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str
    completed: int
    total: int
    error: Optional[str] = None
    started_at: float
    completed_at: Optional[float] = None

class CrawlResultsResponse(BaseModel):
    job_id: str
    status: str
    total_pages: int
    results: List[Dict[str, Any]]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_url(request: ScrapeRequest):
    try:
        options = {
            "word_count_threshold": request.word_count_threshold,
            "content_filter_threshold": request.content_filter_threshold,
            "css_selector": request.css_selector,
            "wait_for": str(request.wait_for),
            "page_timeout": request.page_timeout,
            "excluded_tags": request.excluded_tags or ["nav", "footer", "aside", "script", "style"],
            "enable_infinite_scroll": request.enable_infinite_scroll,
            "max_scrolls": request.max_scrolls,
            "scroll_delay": request.scroll_delay,
            "scroll_step": request.scroll_step,
            "content_stability_checks": request.content_stability_checks,
            "youtube_optimized": request.youtube_optimized,
            "human_behavior_simulation": request.human_behavior_simulation
        }
        
        result = await scraper.scrape_url(str(request.url), options)
        return ScrapeResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in scrape endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

@app.post("/scrape-form")
async def scrape_form(
    request: Request, 
    url: str = Form(...),
    crawl_mode: str = Form("single"),
    enable_infinite_scroll: Optional[str] = Form(None),
    max_scrolls: Optional[int] = Form(10),
    scroll_delay: Optional[int] = Form(2000),
    scroll_step: Optional[int] = Form(1000),
    content_stability_checks: Optional[int] = Form(3),
    youtube_optimized: Optional[str] = Form(None),
    human_behavior_simulation: Optional[str] = Form(None)
):
    try:
        if not scraper.is_valid_url(url):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Please enter a valid URL"
            })
        
        # Handle single page mode with infinite scroll
        if crawl_mode == "single":
            options = {
                "enable_infinite_scroll": enable_infinite_scroll == "on",
                "max_scrolls": max_scrolls,
                "scroll_delay": scroll_delay,
                "scroll_step": scroll_step,
                "content_stability_checks": content_stability_checks,
                "youtube_optimized": youtube_optimized == "on",
                "human_behavior_simulation": human_behavior_simulation == "on"
            }
            
            result = await scraper.scrape_url(url, options)
            
            return templates.TemplateResponse("index.html", {
                "request": request,
                "result": result,
                "url": url
            })
        else:
            # For advanced crawl modes, redirect to the advanced endpoint
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Advanced crawl modes should use the advanced endpoint",
                "url": url
            })
        
    except Exception as e:
        logger.error(f"Error in form scraping: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Scraping failed: {str(e)}",
            "url": url
        })

@app.post("/crawl/advanced", response_model=CrawlJobResponse)
async def start_advanced_crawl(request: AdvancedCrawlRequest):
    """Start an advanced crawling job"""
    try:
        # Convert request to dict for processing
        config = request.model_dump()
        
        # Start the advanced crawl
        job_id = await advanced_scraper.start_advanced_crawl(config)
        
        return CrawlJobResponse(
            job_id=job_id,
            status="started",
            message="Crawl job started successfully"
        )
        
    except Exception as e:
        logger.error(f"Error starting advanced crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start crawl: {str(e)}")

@app.get("/crawl/status/{job_id}", response_model=CrawlStatusResponse)
async def get_crawl_status(job_id: str):
    """Get the status of a crawling job"""
    try:
        status = advanced_scraper.get_job_status(job_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return CrawlStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting crawl status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.get("/crawl/results/{job_id}", response_model=CrawlResultsResponse)
async def get_crawl_results(job_id: str):
    """Get the results of a completed crawling job"""
    try:
        status = advanced_scraper.get_job_status(job_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        results = advanced_scraper.get_job_results(job_id) or []
        
        return CrawlResultsResponse(
            job_id=job_id,
            status=status["status"],
            total_pages=len(results),
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting crawl results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")

@app.get("/crawl/results/{job_id}/page")
async def get_crawl_results_page(job_id: str, request: Request):
    """Display crawl results in a web page"""
    try:
        status = advanced_scraper.get_job_status(job_id)
        
        if not status:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Crawl job not found"
            })
        
        results = advanced_scraper.get_job_results(job_id) or []
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "job_id": job_id,
            "status": status,
            "results": results,
            "total_pages": len(results)
        })
        
    except Exception as e:
        logger.error(f"Error displaying crawl results: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error loading results: {str(e)}"
        })

@app.post("/crawl/cancel/{job_id}")
async def cancel_crawl(job_id: str):
    """Cancel a running crawling job"""
    try:
        success = advanced_scraper.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or not running")
        
        return {"status": "cancelled", "message": "Crawl job cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel crawl: {str(e)}")

@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await websocket_manager.connect(websocket, job_id)
    try:
        while True:
            # Keep connection alive and wait for messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(job_id)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "web-scraper"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Infinite Scroll Feature Documentation

## Overview

The web scraper now supports **automatic infinite scrolling** for single-page crawling. This feature allows the crawler to automatically scroll down on web pages to trigger the loading of lazy-loaded content and infinite scroll elements that are not initially visible.

## Key Features

### 🔄 **Progressive Scrolling**
- Incrementally scrolls down the page to mimic natural user behavior
- Waits for new content to load after each scroll action
- Configurable scroll distance and timing

### 🛡️ **Content Stability Detection**
- Monitors page content changes after each scroll
- Stops scrolling when no new content is detected
- Prevents infinite loops with maximum scroll limits

### ⚙️ **Configurable Parameters**
- **Max Scrolls**: Maximum number of scroll attempts (default: 10)
- **Scroll Delay**: Time to wait after each scroll for content to load (default: 2000ms)
- **Scroll Step**: Distance to scroll down on each step (default: 1000px)
- **Content Stability Checks**: Number of consecutive checks with no new content before stopping (default: 3)

## How It Works

1. **Initial Page Load**: The page loads normally with standard wait times
2. **Scroll Detection**: JavaScript is injected to handle scrolling behavior
3. **Progressive Scrolling**: The page is scrolled down incrementally
4. **Content Monitoring**: After each scroll, the system checks if new content has loaded
5. **Stability Check**: If no new content is detected for N consecutive checks, scrolling stops
6. **Final Extraction**: The page is scrolled back to the top and content is extracted

## Configuration Options

### Web Interface
When using the web interface:
1. Select "Single Page" crawling mode
2. Expand "Advanced Crawling Options"
3. Check "Enable Infinite Scroll"
4. Adjust parameters as needed:
   - **Max Scrolls**: 3-50 (prevents infinite loops)
   - **Scroll Delay**: 500-5000ms (time to wait for content)
   - **Scroll Step**: 500-2000px (scroll distance)
   - **Content Stability Checks**: 1-10 (stability detection)

### API Usage
```python
# Using the Python API
scraper = WebScraper()
result = await scraper.scrape_url("https://example.com", {
    "enable_infinite_scroll": True,
    "max_scrolls": 10,
    "scroll_delay": 2000,
    "scroll_step": 1000,
    "content_stability_checks": 3
})
```

```json
// Using the REST API
{
    "url": "https://example.com",
    "enable_infinite_scroll": true,
    "max_scrolls": 10,
    "scroll_delay": 2000,
    "scroll_step": 1000,
    "content_stability_checks": 3
}
```

## Use Cases

### ✅ **Ideal For**
- Social media feeds (Twitter, Facebook, Instagram)
- News websites with infinite scroll
- E-commerce product listings
- Blog feeds and article lists
- Search result pages with lazy loading
- Image galleries with progressive loading

### ⚠️ **Considerations**
- Increases crawling time (due to scroll delays)
- May not work on all infinite scroll implementations
- Some sites may have anti-bot measures
- Respects robots.txt and rate limiting

## Best Practices

### 🎯 **Performance Tips**
- Start with lower scroll counts (3-5) for testing
- Adjust scroll delay based on page loading speed
- Use appropriate scroll step size for the page layout
- Monitor crawl time vs. content gain ratio

### 🔧 **Troubleshooting**
- If no additional content loads, try increasing scroll delay
- If crawling takes too long, reduce max scrolls
- Check browser console for JavaScript errors
- Verify the page actually has infinite scroll functionality

## Technical Implementation

### JavaScript Injection
The feature uses JavaScript injection to:
```javascript
// Scroll detection and progressive loading
while (scrollCount < maxScrolls && !isAtBottom()) {
    let currentHeight = getContentHeight();
    window.scrollBy(0, scrollStep);
    await sleep(scrollDelay);
    
    let newHeight = getContentHeight();
    if (newHeight > currentHeight) {
        stableCount = 0;  // New content detected
    } else {
        stableCount++;    // No new content
        if (stableCount >= stabilityChecks) break;
    }
}
```

### Content Height Detection
Uses multiple methods to detect page height:
- `document.body.scrollHeight`
- `document.body.offsetHeight`
- `document.documentElement.scrollHeight`
- `document.documentElement.offsetHeight`

## Security & Ethics

### 🛡️ **Built-in Safeguards**
- Respects robots.txt files
- Implements crawl delays
- Limited scroll attempts prevent infinite loops
- Graceful error handling

### 📋 **Ethical Usage**
- Only use on sites you have permission to crawl
- Respect website terms of service
- Be mindful of server load
- Consider the site's bandwidth costs

## API Response Changes

When infinite scroll is enabled, the API response includes:
```json
{
    "success": true,
    "url": "https://example.com",
    "title": "Page Title",
    "markdown": "...content...",
    "word_count": 1500,
    "infinite_scroll_enabled": true,
    ...
}
```

## Testing

Use the included test script to verify functionality:
```bash
python test_infinite_scroll.py
```

This will test both normal and infinite scroll modes on sample URLs to demonstrate the feature's behavior.

---

*This feature enhances the web scraper's ability to handle modern JavaScript-heavy websites while maintaining ethical crawling practices.*
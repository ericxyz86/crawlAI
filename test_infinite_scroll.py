#!/usr/bin/env python3

import asyncio
import json
from scraper import WebScraper

async def test_infinite_scroll():
    """Test the infinite scroll functionality"""
    scraper = WebScraper()
    
    # Test URLs with infinite scroll
    test_urls = [
        "https://httpbin.org/html",  # Simple test page
        "https://example.com",       # Basic static page
    ]
    
    for url in test_urls:
        print(f"\n{'='*50}")
        print(f"Testing URL: {url}")
        print('='*50)
        
        # Test without infinite scroll
        print("\n1. Testing WITHOUT infinite scroll:")
        result_normal = await scraper.scrape_url(url, {
            "enable_infinite_scroll": False
        })
        print(f"   Success: {result_normal['success']}")
        print(f"   Word count: {result_normal['word_count']}")
        print(f"   Infinite scroll enabled: {result_normal.get('infinite_scroll_enabled', False)}")
        
        # Test with infinite scroll
        print("\n2. Testing WITH infinite scroll:")
        result_scroll = await scraper.scrape_url(url, {
            "enable_infinite_scroll": True,
            "max_scrolls": 3,
            "scroll_delay": 1000,
            "scroll_step": 500,
            "content_stability_checks": 2
        })
        print(f"   Success: {result_scroll['success']}")
        print(f"   Word count: {result_scroll['word_count']}")
        print(f"   Infinite scroll enabled: {result_scroll.get('infinite_scroll_enabled', False)}")
        
        # Compare results
        if result_normal['success'] and result_scroll['success']:
            word_diff = result_scroll['word_count'] - result_normal['word_count']
            print(f"   Word count difference: {word_diff}")
            
            if word_diff > 0:
                print("   ✅ Infinite scroll loaded additional content!")
            else:
                print("   ℹ️  No additional content loaded (normal for static pages)")
        else:
            print("   ❌ One or both tests failed")
            if not result_normal['success']:
                print(f"   Normal scraping error: {result_normal.get('error', 'Unknown')}")
            if not result_scroll['success']:
                print(f"   Scroll scraping error: {result_scroll.get('error', 'Unknown')}")

if __name__ == "__main__":
    asyncio.run(test_infinite_scroll())
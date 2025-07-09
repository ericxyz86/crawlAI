#!/usr/bin/env python3

import asyncio
import json
import logging
from scraper import WebScraper

# Enable debug logging
logging.basicConfig(level=logging.INFO)

async def test_youtube_infinite_scroll():
    """Test the YouTube-specific infinite scroll functionality"""
    scraper = WebScraper()
    
    # Test YouTube URLs
    test_urls = [
        "https://www.youtube.com/@PLDTHome/videos",  # The problematic URL from the user
        "https://www.youtube.com/@TEDx/videos",     # Alternative test URL
    ]
    
    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"Testing YouTube URL: {url}")
        print('='*60)
        
        # Test without infinite scroll first
        print("\n1. Testing WITHOUT infinite scroll:")
        result_normal = await scraper.scrape_url(url, {
            "enable_infinite_scroll": False
        })
        print(f"   Success: {result_normal['success']}")
        if result_normal['success']:
            print(f"   Word count: {result_normal['word_count']}")
            print(f"   YouTube optimized: {result_normal.get('youtube_optimized', False)}")
        else:
            print(f"   Error: {result_normal.get('error', 'Unknown error')}")
        
        # Test with YouTube-optimized infinite scroll
        print("\n2. Testing WITH YouTube-optimized infinite scroll:")
        result_youtube = await scraper.scrape_url(url, {
            "enable_infinite_scroll": True,
            "youtube_optimized": True,
            "human_behavior_simulation": True,
            "max_scrolls": 15,  # Reduced for testing
            "scroll_delay": 3000,
            "scroll_step": 700,
            "content_stability_checks": 5
        })
        print(f"   Success: {result_youtube['success']}")
        if result_youtube['success']:
            print(f"   Word count: {result_youtube['word_count']}")
            print(f"   YouTube optimized: {result_youtube.get('youtube_optimized', False)}")
            print(f"   Human behavior: {result_youtube.get('human_behavior_simulation', False)}")
            print(f"   Scroll strategy: {result_youtube.get('scroll_strategy', 'unknown')}")
            
            # Check if we got more content
            if result_normal['success']:
                word_diff = result_youtube['word_count'] - result_normal['word_count']
                print(f"   Word count improvement: {word_diff}")
                
                if word_diff > 0:
                    print("   ✅ YouTube optimization loaded additional content!")
                else:
                    print("   ⚠️  No additional content loaded - may need adjustment")
            
            # Show a sample of the content structure
            markdown_preview = result_youtube.get('markdown', '')[:500]
            if markdown_preview:
                print(f"\n   Content preview (first 500 chars):")
                print(f"   {markdown_preview}...")
            
        else:
            print(f"   Error: {result_youtube.get('error', 'Unknown error')}")
            print(f"   Scroll strategy: {result_youtube.get('scroll_strategy', 'unknown')}")
        
        # Test with generic scroll for comparison
        print("\n3. Testing WITH generic infinite scroll (fallback):")
        result_generic = await scraper.scrape_url(url, {
            "enable_infinite_scroll": True,
            "youtube_optimized": False,
            "human_behavior_simulation": True,
            "max_scrolls": 10,
            "scroll_delay": 2500,
            "scroll_step": 1000,
            "content_stability_checks": 3
        })
        print(f"   Success: {result_generic['success']}")
        if result_generic['success']:
            print(f"   Word count: {result_generic['word_count']}")
            print(f"   YouTube optimized: {result_generic.get('youtube_optimized', False)}")
            print(f"   Scroll strategy: {result_generic.get('scroll_strategy', 'unknown')}")
        else:
            print(f"   Error: {result_generic.get('error', 'Unknown error')}")
        
        # Summary comparison
        print(f"\n📊 RESULTS SUMMARY for {url}:")
        if result_normal['success']:
            print(f"   Normal scraping: {result_normal['word_count']} words")
        if result_youtube['success']:
            print(f"   YouTube optimized: {result_youtube['word_count']} words")
        if result_generic['success']:
            print(f"   Generic scroll: {result_generic['word_count']} words")
        
        # Determine the best approach
        best_count = 0
        best_method = "normal"
        
        if result_normal['success'] and result_normal['word_count'] > best_count:
            best_count = result_normal['word_count']
            best_method = "normal"
        
        if result_youtube['success'] and result_youtube['word_count'] > best_count:
            best_count = result_youtube['word_count']
            best_method = "YouTube optimized"
        
        if result_generic['success'] and result_generic['word_count'] > best_count:
            best_count = result_generic['word_count']
            best_method = "generic scroll"
        
        print(f"   🏆 Best method: {best_method} ({best_count} words)")
        
        print(f"\n" + "="*60)

if __name__ == "__main__":
    asyncio.run(test_youtube_infinite_scroll())
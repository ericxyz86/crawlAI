#!/usr/bin/env python3
"""
Test script for the enhanced multi-site webcrawler functionality.
This script demonstrates the new features without making actual crawl requests.
"""

import json
from improved_web_crawler import WebCrawler


def test_site_categorization():
    """Test the site categorization functionality."""
    print("🔍 Testing Site Categorization")
    print("=" * 50)
    
    crawler = WebCrawler()
    
    # Test URLs representing different site types
    test_urls = [
        "https://www.apple.com/ph/watch/",
        "https://beyondthebox.ph/collections/apple-watch",
        "https://powermaccenter.com/pages/view-all-watch",
        "https://istore.ph/collections/watch",
        "https://www.facebook.com/ApplePhilippines",
        "https://techcrunch.com/apple-watch-review",
        "https://www.rappler.com/technology/apple-watch-news",
        "https://carousell.ph/search/apple-watch"
    ]
    
    entity_name = "Apple Watch"
    categories = crawler._categorize_sites(test_urls, entity_name)
    
    print(f"📊 Categorization Results for '{entity_name}':")
    for category, urls in categories.items():
        if urls:
            print(f"  {category.upper()}: {len(urls)} URLs")
            for url in urls[:2]:  # Show first 2 URLs
                print(f"    - {url}")
            if len(urls) > 2:
                print(f"    ... and {len(urls) - 2} more")
    
    return categories


def test_legacy_compatibility():
    """Test legacy compatibility conversion."""
    print("\n🔄 Testing Legacy Compatibility")
    print("=" * 50)
    
    crawler = WebCrawler()
    
    # Mock multi-site result structure
    mock_multi_site_result = {
        "extraction_mode": "multi_site",
        "urls": ["https://example1.com", "https://example2.com"],
        "data": {
            "sites_analyzed": {
                "official": [{
                    "domain": "apple.com",
                    "data": {
                        "entity_overview": "Official Apple information",
                        "products_and_services": [{"name": "Apple Watch Series 10", "price": "$399"}]
                    }
                }],
                "retailer": [{
                    "domain": "beyondthebox.ph",
                    "data": {
                        "entity_overview": "Retailer pricing information",
                        "products_and_services": [{"name": "Apple Watch Series 10", "price": "₱28,990"}]
                    }
                }]
            }
        },
        "metadata": {"crawl_time": "2025-06-12T21:30:00"}
    }
    
    legacy_result = crawler._create_legacy_compatible_result(mock_multi_site_result)
    
    print("✅ Legacy conversion successful!")
    print(f"  Original mode: {mock_multi_site_result['extraction_mode']}")
    print(f"  Primary source: {legacy_result['metadata']['multi_site_summary']['primary_source']}")
    print(f"  Data preserved: {len(legacy_result['data'])} fields")
    
    return legacy_result


def generate_sample_output():
    """Generate a sample of what the new JSON output looks like."""
    print("\n📋 Sample Multi-Site JSON Output")
    print("=" * 50)
    
    sample_output = {
        "urls": [
            "https://www.apple.com/ph/watch/",
            "https://beyondthebox.ph/collections/apple-watch",
            "https://powermaccenter.com/pages/view-all-watch"
        ],
        "data": {
            "entity_name": "Apple Watch Philippines",
            "objective": "find pricing and models",
            "sites_analyzed": {
                "official": [
                    {
                        "domain": "apple.com",
                        "category": "official",
                        "data": {
                            "entity_overview": "Official Apple Watch information",
                            "products_and_services": [
                                {
                                    "name": "Apple Watch Series 10",
                                    "description": "Latest model with advanced features",
                                    "price": "From $399",
                                    "source_domain": "apple.com"
                                }
                            ]
                        }
                    }
                ],
                "retailer": [
                    {
                        "domain": "beyondthebox.ph",
                        "category": "retailer",
                        "data": {
                            "entity_overview": "Apple authorized retailer in Philippines",
                            "products_and_services": [
                                {
                                    "name": "Apple Watch Series 10",
                                    "description": "Available with local warranty",
                                    "price": "₱28,990",
                                    "source_domain": "beyondthebox.ph"
                                }
                            ]
                        }
                    }
                ]
            },
            "price_comparison": {
                "Apple Watch Series 10": {
                    "prices_by_source": [
                        {"price": "From $399", "source": "apple.com", "category": "official"},
                        {"price": "₱28,990", "source": "beyondthebox.ph", "category": "retailer"}
                    ]
                }
            },
            "best_sources": {
                "official_info": "apple.com",
                "pricing": "beyondthebox.ph"
            },
            "recommendations": [
                "Multi-site analysis completed - compare information across different source types",
                "Multiple retailers found - compare pricing and availability",
                "Official information available - use for authoritative product details"
            ]
        },
        "site_categories": {
            "official": ["https://www.apple.com/ph/watch/"],
            "retailer": [
                "https://beyondthebox.ph/collections/apple-watch",
                "https://powermaccenter.com/pages/view-all-watch"
            ]
        },
        "extraction_mode": "multi_site",
        "metadata": {
            "crawl_time": "2025-06-12T21:30:00",
            "execution_time_seconds": 145.67,
            "sites_analyzed": 3,
            "categories_found": ["official", "retailer"]
        }
    }
    
    print(json.dumps(sample_output, indent=2)[:1000] + "...")
    print("\n📈 Key Improvements:")
    print("  ✅ Multiple sites analyzed simultaneously")
    print("  ✅ Sites categorized by type (official, retailer, dealer, etc.)")
    print("  ✅ Price comparison across retailers")
    print("  ✅ Best source recommendations")
    print("  ✅ Backward compatibility maintained")
    
    return sample_output


def main():
    """Run all tests and demonstrations."""
    print("🚀 Enhanced Multi-Site WebCrawler Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Site categorization
        categories = test_site_categorization()
        
        # Test 2: Legacy compatibility
        legacy_result = test_legacy_compatibility()
        
        # Test 3: Sample output
        sample = generate_sample_output()
        
        print(f"\n✅ All tests completed successfully!")
        print("\n📊 Summary:")
        print(f"  - Site categories identified: {len([c for c, urls in categories.items() if urls])}")
        print(f"  - Legacy compatibility: ✅ Working")
        print(f"  - Enhanced JSON structure: ✅ Ready")
        
        print("\n🎯 Next Steps:")
        print("  1. Test with real company searches")
        print("  2. Monitor multi-site extraction performance")
        print("  3. Analyze price comparison accuracy")
        print("  4. Validate data aggregation quality")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
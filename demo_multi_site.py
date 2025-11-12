#!/usr/bin/env python3
"""
Demonstration script for the enhanced multi-site web crawler functionality.
Shows how the crawler now performs comprehensive multi-site searches with price comparison.
"""

import json
import sys
import os
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_web_crawler import WebCrawler


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"🔍 {title}")
    print("=" * 60)


def print_price_comparison(price_data):
    """Print formatted price comparison data."""
    if not price_data:
        print("No pricing data found.")
        return
    
    for product_name, data in price_data.items():
        print(f"\n📱 {product_name}")
        print("-" * 40)
        
        # Price analysis
        analysis = data.get('price_analysis', {})
        if analysis.get('lowest_price'):
            print(f"💰 Price Range: ${analysis['lowest_price']:.2f} - ${analysis['highest_price']:.2f} USD")
            print(f"📊 Average Price: ${analysis['average_price']:.2f} USD")
            print(f"🏪 Retailers Found: {analysis['retailer_count']}")
            
            if analysis.get('official_price'):
                print(f"🏢 Official Price: ${analysis['official_price']:.2f} USD")
        
        # Price sources
        prices_by_source = data.get('prices_by_source', [])
        if prices_by_source:
            print(f"\n📍 Price Sources:")
            for price_info in prices_by_source[:5]:  # Show top 5
                source = price_info.get('source', 'Unknown')
                price = price_info.get('price', 'N/A')
                category = price_info.get('category', 'unknown')
                print(f"  • {source} ({category}): {price}")
        
        # Recommendations
        recommendations = data.get('price_recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"  • {rec}")


def print_site_analysis(sites_data):
    """Print formatted site analysis data."""
    if not sites_data:
        print("No site analysis data found.")
        return
    
    for category, sites in sites_data.items():
        if sites:
            print(f"\n🌐 {category.title()} Sites ({len(sites)}):")
            for site in sites[:3]:  # Show first 3 sites per category
                domain = site.get('domain', 'Unknown')
                print(f"  • {domain}")
                
                # Show brief data preview
                site_data = site.get('data', {})
                if site_data.get('entity_overview'):
                    overview = site_data['entity_overview'][:100] + "..." if len(site_data['entity_overview']) > 100 else site_data['entity_overview']
                    print(f"    Overview: {overview}")


def demonstrate_search_enhancement():
    """Demonstrate the enhanced search capabilities."""
    print_section_header("ENHANCED SEARCH DEMONSTRATION")
    
    crawler = WebCrawler()
    search_engine = crawler.search_engine
    
    # Demo 1: Region and Category Detection
    print("\n🎯 Region & Category Detection:")
    test_queries = [
        ("Apple Watch Philippines", "find pricing"),
        ("Toyota Camry USA", "find dealers"),
        ("Samsung Galaxy", "compare features")
    ]
    
    for query, objective in test_queries:
        region, category = search_engine.detect_region_and_category(query, objective)
        print(f"  '{query}' + '{objective}' → Region: {region}, Category: {category}")
    
    # Demo 2: Retailer Query Generation
    print("\n🔍 Retailer Query Generation:")
    queries = search_engine.generate_retailer_queries("Apple Watch", "find pricing", max_queries=4)
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")


def demonstrate_price_comparison():
    """Demonstrate the price comparison engine."""
    print_section_header("PRICE COMPARISON ENGINE DEMONSTRATION")
    
    crawler = WebCrawler()
    
    # Demo price parsing
    print("\n💱 Price Parsing Examples:")
    test_prices = ["$399.99", "₱28,990", "€350", "Not specified", "Contact for pricing"]
    
    for price_str in test_prices:
        result = crawler._parse_price_string(price_str)
        if result:
            numeric, currency = result
            usd_equivalent = crawler._normalize_to_usd(numeric, currency)
            print(f"  '{price_str}' → {numeric} {currency} (${usd_equivalent:.2f} USD)")
        else:
            print(f"  '{price_str}' → Unable to parse")
    
    # Demo price comparison
    print("\n📊 Price Comparison Example:")
    mock_prices = [
        {
            'product_name': 'Apple Watch Series 10',
            'price': '$399',
            'source_domain': 'apple.com',
            'source_category': 'official',
            'variants': []
        },
        {
            'product_name': 'Apple Watch Series 10',
            'price': '₱28,990',
            'source_domain': 'powermaccenter.com',
            'source_category': 'retailer',
            'variants': []
        },
        {
            'product_name': 'Apple Watch Series 10',
            'price': '$429',
            'source_domain': 'bestbuy.com',
            'source_category': 'retailer',
            'variants': []
        }
    ]
    
    price_comparison = crawler._create_advanced_price_comparison(mock_prices)
    print_price_comparison(price_comparison)


def demonstrate_live_crawling():
    """Demonstrate live multi-site crawling (requires API keys)."""
    print_section_header("LIVE MULTI-SITE CRAWLING DEMONSTRATION")
    
    # Check if API keys are available
    if not os.getenv('SERP_API_KEY'):
        print("⚠️  SERP_API_KEY not found in environment variables.")
        print("This demo requires API keys to perform live crawling.")
        print("\nTo enable live crawling:")
        print("1. Add your API keys to a .env file:")
        print("   SERP_API_KEY=your_serpapi_key")
        print("   DEEPSEEK_API_KEY=your_deepseek_key")
        print("2. Run this demo again")
        return
    
    print("🌐 Performing live multi-site crawl...")
    print("Query: 'Apple Watch Philippines'")
    print("Objective: 'find pricing and availability'")
    
    try:
        crawler = WebCrawler()
        
        # Perform the crawl
        start_time = datetime.now()
        result = crawler.crawl_website("Apple Watch Philippines", "find pricing and availability")
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        print(f"\n⏱️  Crawl completed in {execution_time:.2f} seconds")
        
        if result and isinstance(result, dict):
            # Display extraction mode
            extraction_mode = result.get('extraction_mode', 'unknown')
            print(f"🔧 Extraction Mode: {extraction_mode}")
            
            # Display sites analyzed
            metadata = result.get('metadata', {})
            sites_analyzed = metadata.get('sites_analyzed', 0)
            categories_found = metadata.get('categories_found', [])
            print(f"🌐 Sites Analyzed: {sites_analyzed}")
            print(f"📂 Categories Found: {', '.join(categories_found)}")
            
            # Display price comparison if available
            data = result.get('data', {})
            if 'price_comparison' in data:
                print_price_comparison(data['price_comparison'])
            
            # Display site analysis if available
            if 'sites_analyzed' in data:
                print_site_analysis(data['sites_analyzed'])
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"demo_results_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Detailed results saved to: {filename}")
            
        else:
            print("❌ Crawl failed or returned invalid data")
            if isinstance(result, dict) and 'error' in result:
                print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"❌ Error during live crawling: {str(e)}")
        print("This might be due to missing API keys or network issues.")


def show_api_examples():
    """Show API usage examples."""
    print_section_header("API USAGE EXAMPLES")
    
    print("🚀 Flask API Examples:")
    print("\n1. Multi-site crawling (default):")
    print("""
curl -X POST http://localhost:5001/crawl \\
  -H "Content-Type: application/json" \\
  -d '{
    "company_name": "Apple Watch Philippines",
    "objective": "find pricing and availability"
  }'
""")
    
    print("\n2. Legacy compatibility mode:")
    print("""
curl -X POST http://localhost:5001/crawl \\
  -H "Content-Type: application/json" \\
  -d '{
    "company_name": "Apple Watch",
    "objective": "find pricing",
    "legacy_mode": true
  }'
""")
    
    print("\n3. Expected multi-site response structure:")
    example_response = {
        "success": True,
        "extraction_mode": "multi_site",
        "urls": ["https://apple.com/watch", "https://powermac.com.ph/apple-watch"],
        "price_summary": {
            "Apple Watch Series 10": {
                "lowest_price_usd": 399.99,
                "highest_price_usd": 521.82,
                "retailer_count": 2,
                "price_recommendations": ["Best price found at apple.com ($399)"]
            }
        },
        "site_categories": {
            "official": ["https://apple.com/watch"],
            "retailer": ["https://powermac.com.ph/apple-watch"]
        },
        "multi_site_summary": {
            "sites_analyzed": 2,
            "categories_found": ["official", "retailer"]
        }
    }
    print(json.dumps(example_response, indent=2))


def main():
    """Main demonstration function."""
    print("🎯 MULTI-SITE WEB CRAWLER DEMONSTRATION")
    print("This demo showcases the enhanced multi-site crawling capabilities")
    print("that provide comprehensive company information and price comparison.")
    
    # Run demonstrations
    demonstrate_search_enhancement()
    demonstrate_price_comparison()
    show_api_examples()
    
    # Ask user if they want to try live crawling
    print_section_header("LIVE CRAWLING OPTION")
    print("Would you like to try live multi-site crawling?")
    print("⚠️  This requires valid API keys (SERP_API_KEY, DEEPSEEK_API_KEY)")
    
    response = input("\nProceed with live crawling? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        demonstrate_live_crawling()
    else:
        print("✅ Demo completed. Live crawling skipped.")
        print("\nTo enable live crawling, set up your API keys and run this demo again.")
    
    print("\n🎉 Multi-site crawler demonstration complete!")
    print("The enhanced crawler now provides:")
    print("  • Comprehensive multi-site search across retailers")
    print("  • Advanced price comparison with currency conversion")
    print("  • Site categorization and intelligent aggregation")
    print("  • Backward compatibility for existing applications")


if __name__ == '__main__':
    main()
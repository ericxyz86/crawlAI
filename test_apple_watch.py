#!/usr/bin/env python3
"""
Test script specifically for Apple Watch pricing extraction
"""
import requests
import json
import time

def test_apple_watch_pricing():
    """Test Apple Watch pricing extraction through the API"""
    print("Testing Apple Watch Pricing Extraction")
    print("=" * 50)
    
    # Test data - Apple Watch Philippines
    test_data = {
        "company_name": "apple watch philippines",
        "objective": "find apple watch models, features, and pricing in Philippines",
        "llm": "R1"
    }
    
    print(f"Testing: {test_data['company_name']}")
    print(f"Objective: {test_data['objective']}")
    print(f"LLM Model: {test_data['llm']}")
    print("-" * 50)
    
    try:
        # Make request to the Flask API
        response = requests.post(
            "http://127.0.0.1:5001/crawl",
            json=test_data,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return False
            else:
                print("✅ Success! Crawler completed successfully")
                print(f"📊 URLs crawled: {len(result.get('urls', []))}")
                
                # Print the URLs that were crawled
                urls = result.get('urls', [])
                print("\n🔗 URLs crawled:")
                for i, url in enumerate(urls[:5], 1):
                    print(f"  {i}. {url}")
                
                # Check if we got meaningful data
                data = result.get('data', {})
                if data:
                    print(f"\n📝 Entity overview: {data.get('entity_overview', 'N/A')[:150]}...")
                    
                    products = data.get('products_and_services', [])
                    print(f"\n🍎 Products found: {len(products)}")
                    
                    # Show product details
                    for i, product in enumerate(products[:3], 1):
                        name = product.get('name', 'N/A')
                        price = product.get('price', 'N/A')
                        features = product.get('features', [])
                        print(f"  {i}. {name}")
                        print(f"     Price: {price}")
                        if features:
                            print(f"     Features: {', '.join(features[:3])}")
                    
                    # Check objective-related information
                    objective_info = data.get('objective_related_information', {})
                    if objective_info:
                        print(f"\n💰 Objective-related info:")
                        pricing_table = objective_info.get('pricing_table', {})
                        models = pricing_table.get('models', [])
                        prices = pricing_table.get('prices', [])
                        
                        print(f"  Models found: {len(models)}")
                        print(f"  Prices found: {len(prices)}")
                        
                        if models:
                            print(f"  Sample models: {models[:3]}")
                        if prices:
                            print(f"  Sample prices: {prices[:3]}")
                        
                        # Show additional details
                        details = objective_info.get('details', [])
                        if details:
                            print(f"  Additional details: {details[:3]}")
                    
                    # Check source URL
                    source_url = data.get('source_url', 'N/A')
                    print(f"\n🌐 Primary source: {source_url}")
                    
                    return True
                else:
                    print("❌ No data extracted")
                    return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the Flask app running?")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def test_direct_apple_url():
    """Test with direct Apple Watch URL"""
    print("\n" + "=" * 50)
    print("Testing Direct Apple Watch URL")
    print("=" * 50)
    
    test_data = {
        "company_name": "https://www.apple.com/watch/",
        "objective": "find apple watch models and pricing",
        "llm": "R1"
    }
    
    print(f"Testing URL: {test_data['company_name']}")
    print(f"Objective: {test_data['objective']}")
    print("-" * 50)
    
    try:
        response = requests.post(
            "http://127.0.0.1:5001/crawl",
            json=test_data,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return False
            else:
                print("✅ Success! Direct URL crawl completed")
                data = result.get('data', {})
                products = data.get('products_and_services', [])
                print(f"🍎 Products found: {len(products)}")
                
                for i, product in enumerate(products[:2], 1):
                    print(f"  {i}. {product.get('name', 'N/A')} - {product.get('price', 'N/A')}")
                
                return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🍎 Apple Watch Pricing Test Suite")
    print("Make sure the Flask app is running on port 5001")
    print()
    
    # Test 1: Search-based approach
    success1 = test_apple_watch_pricing()
    
    # Test 2: Direct URL approach
    success2 = test_direct_apple_url()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"Search-based test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Direct URL test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 or success2:
        print("\n🎉 At least one test passed! Apple Watch extraction is working.")
    else:
        print("\n💥 Both tests failed. Check the crawler configuration.")

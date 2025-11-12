#!/usr/bin/env python3
"""
Test script to verify the enhanced web crawler fixes
"""
import requests
import json
import time

def test_enhanced_crawler():
    """Test the enhanced crawler through the API"""
    print("Testing Enhanced Web Crawler")
    print("=" * 50)
    
    # Test data - Tesla Philippines
    test_data = {
        "company_name": "tesla philippines",
        "objective": "find tesla car models, features, and pricing",
        "llm": "R1"
    }
    
    print(f"Testing URL: {test_data['company_name']}")
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
                
                # Check if we got meaningful data
                data = result.get('data', {})
                if data:
                    print(f"📝 Entity overview: {data.get('entity_overview', 'N/A')[:100]}...")
                    
                    products = data.get('products_and_services', [])
                    print(f"🚗 Products found: {len(products)}")
                    
                    objective_info = data.get('objective_related_information', {})
                    pricing_table = objective_info.get('pricing_table', {})
                    models = pricing_table.get('models', [])
                    prices = pricing_table.get('prices', [])
                    
                    print(f"💰 Pricing models found: {len(models)}")
                    print(f"💵 Prices found: {len(prices)}")
                    
                    if models:
                        print(f"🔍 Sample models: {models[:3]}")
                    if prices:
                        print(f"🔍 Sample prices: {prices[:3]}")
                    
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

if __name__ == "__main__":
    success = test_enhanced_crawler()
    if success:
        print("\n🎉 Enhanced crawler is working correctly!")
    else:
        print("\n💥 Enhanced crawler test failed!")

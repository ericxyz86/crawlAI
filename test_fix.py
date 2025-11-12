#!/usr/bin/env python3

import json

# Test our JSON parsing fix
test_response = '''Based on content:

```json
{
  "entity_overview": "BYD Cars Philippines website",
  "products_and_services": [
    {"name": "BYD eMAX 7", "price": "1,748,000"}
  ]
}
```'''

content = test_response

if '```json' in content:
    json_start = content.find('```json') + 7
    json_end = content.find('```', json_start)
    if json_end != -1:
        content = content[json_start:json_end]

content = content.strip()

try:
    parsed = json.loads(content)
    print('JSON parsing SUCCESS!')
    print('Products found:', len(parsed.get('products_and_services', [])))
    print('First product:', parsed.get('products_and_services', [{}])[0])
except Exception as e:
    print('JSON parsing FAILED:', str(e))
    print('Content preview:', repr(content[:100]))

print('\nTesting the actual crawler now...')

# Test the actual crawler with improvements
from improved_web_crawler import WebCrawler

try:
    crawler = WebCrawler()
    result = crawler.crawl_website('bydcarsphilippines.com', 'find all byd car models and pricing', 'R1')
    
    if result and 'error' not in result:
        print('SUCCESS! Data extracted successfully')
        data = result.get('data', {})
        products = data.get('products_and_services', [])
        print(f'Number of products found: {len(products)}')
        
        if products:
            for i, product in enumerate(products[:3]):
                print(f'Product {i+1}: {product.get("name", "N/A")} - {product.get("price", "N/A")}')
        else:
            print('No products in result, but extraction succeeded')
            print('Entity overview:', data.get('entity_overview', 'N/A'))
    else:
        print('FAILED:', result)
        
except KeyboardInterrupt:
    print('\nTest interrupted')
except Exception as e:
    print('Test error:', str(e))
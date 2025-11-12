#!/usr/bin/env python3

from improved_web_crawler import ContentExtractor, ConfigManager

config = ConfigManager()
extractor = ContentExtractor(config)

print("Testing extraction from BYD pricelist...")
try:
    result = extractor.extract_company_info(
        'bydcarsphilippines.com',
        'https://bydcarsphilippines.com/pricelist',
        'find all byd car models and pricing',
        'R1'
    )
    
    if result:
        print("SUCCESS! Extraction worked!")
        print("Entity overview:", result.get('entity_overview', 'N/A')[:100])
        products = result.get('products_and_services', [])
        print(f"Products found: {len(products)}")
        if products:
            for i, product in enumerate(products[:2]):
                print(f"  {i+1}. {product.get('name', 'N/A')} - {product.get('price', 'N/A')}")
    else:
        print("FAILED: No result returned")
        
except Exception as e:
    print("ERROR:", str(e))
    import traceback
    traceback.print_exc()
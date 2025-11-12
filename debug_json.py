#!/usr/bin/env python3

# Test exactly what's happening with JSON parsing
test_response = '''```json
{
    "entity_overview": "BYD Cars Philippines offers a range of electric and hybrid vehicles with various models and variants.",
    "products_and_services": [
        {
            "name": "BYD eMAX 7",
            "description": "Electric vehicle with superior and standard variants.",
            "price": "Starts at Php 1,748,000 (Superior Captain), Php 1,498,000 (Standard)",
            "features": ["Electric vehicle", "Superior and standard variants"],
            "variants": []
        }
    ]
}
```'''

print("Original content length:", len(test_response))
print("Original first 50 chars:", repr(test_response[:50]))

content = test_response

# Our current logic
if '```json' in content:
    print("Found ```json marker")
    json_start = content.find('```json') + 7
    json_end = content.find('```', json_start)
    print(f"json_start: {json_start}, json_end: {json_end}")
    if json_end != -1:
        content = content[json_start:json_end]
        print("Extracted content length:", len(content))
        print("Extracted first 50 chars:", repr(content[:50]))

content = content.strip()
print("After strip - length:", len(content))
print("After strip - first 50 chars:", repr(content[:50]))

import json
try:
    parsed = json.loads(content)
    print("SUCCESS: JSON parsed successfully!")
    print("Products found:", len(parsed.get('products_and_services', [])))
except Exception as e:
    print("FAILED:", str(e))
    print("Full content to parse:", repr(content))
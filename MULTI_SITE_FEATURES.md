# Multi-Site WebCrawler Enhancement Documentation

## Overview

The WebCrawler has been enhanced with comprehensive multi-site crawling capabilities that intelligently categorize websites and extract data from multiple sources simultaneously. This provides richer, more comprehensive information with built-in comparison features.

## Key Features

### 🎯 Intelligent Site Categorization

The crawler automatically categorizes websites into the following types:

- **Official**: Company websites, brand sites
- **Retailer**: E-commerce sites, authorized dealers
- **Dealer**: Authorized dealerships, service centers
- **Review**: Review sites, comparison platforms
- **Social**: Social media pages
- **News**: News articles, press releases
- **Marketplace**: Classified ads, second-hand markets
- **Other**: Uncategorized relevant sites

### 🔄 Multi-Site Data Extraction

- **Parallel Processing**: Extracts data from multiple sites simultaneously
- **Category Prioritization**: Focuses on most valuable site types
- **Rate Limiting**: Respects server resources with controlled concurrency
- **Error Handling**: Graceful failure handling per site

### 📊 Data Aggregation & Comparison

- **Price Comparison**: Automatic price comparison across retailers
- **Feature Analysis**: Consolidated feature comparison
- **Contact Aggregation**: Contact information from multiple sources
- **Best Source Recommendations**: AI-powered source recommendations

## Enhanced JSON Output Structure

### Multi-Site Mode Output

```json
{
  "urls": ["list of all discovered URLs"],
  "data": {
    "entity_name": "Entity being analyzed",
    "objective": "User's search objective",
    "sites_analyzed": {
      "official": [
        {
          "domain": "company.com",
          "category": "official",
          "extraction_time": 15.2,
          "data": { /* extracted data */ }
        }
      ],
      "retailer": [ /* retailer sites data */ ],
      "dealer": [ /* dealer sites data */ ]
    },
    "price_comparison": {
      "Product Name": {
        "prices_by_source": [
          {
            "price": "$399",
            "source": "apple.com",
            "category": "official"
          }
        ]
      }
    },
    "feature_comparison": {
      "common_features": ["feature1", "feature2"],
      "feature_frequency": {}
    },
    "contact_aggregation": {
      "email": [{"value": "contact@company.com", "source_domain": "company.com"}],
      "phone": [{"value": "+1234567890", "source_domain": "company.com"}]
    },
    "best_sources": {
      "pricing": "retailer.com",
      "official_info": "company.com",
      "reviews": "reviewsite.com"
    },
    "recommendations": [
      "Multi-site analysis completed",
      "Compare pricing across retailers"
    ],
    "consolidated_overview": "Summary of analysis"
  },
  "multi_site_data": { /* Raw extractions by category */ },
  "site_categories": { /* URL categorization */ },
  "extraction_mode": "multi_site",
  "metadata": {
    "crawl_time": "2025-06-12T21:30:00",
    "execution_time_seconds": 145.67,
    "sites_analyzed": 8,
    "categories_found": ["official", "retailer", "dealer"]
  }
}
```

### Single-Site Mode (Legacy Compatible)

For backward compatibility, when only one site/category is found:

```json
{
  "urls": ["single URL or limited URLs"],
  "data": { /* traditional single-site extraction */ },
  "site_categories": { /* categorization still included */ },
  "extraction_mode": "single_site",
  "metadata": { /* standard metadata */ }
}
```

## API Usage

### Basic Multi-Site Request

```bash
curl -X POST http://localhost:5001/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "Apple Watch Philippines",
    "objective": "find pricing and models",
    "llm": "R1"
  }'
```

### Legacy Compatible Request

```bash
curl -X POST http://localhost:5001/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "Apple Watch Philippines",
    "objective": "find pricing and models",
    "llm": "R1",
    "legacy_mode": true
  }'
```

## Configuration

### Site Category Limits

The crawler limits extraction per category to optimize performance:

- **Official**: Up to 3 sites
- **Retailer**: Up to 5 sites (important for pricing)
- **Dealer**: Up to 3 sites
- **Review**: Up to 2 sites
- **Other**: Up to 2 sites
- **News**: Up to 1 site
- **Marketplace**: Up to 2 sites

### Performance Settings

- **Concurrent Workers**: 3 per category
- **Timeout**: 120 seconds per site
- **Max Content Length**: 10,000 characters per extraction

## Use Cases

### 1. E-commerce Price Comparison

**Input**: "iPhone 15 Philippines pricing"

**Output**: 
- Official Apple pricing
- Retailer pricing from multiple stores
- Price comparison table
- Best deals recommendations

### 2. Product Research

**Input**: "Tesla Model 3 features and specifications"

**Output**:
- Official specifications from Tesla
- Dealer information and availability
- Review site comparisons
- Feature analysis across sources

### 3. Company Information Gathering

**Input**: "Microsoft Azure services"

**Output**:
- Official service descriptions
- Pricing from multiple regions
- Third-party reviews and comparisons
- Contact information aggregation

## Benefits

### 🚀 Enhanced Data Quality

- **Comprehensive Coverage**: Information from multiple authoritative sources
- **Cross-Validation**: Compare information across sources for accuracy
- **Rich Context**: Official specs + retailer pricing + user reviews

### 💰 Price Intelligence

- **Real-time Comparison**: Compare prices across multiple retailers
- **Best Deal Detection**: Automatic identification of best pricing
- **Regional Variations**: Local pricing vs international pricing

### 🎯 Intelligent Prioritization

- **Source Authority**: Prioritizes official sources for specifications
- **Pricing Focus**: Emphasizes retailers for current pricing
- **User Feedback**: Includes review sites for user perspectives

### 🔧 Developer Benefits

- **Backward Compatible**: Existing integrations continue to work
- **Enhanced APIs**: New endpoints provide richer data
- **Flexible Output**: Choose between detailed multi-site or simple single-site

## Implementation Details

### Site Detection Patterns

The crawler uses intelligent pattern matching to categorize sites:

```python
# Official site detection
- Known brand domains (apple.com, microsoft.com)
- Entity keywords in domain names
- Official company indicators

# Retailer detection  
- Known e-commerce platforms
- Shopping-related URL patterns
- Authorized dealer indicators

# Review site detection
- Review platform domains
- Comparison-related keywords
- Rating and review URL patterns
```

### Data Aggregation Logic

1. **Category Processing**: Process sites by category priority
2. **Parallel Extraction**: Extract from multiple sites simultaneously
3. **Data Consolidation**: Merge and compare extracted information
4. **Quality Assessment**: Evaluate source reliability
5. **Recommendation Generation**: Suggest best sources for different needs

## Testing

Run the test suite to verify functionality:

```bash
python test_multi_site.py
```

This tests:
- Site categorization accuracy
- Legacy compatibility
- JSON structure validation
- Error handling

## Migration Guide

### For Existing Applications

**No Changes Required**: Existing applications will continue to work without modification. The enhanced crawler maintains full backward compatibility.

**Optional Enhancements**: To leverage new features:

1. **Check `extraction_mode`** in response to detect multi-site results
2. **Access `site_categories`** for URL categorization information
3. **Use `price_comparison`** for comparative pricing data
4. **Leverage `best_sources`** for source recommendations

### For New Applications

**Recommended Approach**: Design applications to handle both modes:

```python
def process_crawl_result(result):
    if result.get('extraction_mode') == 'multi_site':
        # Handle rich multi-site data
        price_comparison = result['data']['price_comparison']
        best_sources = result['data']['best_sources']
        recommendations = result['data']['recommendations']
    else:
        # Handle traditional single-site data
        traditional_data = result['data']
```

## Future Enhancements

- **Real-time Price Monitoring**: Track price changes across retailers
- **Stock Availability**: Monitor product availability
- **Sentiment Analysis**: Analyze review sentiment across sources
- **Geographic Optimization**: Optimize for regional preferences
- **Custom Site Categories**: Allow user-defined site categories
#!/usr/bin/env python3
"""
Comprehensive test suite for multi-site web crawler functionality.
Tests the enhanced multi-site search, price comparison, and data aggregation features.
"""

import unittest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add the parent directory to the path to import the crawler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_web_crawler import WebCrawler, SearchEngine, ConfigManager

class TestMultiSiteSearchEngine(unittest.TestCase):
    """Test the enhanced SearchEngine with multi-query capabilities."""
    
    def setUp(self):
        self.config = Mock(spec=ConfigManager)
        self.config.serp_api_key = "test_key"
        self.config.timeout_medium = 30
        self.search_engine = SearchEngine(self.config)
    
    def test_retailer_database_initialization(self):
        """Test that retailer databases are properly initialized."""
        self.assertIn('philippines', self.search_engine.retailer_databases)
        self.assertIn('us', self.search_engine.retailer_databases)
        self.assertIn('global', self.search_engine.retailer_databases)
        
        # Test specific retailers
        ph_electronics = self.search_engine.retailer_databases['philippines']['electronics']
        self.assertIn('powermaccenter.com', ph_electronics)
        self.assertIn('beyondthebox.ph', ph_electronics)
        self.assertIn('istore.ph', ph_electronics)
    
    def test_detect_region_and_category(self):
        """Test region and category detection from queries."""
        # Test Philippines detection
        region, category = self.search_engine.detect_region_and_category(
            "Apple Watch Philippines", "find pricing"
        )
        self.assertEqual(region, 'philippines')
        self.assertEqual(category, 'electronics')
        
        # Test US detection
        region, category = self.search_engine.detect_region_and_category(
            "Toyota Camry USA", "find dealers"
        )
        self.assertEqual(region, 'us')
        self.assertEqual(category, 'automotive')
        
        # Test default values
        region, category = self.search_engine.detect_region_and_category(
            "Some Company", "general info"
        )
        self.assertEqual(region, 'global')
        self.assertEqual(category, 'general')
    
    def test_generate_retailer_queries(self):
        """Test generation of retailer-specific queries."""
        queries = self.search_engine.generate_retailer_queries(
            "Apple Watch", "find pricing", max_queries=3
        )
        
        self.assertGreaterEqual(len(queries), 1)
        self.assertEqual(queries[0], "Apple Watch")  # Original query should be first
        
        # Should contain retailer-specific queries
        retailer_queries = [q for q in queries if 'Apple Watch' in q and q != 'Apple Watch']
        self.assertGreater(len(retailer_queries), 0)
    
    @patch('improved_web_crawler.SearchEngine.search_google')
    def test_search_comprehensive(self, mock_search_google):
        """Test comprehensive multi-query search functionality."""
        # Mock search results
        mock_search_google.return_value = [
            {'link': 'https://apple.com/watch', 'title': 'Apple Watch', 'position': 1},
            {'link': 'https://powermac.com.ph/apple-watch', 'title': 'Apple Watch - PowerMac', 'position': 2}
        ]
        
        results = self.search_engine.search_comprehensive("Apple Watch", "find pricing")
        
        # Should have called search_google multiple times
        self.assertGreater(mock_search_google.call_count, 1)
        
        # Should return unique results
        urls = [r.get('link') for r in results]
        self.assertEqual(len(urls), len(set(urls)))  # No duplicates
        
        # Should have metadata about which query found each result
        for result in results:
            self.assertIn('found_by_query', result)


class TestMultiSitePriceComparison(unittest.TestCase):
    """Test the advanced price comparison engine."""
    
    def setUp(self):
        self.config = Mock(spec=ConfigManager)
        self.crawler = WebCrawler()
    
    def test_parse_price_string(self):
        """Test price string parsing functionality."""
        # Test USD prices
        result = self.crawler._parse_price_string("$399.99")
        self.assertEqual(result, (399.99, 'USD'))
        
        # Test PHP prices
        result = self.crawler._parse_price_string("₱28,990")
        self.assertEqual(result, (28990.0, 'PHP'))
        
        # Test invalid prices
        result = self.crawler._parse_price_string("Not specified")
        self.assertIsNone(result)
        
        result = self.crawler._parse_price_string("Contact for pricing")
        self.assertIsNone(result)
    
    def test_normalize_to_usd(self):
        """Test currency normalization to USD."""
        # Test PHP to USD conversion
        usd_price = self.crawler._normalize_to_usd(28990, 'PHP')
        self.assertAlmostEqual(usd_price, 521.82, places=2)
        
        # Test USD (no conversion)
        usd_price = self.crawler._normalize_to_usd(399.99, 'USD')
        self.assertEqual(usd_price, 399.99)
    
    def test_create_advanced_price_comparison(self):
        """Test advanced price comparison creation."""
        # Mock price data from multiple sources
        all_prices = [
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
        
        price_comparison = self.crawler._create_advanced_price_comparison(all_prices)
        
        # Should have product entry
        self.assertIn('Apple Watch Series 10', price_comparison)
        
        product_data = price_comparison['Apple Watch Series 10']
        
        # Should have price analysis
        self.assertIn('price_analysis', product_data)
        analysis = product_data['price_analysis']
        
        # Should have calculated price statistics
        self.assertIsNotNone(analysis['lowest_price'])
        self.assertIsNotNone(analysis['highest_price'])
        self.assertIsNotNone(analysis['average_price'])
        self.assertEqual(analysis['retailer_count'], 2)
        
        # Should have recommendations
        self.assertIn('price_recommendations', product_data)
        self.assertGreater(len(product_data['price_recommendations']), 0)


class TestMultiSiteTriggering(unittest.TestCase):
    """Test the multi-site triggering logic."""
    
    def setUp(self):
        self.crawler = WebCrawler()
    
    @patch('improved_web_crawler.WebCrawler._extract_from_multiple_sites')
    @patch('improved_web_crawler.WebCrawler._categorize_sites')
    @patch('improved_web_crawler.SearchEngine.search_comprehensive')
    @patch('improved_web_crawler.URLProcessor.select_urls_with_ai')
    @patch('improved_web_crawler.URLProcessor.validate_urls')
    def test_company_name_triggers_multisite(self, mock_validate, mock_select, mock_search, mock_categorize, mock_extract):
        """Test that company names always trigger multi-site extraction."""
        # Mock search results
        mock_search.return_value = [
            {'link': 'https://apple.com/watch', 'title': 'Apple Watch'},
            {'link': 'https://powermac.com.ph/apple-watch', 'title': 'Apple Watch - PowerMac'}
        ]
        
        # Mock URL selection
        mock_select.return_value = ['https://apple.com/watch', 'https://powermac.com.ph/apple-watch']
        
        # Mock URL validation
        mock_validate.return_value = ['https://apple.com/watch', 'https://powermac.com.ph/apple-watch']
        
        # Mock site categorization
        mock_categorize.return_value = {
            'official': ['https://apple.com/watch'],
            'retailer': ['https://powermac.com.ph/apple-watch']
        }
        
        # Mock multi-site extraction
        mock_extract.return_value = {
            'extractions_by_category': {},
            'extraction_summary': {'successful_extractions': 2, 'failed_extractions': 0}
        }
        
        # Test company name input (should trigger multi-site)
        with patch('improved_web_crawler.WebCrawler._aggregate_multi_site_data') as mock_aggregate:
            mock_aggregate.return_value = {'test': 'data'}
            
            result = self.crawler.crawl_website("Apple Watch", "find pricing")
            
            # Should have called multi-site extraction
            mock_extract.assert_called_once()
            mock_aggregate.assert_called_once()


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility features."""
    
    def setUp(self):
        self.crawler = WebCrawler()
    
    def test_legacy_compatible_result(self):
        """Test conversion of multi-site results to legacy format."""
        # Mock multi-site result
        multi_site_result = {
            'extraction_mode': 'multi_site',
            'urls': ['https://apple.com', 'https://powermac.com'],
            'data': {
                'sites_analyzed': {
                    'official': [{'domain': 'apple.com', 'data': {'entity_overview': 'Apple Watch'}}],
                    'retailer': [{'domain': 'powermac.com', 'data': {'entity_overview': 'PowerMac Apple Watch'}}]
                }
            },
            'metadata': {'test': 'metadata'}
        }
        
        legacy_result = self.crawler._create_legacy_compatible_result(multi_site_result)
        
        # Should have legacy structure
        self.assertIn('urls', legacy_result)
        self.assertIn('data', legacy_result)
        self.assertIn('metadata', legacy_result)
        
        # Should prioritize official source
        self.assertIn('entity_overview', legacy_result['data'])
        self.assertEqual(legacy_result['data']['entity_overview'], 'Apple Watch')


class TestFlaskAPIIntegration(unittest.TestCase):
    """Test Flask API integration with multi-site features."""
    
    def setUp(self):
        os.environ['TESTING'] = 'True'
        from app import app
        self.app = app.test_client()
        self.app.testing = True
    
    @patch('app.WebCrawler.crawl_website')
    def test_multisite_api_response(self, mock_crawl):
        """Test API response format for multi-site extraction."""
        # Mock multi-site crawler result
        mock_crawl.return_value = {
            'extraction_mode': 'multi_site',
            'urls': ['https://apple.com', 'https://powermac.com'],
            'data': {
                'price_comparison': {
                    'Apple Watch': {
                        'price_analysis': {
                            'lowest_price': 399.99,
                            'highest_price': 429.99,
                            'retailer_count': 2
                        },
                        'price_recommendations': ['Best price at apple.com']
                    }
                }
            },
            'site_categories': {
                'official': ['https://apple.com'],
                'retailer': ['https://powermac.com']
            },
            'metadata': {
                'sites_analyzed': 2,
                'categories_found': ['official', 'retailer']
            }
        }
        
        response = self.app.post('/crawl', 
                               data=json.dumps({
                                   'company_name': 'Apple Watch',
                                   'objective': 'find pricing'
                               }),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Should have multi-site specific fields
        self.assertEqual(data['extraction_mode'], 'multi_site')
        self.assertIn('price_summary', data)
        self.assertIn('site_categories', data)
        self.assertIn('multi_site_summary', data)
    
    @patch('app.WebCrawler.crawl_website')
    def test_legacy_mode_api(self, mock_crawl):
        """Test API legacy mode conversion."""
        # Mock multi-site result
        mock_result = {
            'extraction_mode': 'multi_site',
            'data': {'test': 'multi_site_data'}
        }
        
        # Mock legacy conversion
        with patch('app.WebCrawler._create_legacy_compatible_result') as mock_legacy:
            mock_crawl.return_value = mock_result
            mock_legacy.return_value = {
                'extraction_mode': 'single_site',
                'data': {'test': 'legacy_data'}
            }
            
            response = self.app.post('/crawl', 
                                   data=json.dumps({
                                       'company_name': 'Apple Watch',
                                       'legacy_mode': True
                                   }),
                                   content_type='application/json')
            
            self.assertEqual(response.status_code, 200)
            
            # Should have called legacy conversion
            mock_legacy.assert_called_once_with(mock_result)


def run_tests():
    """Run all test suites."""
    print("🧪 Running Comprehensive Multi-Site Crawler Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestMultiSiteSearchEngine,
        TestMultiSitePriceComparison,
        TestMultiSiteTriggering,
        TestBackwardCompatibility,
        TestFlaskAPIIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed successfully!")
        print(f"Ran {result.testsRun} tests with 0 failures")
    else:
        print("❌ Some tests failed:")
        print(f"Ran {result.testsRun} tests with {len(result.failures)} failures and {len(result.errors)} errors")
        
        for failure in result.failures:
            print(f"\nFAILURE: {failure[0]}")
            print(failure[1])
        
        for error in result.errors:
            print(f"\nERROR: {error[0]}")
            print(error[1])
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run offline tests only
    success = run_tests()
    sys.exit(0 if success else 1)
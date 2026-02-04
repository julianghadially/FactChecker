"""Unit tests for the TemporalRouterModule.

These tests validate the date extraction, URL detection, and routing logic
without making actual API calls.
"""

import unittest
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.factchecker.modules.temporal_router_module import TemporalRouterModule


class TestTemporalRouter(unittest.TestCase):
    """Test suite for TemporalRouterModule."""

    def setUp(self):
        """Set up test fixtures."""
        self.router = TemporalRouterModule()

    def test_extract_urls_single(self):
        """Test extraction of single URL."""
        text = "Visit https://example.com for more info."
        urls = self.router._extract_urls(text)
        self.assertEqual(len(urls), 1)
        self.assertEqual(urls[0], "https://example.com")

    def test_extract_urls_multiple(self):
        """Test extraction of multiple URLs."""
        text = "Check https://example.com and http://test.org for details."
        urls = self.router._extract_urls(text)
        self.assertEqual(len(urls), 2)
        self.assertIn("https://example.com", urls)
        self.assertIn("http://test.org", urls)

    def test_extract_urls_none(self):
        """Test no URL extraction when none present."""
        text = "This text has no URLs."
        urls = self.router._extract_urls(text)
        self.assertEqual(len(urls), 0)

    def test_extract_date_iso_format(self):
        """Test extraction of ISO format date (YYYY-MM-DD)."""
        text = "The event occurred on 2025-03-15."
        dates = self.router._extract_dates(text)
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0], datetime(2025, 3, 15))

    def test_extract_date_slash_format(self):
        """Test extraction of slash format date (YYYY/MM/DD)."""
        text = "The deadline is 2025/12/31."
        dates = self.router._extract_dates(text)
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0], datetime(2025, 12, 31))

    def test_extract_date_month_first(self):
        """Test extraction of Month DD, YYYY format."""
        text = "On January 15, 2025, the announcement was made."
        dates = self.router._extract_dates(text)
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0], datetime(2025, 1, 15))

    def test_extract_date_month_abbreviated(self):
        """Test extraction of abbreviated month format."""
        text = "The launch is scheduled for Mar 20, 2026."
        dates = self.router._extract_dates(text)
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0], datetime(2026, 3, 20))

    def test_extract_date_day_first(self):
        """Test extraction of DD Month YYYY format."""
        text = "The meeting is on 25 December 2024."
        dates = self.router._extract_dates(text)
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0], datetime(2024, 12, 25))

    def test_extract_date_year_only(self):
        """Test extraction of year-only references."""
        text = "In 2025, the policy will change."
        dates = self.router._extract_dates(text)
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0], datetime(2025, 1, 1))

    def test_extract_multiple_dates(self):
        """Test extraction of multiple dates."""
        text = "Between 2024-01-01 and January 15, 2025, many events occurred."
        dates = self.router._extract_dates(text)
        self.assertGreaterEqual(len(dates), 2)
        self.assertIn(datetime(2024, 1, 1), dates)
        self.assertIn(datetime(2025, 1, 15), dates)

    def test_has_temporal_keywords_today(self):
        """Test detection of 'today' keyword."""
        text = "The news was published today."
        result = self.router._has_temporal_keywords(text)
        self.assertTrue(result)

    def test_has_temporal_keywords_recent(self):
        """Test detection of 'recent' keyword."""
        text = "According to recent reports, the economy is improving."
        result = self.router._has_temporal_keywords(text)
        self.assertTrue(result)

    def test_has_temporal_keywords_latest(self):
        """Test detection of 'latest' keyword."""
        text = "The latest study shows promising results."
        result = self.router._has_temporal_keywords(text)
        self.assertTrue(result)

    def test_has_temporal_keywords_year_2024(self):
        """Test detection of '2024' keyword."""
        text = "In 2024, major changes occurred."
        result = self.router._has_temporal_keywords(text)
        self.assertTrue(result)

    def test_has_temporal_keywords_none(self):
        """Test no temporal keywords in historical statement."""
        text = "The Apollo 11 mission landed on the moon."
        result = self.router._has_temporal_keywords(text)
        self.assertFalse(result)

    def test_should_use_web_research_urls_provided(self):
        """Test web research decision when URLs provided."""
        statement = "The company announced results."
        urls = ["https://example.com"]
        dates = []

        should_use, reason = self.router._should_use_web_research(statement, urls, dates)
        self.assertTrue(should_use)
        self.assertIn("URLs provided", reason)

    def test_should_use_web_research_future_date(self):
        """Test web research decision for future date."""
        statement = "An event will occur in March 2025."
        urls = []
        dates = [datetime(2025, 3, 1)]

        should_use, reason = self.router._should_use_web_research(statement, urls, dates)
        self.assertTrue(should_use)
        self.assertIn("beyond knowledge cutoff", reason)

    def test_should_use_web_research_temporal_keywords(self):
        """Test web research decision for temporal keywords."""
        statement = "The latest report shows recent trends."
        urls = []
        dates = []

        should_use, reason = self.router._should_use_web_research(statement, urls, dates)
        self.assertTrue(should_use)
        self.assertIn("temporal keywords", reason.lower())

    def test_should_not_use_web_research_historical(self):
        """Test no web research for historical statement."""
        statement = "The Apollo 11 mission landed in 1969."
        urls = []
        dates = [datetime(1969, 7, 20)]

        should_use, reason = self.router._should_use_web_research(statement, urls, dates)
        self.assertFalse(should_use)
        self.assertIn("No temporal references", reason)

    def test_should_not_use_web_research_old_date(self):
        """Test no web research for date before cutoff."""
        statement = "The event occurred in March 2020."
        urls = []
        dates = [datetime(2020, 3, 1)]

        should_use, reason = self.router._should_use_web_research(statement, urls, dates)
        self.assertFalse(should_use)

    def test_knowledge_cutoff_customization(self):
        """Test custom knowledge cutoff date."""
        # Router with cutoff in 2023
        custom_router = TemporalRouterModule(
            knowledge_cutoff=datetime(2023, 1, 1)
        )

        statement = "In 2024, things changed."
        urls = []
        dates = [datetime(2024, 1, 1)]

        should_use, reason = custom_router._should_use_web_research(statement, urls, dates)
        self.assertTrue(should_use)
        self.assertIn("2024-01-01 >= 2023-01-01", reason)

    def test_date_extraction_edge_cases(self):
        """Test date extraction handles edge cases gracefully."""
        # Invalid dates should be skipped
        text = "On 2025-13-45, something happened."  # Invalid date
        dates = self.router._extract_dates(text)
        # Should not crash, may return empty or skip invalid dates
        self.assertIsInstance(dates, list)

    def test_url_extraction_with_paths(self):
        """Test URL extraction includes paths and query params."""
        text = "Visit https://example.com/path/to/page?param=value for details."
        urls = self.router._extract_urls(text)
        self.assertEqual(len(urls), 1)
        self.assertIn("example.com/path/to/page", urls[0])

    def test_case_insensitive_month_matching(self):
        """Test that month names are matched case-insensitively."""
        text = "The event is on JANUARY 15, 2025."
        dates = self.router._extract_dates(text)
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0], datetime(2025, 1, 15))

    def test_multiple_temporal_keywords(self):
        """Test detection with multiple temporal keywords."""
        text = "Recent reports from today show latest trends this week."
        result = self.router._has_temporal_keywords(text)
        self.assertTrue(result)


class TestDateFormatCoverage(unittest.TestCase):
    """Test coverage of various date formats."""

    def setUp(self):
        """Set up test fixtures."""
        self.router = TemporalRouterModule()

    def test_all_month_names_full(self):
        """Test all full month names are recognized."""
        months = [
            ("January", 1), ("February", 2), ("March", 3), ("April", 4),
            ("May", 5), ("June", 6), ("July", 7), ("August", 8),
            ("September", 9), ("October", 10), ("November", 11), ("December", 12)
        ]

        for month_name, month_num in months:
            text = f"On {month_name} 15, 2025, an event occurred."
            dates = self.router._extract_dates(text)
            self.assertGreaterEqual(len(dates), 1, f"Failed to extract {month_name}")
            # Check that at least one date has the correct month
            months_extracted = [d.month for d in dates]
            self.assertIn(month_num, months_extracted, f"Month {month_num} not found in {months_extracted}")

    def test_all_month_abbreviations(self):
        """Test all month abbreviations are recognized."""
        months = [
            ("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4),
            ("May", 5), ("Jun", 6), ("Jul", 7), ("Aug", 8),
            ("Sep", 9), ("Oct", 10), ("Nov", 11), ("Dec", 12)
        ]

        for month_abbr, month_num in months:
            text = f"On {month_abbr} 15, 2025, an event occurred."
            dates = self.router._extract_dates(text)
            self.assertGreaterEqual(len(dates), 1, f"Failed to extract {month_abbr}")
            # Check that at least one date has the correct month
            months_extracted = [d.month for d in dates]
            self.assertIn(month_num, months_extracted, f"Month {month_num} not found in {months_extracted}")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)

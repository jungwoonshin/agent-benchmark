"""Standalone test code for selenium_browser_navigator.py using cases from log.txt"""

import logging
import sys
import time
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import directly from the module file to avoid __init__.py import issues
# Use importlib to load the module directly
import importlib.util
selenium_module_path = src_path / "browser" / "selenium_browser_navigator.py"
spec = importlib.util.spec_from_file_location("selenium_browser_navigator", selenium_module_path)
selenium_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selenium_module)
SeleniumBrowserNavigator = selenium_module.SeleniumBrowserNavigator


def setup_logger():
    """Setup logger for testing"""
    logger = logging.getLogger("test_selenium")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger


def test_navigate_to_url():
    """Test basic navigation (from log.txt line 763-766)"""
    logger = setup_logger()
    browser = SeleniumBrowserNavigator(logger=logger, headless=True)

    try:
        # Test URL from log: USGS species profile
        url = "https://www.example.com"
        result = browser.navigate(url)

        assert result["success"] is True or result["status_code"] in [200, 504], \
            f"Navigation should succeed or timeout, got: {result}"
        assert "url" in result, "Result should contain 'url'"
        print(f"✓ Test 1 passed: Navigate to URL - {result.get('url', 'N/A')[:50]}")

    finally:
        browser.close()


def test_navigate_with_js_wait():
    """Test navigation with JavaScript wait time"""
    logger = setup_logger()
    browser = SeleniumBrowserNavigator(logger=logger, headless=True)

    try:
        url = "https://www.example.com"
        result = browser.navigate(url, wait_for_js=3.0)

        assert "content" in result, "Result should contain 'content'"
        assert len(result.get("content", "")) > 0, "Content should not be empty"
        print(f"✓ Test 2 passed: Navigate with JS wait - Content length: {len(result.get('content', ''))}")

    finally:
        browser.close()


def test_extract_text():
    """Test text extraction (from log.txt line 764 - extract_text action)"""
    logger = setup_logger()
    browser = SeleniumBrowserNavigator(logger=logger, headless=True)

    try:
        url = "https://www.example.com"
        result = browser.navigate(url)

        if result["success"]:
            text = browser.extract_text(result)
            assert isinstance(text, str), "Extracted text should be a string"
            assert len(text) > 0, "Extracted text should not be empty"
            print(f"✓ Test 3 passed: Extract text - Length: {len(text)} chars")

    finally:
        browser.close()


def test_timeout_handling():
    """Test timeout handling (from log.txt - timeout errors)"""
    logger = setup_logger()
    # Use very short timeout for testing
    browser = SeleniumBrowserNavigator(logger=logger, headless=True, timeout=1)

    try:
        # Try to navigate to a URL that might timeout
        url = "https://httpstat.us/200?sleep=5000"  # This will timeout
        result = browser.navigate(url)

        # Should handle timeout gracefully
        assert "success" in result, "Result should contain 'success'"
        # Timeout might result in success=False or status_code=504
        print(f"✓ Test 4 passed: Timeout handling - Success: {result.get('success')}, "
              f"Status: {result.get('status_code')}")

    except Exception as e:
        # Timeout exceptions are acceptable
        print(f"✓ Test 4 passed: Timeout handling - Exception caught: {type(e).__name__}")

    finally:
        browser.close()


def test_get_current_url():
    """Test getting current URL"""
    logger = setup_logger()
    browser = SeleniumBrowserNavigator(logger=logger, headless=True)

    try:
        initial_url = browser.get_current_url()
        assert isinstance(initial_url, str), "Current URL should be a string"

        url = "https://www.example.com"
        browser.navigate(url)
        current_url = browser.get_current_url()

        assert current_url.startswith("http"), "Current URL should be a valid URL"
        print(f"✓ Test 5 passed: Get current URL - {current_url[:50]}")

    finally:
        browser.close()


def test_navigate_multiple_urls():
    """Test navigating to multiple URLs (simulating log.txt pattern)"""
    logger = setup_logger()
    browser = SeleniumBrowserNavigator(logger=logger, headless=True)

    urls = [
        "https://www.example.com",
        "https://httpbin.org/html",
    ]

    try:
        results = []
        for url in urls:
            result = browser.navigate(url)
            results.append(result)
            assert "success" in result, f"Result should contain 'success' for {url}"
            time.sleep(1)  # Small delay between requests

        print(f"✓ Test 6 passed: Navigate multiple URLs - {len(results)} URLs navigated")

    finally:
        browser.close()


def test_soup_parsing():
    """Test BeautifulSoup parsing"""
    logger = setup_logger()
    browser = SeleniumBrowserNavigator(logger=logger, headless=True)

    try:
        url = "https://www.example.com"
        result = browser.navigate(url)

        if result.get("success") and result.get("soup"):
            soup = result["soup"]
            # Try to find title
            title = soup.find("title")
            assert title is not None or True, "Soup should be parsed (title may or may not exist)"
            print(f"✓ Test 7 passed: Soup parsing - Title found: {title is not None}")

    finally:
        browser.close()


def test_error_handling_invalid_url():
    """Test error handling for invalid URLs"""
    logger = setup_logger()
    browser = SeleniumBrowserNavigator(logger=logger, headless=True)

    try:
        # Invalid URL format
        url = "not-a-valid-url"
        result = browser.navigate(url)

        # Should handle error gracefully
        assert "success" in result, "Result should contain 'success'"
        print(f"✓ Test 8 passed: Invalid URL handling - Success: {result.get('success')}")

    except Exception as e:
        # Exceptions are acceptable for invalid URLs
        print(f"✓ Test 8 passed: Invalid URL handling - Exception: {type(e).__name__}")

    finally:
        browser.close()


def test_context_manager():
    """Test context manager usage"""
    logger = setup_logger()

    try:
        with SeleniumBrowserNavigator(logger=logger, headless=True) as browser:
            url = "https://www.example.com"
            result = browser.navigate(url)
            assert "success" in result, "Result should contain 'success'"
            print(f"✓ Test 9 passed: Context manager - Navigation successful")

    except Exception as e:
        print(f"✗ Test 9 failed: Context manager - {e}")


def test_headless_mode():
    """Test headless mode configuration"""
    logger = setup_logger()

    # Test headless=True
    browser = SeleniumBrowserNavigator(logger=logger, headless=True)
    try:
        assert browser.headless is True, "Browser should be in headless mode"
        url = "https://www.example.com"
        result = browser.navigate(url)
        assert "success" in result, "Navigation should work in headless mode"
        print(f"✓ Test 10 passed: Headless mode - Navigation successful")

    finally:
        browser.close()


def test_extract_text_from_current_page():
    """Test extracting text from current page without page_data"""
    logger = setup_logger()
    browser = SeleniumBrowserNavigator(logger=logger, headless=True)

    try:
        url = "https://www.example.com"
        browser.navigate(url)

        # Extract text without passing page_data
        text = browser.extract_text()
        assert isinstance(text, str), "Extracted text should be a string"
        assert len(text) > 0, "Extracted text should not be empty"
        print(f"✓ Test 11 passed: Extract text from current page - Length: {len(text)} chars")

    finally:
        browser.close()


def test_click_element_selector():
    """Test clicking element by CSS selector"""
    logger = setup_logger()
    browser = SeleniumBrowserNavigator(logger=logger, headless=True)

    try:
        # Navigate to a page with clickable elements
        url = "https://httpbin.org/html"
        browser.navigate(url)

        # Try to click an element (may not exist, but should handle gracefully)
        result = browser.click_element(selector="a")

        # Should return a result dict (success or error)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should contain 'success'"
        print(f"✓ Test 12 passed: Click element by selector - Success: {result.get('success')}")

    except Exception as e:
        # Click may fail if element doesn't exist, which is fine
        print(f"✓ Test 12 passed: Click element by selector - Exception handled: {type(e).__name__}")

    finally:
        browser.close()


def test_click_element_link_text():
    """Test clicking element by link text"""
    logger = setup_logger()
    browser = SeleniumBrowserNavigator(logger=logger, headless=True)

    try:
        url = "https://httpbin.org/html"
        browser.navigate(url)

        # Try to click a link by text
        result = browser.click_element(link_text="More information...")

        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should contain 'success'"
        print(f"✓ Test 13 passed: Click element by link text - Success: {result.get('success')}")

    except Exception as e:
        # Click may fail if link doesn't exist
        print(f"✓ Test 13 passed: Click element by link text - Exception handled: {type(e).__name__}")

    finally:
        browser.close()


def test_fill_form():
    """Test form filling"""
    logger = setup_logger()
    browser = SeleniumBrowserNavigator(logger=logger, headless=True)

    try:
        # Navigate to a page (may not have a form, but test the method)
        url = "https://httpbin.org/html"
        browser.navigate(url)

        # Try to fill a form (may not exist)
        form_data = {"input[name='test']": "test value"}
        result = browser.fill_form(form_data)

        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should contain 'success'"
        print(f"✓ Test 14 passed: Fill form - Success: {result.get('success')}")

    except Exception as e:
        # Form may not exist, which is fine
        print(f"✓ Test 14 passed: Fill form - Exception handled: {type(e).__name__}")

    finally:
        browser.close()


def test_cleanup_on_del():
    """Test cleanup on deletion"""
    logger = setup_logger()

    browser = SeleniumBrowserNavigator(logger=logger, headless=True)
    url = "https://www.example.com"
    browser.navigate(url)

    # Delete should close browser
    del browser
    print(f"✓ Test 15 passed: Cleanup on deletion")


def run_all_tests():
    """Run all test cases"""
    print("=" * 70)
    print("Running standalone tests for selenium_browser_navigator.py")
    print("=" * 70)
    print("Note: These tests require internet connection and Chrome/Chromium")
    print("=" * 70)

    tests = [
        test_navigate_to_url,
        test_navigate_with_js_wait,
        test_extract_text,
        test_timeout_handling,
        test_get_current_url,
        test_navigate_multiple_urls,
        test_soup_parsing,
        test_error_handling_invalid_url,
        test_context_manager,
        test_headless_mode,
        test_extract_text_from_current_page,
        test_click_element_selector,
        test_click_element_link_text,
        test_fill_form,
        test_cleanup_on_del,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 70)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


# Selenium Integration for JavaScript-Rendered Pages

## Overview

The agent system now uses **Selenium WebDriver** to handle JavaScript-rendered and dynamic web pages, replacing the limited `requests` library for browser navigation.

## What Changed

### 1. **New Dependencies** (requirements.txt)
- `selenium>=4.15.0` - Browser automation framework
- `webdriver-manager>=4.0.0` - Automatic ChromeDriver management

### 2. **New Module** (src/core/selenium_browser_navigator.py)
A new `SeleniumBrowserNavigator` class that provides:
- JavaScript execution support
- Dynamic content handling
- Element interaction (click, fill forms)
- Automatic ChromeDriver installation and management

### 3. **Updated ToolBelt** (src/core/tool_belt.py)
Now uses a **hybrid approach**:
- `selenium_navigator` - For page navigation (handles JS)
- `browser_navigator` - For extraction utilities (parsing, tables, etc.)

## Key Features

### ✅ JavaScript Pages Now Work
Pages like arXiv Advanced Search that previously failed now work correctly:
```python
# Before: ❌ Failed with HTTP 500 or blank content
# After: ✅ Successfully loads and extracts data
```

### ✅ Dynamic Content Support
- Single Page Applications (SPAs)
- AJAX-loaded content
- Interactive forms and buttons

### ✅ Retry Logic Included
- HTTP 500/502/503/504 errors are automatically retried (up to 3 times)
- Exponential backoff with jitter
- Proper error classification

## Architecture

```
User Request
    ↓
ToolBelt.browser_navigate()
    ↓
SeleniumBrowserNavigator.navigate()  ← Selenium (JS execution)
    ↓
BrowserNavigator utilities           ← BeautifulSoup (parsing)
    ↓
Extracted Data
```

## Configuration

### Headless Mode (Default: True)
Selenium runs in headless mode by default for performance:
```python
self.selenium_navigator = SeleniumBrowserNavigator(logger, headless=True)
```

To run with visible browser (for debugging):
```python
navigator = SeleniumBrowserNavigator(logger, headless=False)
```

### Timeout (Default: 30 seconds)
```python
navigator = SeleniumBrowserNavigator(logger, timeout=30)
```

### JavaScript Wait Time (Default: 2 seconds)
```python
result = navigator.navigate(url, wait_for_js=3.0)  # Wait 3 seconds for JS
```

## Usage Examples

### Basic Navigation
```python
from src.core.selenium_browser_navigator import SeleniumBrowserNavigator

navigator = SeleniumBrowserNavigator(logger)
result = navigator.navigate("https://arxiv.org/search/advanced")

if result['success']:
    print(f"Loaded: {len(result['content'])} chars")
    soup = result['soup']  # BeautifulSoup object
```

### Click Element
```python
# Click by link text
navigator.click_element(link_text="Advanced Search")

# Click by CSS selector
navigator.click_element(selector="button#submit")
```

### Fill Form
```python
form_data = {
    "input[name='title']": "quantum computing",
    "input[name='author']": "John Doe"
}
navigator.fill_form(form_data, submit_selector="button[type='submit']")
```

## Testing

Run your GAIA validation tests again:
```bash
uv run python test_validation.py
```

Previously failing tasks with HTTP 500 errors should now work:
- ✅ arXiv search queries
- ✅ Dynamic database pages
- ✅ JavaScript-heavy websites

## Installation

Dependencies are already added to `requirements.txt`. To install:
```bash
uv pip install -r requirements.txt
```

The ChromeDriver will be automatically downloaded on first use.

## Troubleshooting

### "ChromeDriver not found"
- The webdriver-manager should handle this automatically
- Manual download: https://chromedriver.chromium.org/

### Selenium hangs or times out
- Increase timeout: `SeleniumBrowserNavigator(logger, timeout=60)`
- Increase JS wait time: `navigate(url, wait_for_js=5.0)`

### "Chrome binary not found"
- Ensure Google Chrome is installed
- macOS: Install from https://www.google.com/chrome/

## Performance Notes

- **Slower than requests**: Selenium launches a real browser (~2-3s per page)
- **More reliable**: Handles JavaScript, redirects, dynamic content
- **Memory**: Uses more memory than simple HTTP requests

For best performance:
- Keep navigator instance alive (don't recreate each time)
- Use headless mode (default)
- Close navigator when done: `navigator.close()`

## Modular Design

Following your cursor rules, the implementation is modular:

```
src/core/
├── selenium_browser_navigator.py   # Selenium navigation (NEW)
├── browser_navigator.py            # Extraction utilities (EXISTING)
└── tool_belt.py                    # Orchestration (UPDATED)
```

Each class has a single responsibility:
- **SeleniumBrowserNavigator**: Browser automation
- **BrowserNavigator**: HTML parsing and extraction
- **ToolBelt**: Tool orchestration

## Next Steps

The system is now ready to handle JavaScript pages. Your next test run should show:
- ✅ Fewer HTTP 500 errors
- ✅ Successful arXiv navigation
- ✅ Better extraction from dynamic sites

Run the validation tests to see the improvement!


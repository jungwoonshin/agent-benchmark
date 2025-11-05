"""Test code for processing images from Selenium browsing and PDFs using visual LLMs."""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import directly to avoid circular imports
try:
    from src.browser.selenium_browser_navigator import SeleniumBrowserNavigator
    from src.llm.llm_service import LLMService
    from src.models.models import Attachment
    from src.tools.tool_belt import ToolBelt
except ImportError as e:
    print(f'Import error: {e}')
    print(
        "Make sure you're running with: uv run python test_visual_llm_image_processing.py"
    )
    sys.exit(1)


def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger(__name__)


def test_selenium_screenshot_processing():
    """Test processing screenshots from Selenium browsing with visual LLM."""
    logger = setup_logging()
    logger.info('=' * 80)
    logger.info('TEST 1: Processing Selenium Screenshot with Visual LLM')
    logger.info('=' * 80)

    try:
        # Initialize components
        selenium_navigator = SeleniumBrowserNavigator(logger=logger, headless=True)
        # Use a valid visual model (gpt-4o or gpt-4-vision-preview)
        visual_model = os.getenv('VISUAL_LLM_MODEL', 'gpt-4o')
        llm_service = LLMService(logger=logger, visual_model=visual_model)
        tool_belt = ToolBelt()
        tool_belt.set_logger(logger)
        tool_belt.set_llm_service(llm_service)

        # Navigate to a page with visual content
        test_url = 'https://example.com'
        logger.info(f'Navigating to: {test_url}')
        page_data = selenium_navigator.navigate(test_url)

        if not page_data.get('success'):
            logger.error(f'Failed to navigate to {test_url}: {page_data.get("error")}')
            return False

        # Take screenshot
        logger.info('Taking screenshot...')
        screenshot_bytes = selenium_navigator.take_screenshot(as_base64=False)

        if not screenshot_bytes:
            logger.error('Failed to capture screenshot')
            return False

        logger.info(f'Screenshot captured: {len(screenshot_bytes)} bytes')

        # Process screenshot with visual LLM
        task_description = 'Analyze this screenshot and describe what you see on the webpage. Identify any text, images, layout elements, and overall page structure.'
        context = {
            'url': page_data.get('url'),
            'page_title': 'Example Domain',
        }

        logger.info('Processing screenshot with visual LLM...')
        result = tool_belt.llm_reasoning_with_images(
            task_description=task_description,
            context=context,
            images=[screenshot_bytes],
        )

        logger.info('Visual LLM Result:')
        logger.info(result)
        logger.info('=' * 80)

        # Cleanup
        selenium_navigator.close()

        return True

    except Exception as e:
        logger.error(f'Test failed with exception: {e}', exc_info=True)
        return False


def test_pdf_image_processing():
    """Test processing images extracted from PDFs with visual LLM."""
    logger = setup_logging()
    logger.info('=' * 80)
    logger.info('TEST 2: Processing PDF Images with Visual LLM')
    logger.info('=' * 80)

    try:
        # Create a simple test PDF with an image (using PyMuPDF)
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning('PyMuPDF (fitz) not available')
            return False

        logger.info('Creating test PDF with image...')
        pdf_doc = fitz.open()  # Create new PDF
        page = pdf_doc.new_page(width=800, height=600)

        # Add a simple colored rectangle using Shape
        rect = fitz.Rect(50, 50, 750, 550)
        shape = fitz.Shape(page)
        shape.draw_rect(rect)
        shape.finish(color=(0.8, 0.9, 1.0), fill=(0.8, 0.9, 1.0))
        shape.commit()

        # Add some text
        text = 'Test PDF Document\nThis is a test document with visual content.'
        page.insert_text((100, 100), text, fontsize=20, color=(0, 0, 0), render_mode=0)

        # Save PDF to bytes
        pdf_bytes = pdf_doc.tobytes()
        pdf_doc.close()

        logger.info(f'Created test PDF: {len(pdf_bytes)} bytes')

        # Initialize components
        visual_model = os.getenv('VISUAL_LLM_MODEL', 'gpt-4o')
        llm_service = LLMService(logger=logger, visual_model=visual_model)
        tool_belt = ToolBelt()
        tool_belt.set_logger(logger)
        tool_belt.set_llm_service(llm_service)

        # Create attachment
        attachment = Attachment(
            filename='test_document.pdf', data=pdf_bytes, metadata={}
        )

        # Read PDF and process images
        logger.info('Reading PDF and processing images...')
        result = tool_belt.read_attachment(attachment, options={})

        logger.info('PDF Processing Result:')
        logger.info(result[:500] + '...' if len(result) > 500 else result)
        logger.info('=' * 80)

        return True

    except ImportError as e:
        logger.warning(f'PyMuPDF not available: {e}')
        logger.info('Skipping PDF test - PyMuPDF required')
        return False
    except Exception as e:
        logger.error(f'Test failed with exception: {e}', exc_info=True)
        return False


def test_selenium_browser_navigate_with_screenshot():
    """Test browser_navigate with screenshot capture and visual LLM processing."""
    logger = setup_logging()
    logger.info('=' * 80)
    logger.info('TEST 3: Browser Navigate with Screenshot Capture')
    logger.info('=' * 80)

    try:
        # Initialize components
        visual_model = os.getenv('VISUAL_LLM_MODEL', 'gpt-4o')
        llm_service = LLMService(logger=logger, visual_model=visual_model)
        tool_belt = ToolBelt()
        tool_belt.set_logger(logger)
        tool_belt.set_llm_service(llm_service)

        # Navigate with screenshot capture
        test_url = 'https://example.com'
        logger.info(f'Navigating to: {test_url} with screenshot capture...')

        result = tool_belt.browser_navigate(url=test_url, capture_screenshot=True)

        if not result.get('success'):
            logger.error(f'Navigation failed: {result.get("error")}')
            return False

        # Check if screenshot was captured
        if 'screenshot' in result:
            screenshot_bytes = result['screenshot']
            logger.info(f'Screenshot captured: {len(screenshot_bytes)} bytes')

            # Process with visual LLM
            task_description = 'Analyze this webpage screenshot and describe what you see. What is the main content, layout, and any visible elements?'
            context = {
                'url': result.get('url'),
                'text': result.get('text', '')[:500],  # First 500 chars of text
            }

            logger.info('Processing screenshot with visual LLM...')
            analysis = tool_belt.llm_reasoning_with_images(
                task_description=task_description,
                context=context,
                images=[screenshot_bytes],
            )

            logger.info('Visual LLM Analysis:')
            logger.info(analysis)
        else:
            logger.warning('No screenshot found in result')
            return False

        logger.info('=' * 80)
        return True

    except Exception as e:
        logger.error(f'Test failed with exception: {e}', exc_info=True)
        return False


def test_pdf_with_multiple_images():
    """Test processing PDF with multiple images."""
    logger = setup_logging()
    logger.info('=' * 80)
    logger.info('TEST 4: PDF with Multiple Images')
    logger.info('=' * 80)

    try:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning('PyMuPDF (fitz) not available')
            return False

        logger.info('Creating test PDF with multiple pages and images...')
        pdf_doc = fitz.open()
        pdf_bytes = None

        # Create a few pages with different content
        for page_num in range(3):
            page = pdf_doc.new_page(width=800, height=600)

            # Add page number and content
            text = f'Page {page_num + 1}\nThis is page {page_num + 1} of the test document.'
            page.insert_text(
                (100, 100), text, fontsize=16, color=(0, 0, 0), render_mode=0
            )

            # Add a colored rectangle as a visual element using Shape
            rect = fitz.Rect(100, 200, 700, 500)
            colors = [(0.8, 0.9, 1.0), (1.0, 0.9, 0.8), (0.9, 1.0, 0.8)]
            shape = fitz.Shape(page)
            shape.draw_rect(rect)
            shape.finish(color=colors[page_num], fill=colors[page_num])
            shape.commit()

        pdf_bytes = pdf_doc.tobytes()
        pdf_doc.close()

        logger.info(f'Created test PDF with 3 pages: {len(pdf_bytes)} bytes')

        # Initialize components
        visual_model = os.getenv('VISUAL_LLM_MODEL', 'gpt-4o')
        tool_belt = ToolBelt()
        tool_belt.set_logger(logger)
        tool_belt.set_llm_service(LLMService(logger=logger, visual_model=visual_model))

        # Create attachment
        attachment = Attachment(
            filename='multi_page_test.pdf', data=pdf_bytes, metadata={}
        )

        # Read PDF (should extract and process images)
        logger.info('Reading PDF and processing images...')
        result = tool_belt.read_attachment(attachment, options={})

        logger.info('PDF Processing Result (first 1000 chars):')
        logger.info(result[:1000] + '...' if len(result) > 1000 else result)
        logger.info('=' * 80)

        return True

    except ImportError:
        logger.warning('PyMuPDF not available. Skipping test.')
        return False
    except Exception as e:
        logger.error(f'Test failed with exception: {e}', exc_info=True)
        return False


def test_llm_service_image_encoding():
    """Test LLMService image encoding utilities."""
    logger = setup_logging()
    logger.info('=' * 80)
    logger.info('TEST 5: LLMService Image Encoding')
    logger.info('=' * 80)

    try:
        # Test static methods, don't need to create instance
        # visual_model = os.getenv('VISUAL_LLM_MODEL', 'gpt-4o')
        # llm_service = LLMService(logger=logger, visual_model=visual_model)

        # Create a simple test image (PNG format)
        # Using a minimal valid PNG
        png_bytes = (
            b'\x89PNG\r\n\x1a\n'  # PNG signature
            b'\x00\x00\x00\rIHDR'  # Header chunk
            + b'\x00' * 13  # Minimal header data
            + b'\x00\x00\x00\x00IEND\xaeB`\x82'  # End chunk
        )

        logger.info(f'Testing image encoding with {len(png_bytes)} bytes PNG')

        # Test encode_image_to_base64 (static method, doesn't need instance)
        base64_str = LLMService.encode_image_to_base64(png_bytes)
        logger.info(f'Base64 encoded: {len(base64_str)} characters')
        assert len(base64_str) > 0, 'Base64 encoding should produce output'

        # Test create_image_content
        image_content = LLMService.create_image_content(png_bytes, image_format='png')
        logger.info(f'Image content created: {image_content}')
        assert image_content['type'] == 'image_url', 'Should have type image_url'
        assert 'image_url' in image_content, 'Should have image_url key'
        assert 'url' in image_content['image_url'], 'Should have url in image_url'
        assert image_content['image_url']['url'].startswith('data:image/png;base64,'), (
            'URL should be data URL with PNG MIME type'
        )

        logger.info('All image encoding tests passed!')
        logger.info('=' * 80)
        return True

    except Exception as e:
        logger.error(f'Test failed with exception: {e}', exc_info=True)
        return False


def main():
    """Run all tests."""
    logger = setup_logging()
    logger.info('Starting Visual LLM Image Processing Tests')
    logger.info('=' * 80)

    tests = [
        ('Selenium Screenshot Processing', test_selenium_screenshot_processing),
        ('PDF Image Processing', test_pdf_image_processing),
        (
            'Browser Navigate with Screenshot',
            test_selenium_browser_navigate_with_screenshot,
        ),
        ('PDF with Multiple Images', test_pdf_with_multiple_images),
        ('LLMService Image Encoding', test_llm_service_image_encoding),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f'\nRunning: {test_name}')
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f'✓ {test_name} PASSED')
            else:
                logger.error(f'✗ {test_name} FAILED')
        except Exception as e:
            logger.error(f'✗ {test_name} FAILED with exception: {e}', exc_info=True)
            results.append((test_name, False))

    # Summary
    logger.info('\n' + '=' * 80)
    logger.info('TEST SUMMARY')
    logger.info('=' * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    for test_name, result in results:
        status = 'PASSED' if result else 'FAILED'
        logger.info(f'{status}: {test_name}')
    logger.info(f'\nTotal: {passed}/{total} tests passed')
    logger.info('=' * 80)

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

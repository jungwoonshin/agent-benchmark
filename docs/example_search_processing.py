"""
Example demonstrating the systematic search result processing workflow.

This example shows how the SearchResultProcessor handles search results by:
1. Checking relevance with LLM
2. Classifying as web page vs file
3. Dispatching to appropriate handler
4. Extracting and structuring content
"""

import logging
import os
from dotenv import load_dotenv

from src.core.llm_service import LLMService
from src.core.models import SearchResult
from src.core.search_result_processor import SearchResultProcessor
from src.core.tool_belt import ToolBelt

# Load environment variables
load_dotenv()


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger(__name__)


def main():
    """Demonstrate search result processing."""
    logger = setup_logging()
    logger.info('Starting search result processing example...')

    # Initialize components
    tool_belt = ToolBelt()
    tool_belt.set_logger(logger)

    llm_service = LLMService(logger, model='gpt-4o')

    search_processor = SearchResultProcessor(
        llm_service=llm_service,
        tool_belt=tool_belt,
        logger=logger,
    )

    # Example 1: Mixed search results (web pages and files)
    logger.info('\n' + '=' * 80)
    logger.info('Example 1: Processing mixed search results')
    logger.info('=' * 80)

    problem = 'Find information about climate change research papers published in 2020'
    subtask_description = 'Search for climate change papers from 2020'

    # Simulate search results
    search_results = [
        SearchResult(
            title='Nature Climate Change 2020 Archive',
            snippet='Archive of Nature Climate Change articles published in 2020',
            url='https://www.nature.com/nclimate/articles?year=2020',
            relevance_score=0.9,
        ),
        SearchResult(
            title='Climate Report 2020.pdf',
            snippet='Comprehensive report on climate change research in 2020',
            url='https://example.com/reports/climate-2020.pdf',
            relevance_score=0.85,
        ),
        SearchResult(
            title='Blog: My thoughts on climate',
            snippet='Personal blog post about climate change opinions',
            url='https://blog.example.com/my-climate-thoughts',
            relevance_score=0.3,
        ),
    ]

    query_analysis = {
        'explicit_requirements': ['climate change research', 'published in 2020'],
    }

    # Process search results
    processing_result = search_processor.process_search_results(
        search_results=search_results,
        subtask_description=subtask_description,
        problem=problem,
        query_analysis=query_analysis,
        attachments=[],  # Empty list to collect downloaded files
        max_results_to_process=3,
    )

    # Display results
    logger.info('\nProcessing Results:')
    logger.info(f"  Total processed: {processing_result['processed_count']}")
    logger.info(f"  Relevant: {processing_result['relevant_count']}")
    logger.info(f"  Web pages processed: {len(processing_result['web_pages'])}")
    logger.info(f"  Files downloaded: {len(processing_result['downloaded_files'])}")

    if processing_result['web_pages']:
        logger.info('\nWeb Pages:')
        for page in processing_result['web_pages']:
            logger.info(f"  - {page['title']}")
            logger.info(f"    URL: {page['url']}")
            logger.info(
                f"    Content length: {len(page['content'])} chars"
            )

    if processing_result['downloaded_files']:
        logger.info('\nDownloaded Files:')
        for file in processing_result['downloaded_files']:
            logger.info(f"  - Type: {file['type']}")
            logger.info(f"    URL: {file['url']}")
            logger.info(f"    Content length: {len(file['content'])} chars")

    # Example 2: Web page only results
    logger.info('\n' + '=' * 80)
    logger.info('Example 2: Processing web page results only')
    logger.info('=' * 80)

    problem = 'Count the number of articles about machine learning on ArXiv in June 2023'
    subtask_description = 'Navigate to ArXiv and count machine learning articles from June 2023'

    search_results = [
        SearchResult(
            title='ArXiv Machine Learning Archive',
            snippet='Archive of machine learning papers on ArXiv.org',
            url='https://arxiv.org/list/cs.LG/2306',
            relevance_score=0.95,
        ),
    ]

    query_analysis = {
        'explicit_requirements': ['count articles', 'machine learning', 'June 2023'],
    }

    processing_result = search_processor.process_search_results(
        search_results=search_results,
        subtask_description=subtask_description,
        problem=problem,
        query_analysis=query_analysis,
        attachments=[],
        max_results_to_process=1,
    )

    logger.info('\nProcessing Results:')
    logger.info(f"  Total processed: {processing_result['processed_count']}")
    logger.info(f"  Relevant: {processing_result['relevant_count']}")
    logger.info(f"  Web pages processed: {len(processing_result['web_pages'])}")

    # Example 3: Demonstrating relevance filtering
    logger.info('\n' + '=' * 80)
    logger.info('Example 3: Demonstrating LLM-based relevance filtering')
    logger.info('=' * 80)

    problem = 'Find the population of Tokyo in 2020'
    subtask_description = 'Search for Tokyo population statistics for 2020'

    search_results = [
        SearchResult(
            title='Tokyo Population 2020 - Official Statistics',
            snippet='Official population statistics for Tokyo metropolitan area in 2020',
            url='https://www.metro.tokyo.lg.jp/english/statistics/2020',
            relevance_score=0.95,
        ),
        SearchResult(
            title='Tokyo Travel Guide',
            snippet='Best places to visit in Tokyo, restaurants, hotels',
            url='https://travel.example.com/tokyo',
            relevance_score=0.2,
        ),
        SearchResult(
            title='Tokyo Olympics 2020',
            snippet='Information about the Tokyo 2020 Olympic Games',
            url='https://olympics.example.com/tokyo2020',
            relevance_score=0.4,
        ),
    ]

    query_analysis = {
        'explicit_requirements': ['Tokyo population', 'year 2020'],
    }

    processing_result = search_processor.process_search_results(
        search_results=search_results,
        subtask_description=subtask_description,
        problem=problem,
        query_analysis=query_analysis,
        attachments=[],
        max_results_to_process=3,
    )

    logger.info('\nProcessing Results (with relevance filtering):')
    logger.info(f"  Total processed: {processing_result['processed_count']}")
    logger.info(
        f"  Relevant (after LLM filtering): {processing_result['relevant_count']}"
    )
    logger.info(
        f"  Note: LLM should filter out travel guide and Olympics results as not relevant"
    )

    logger.info('\n' + '=' * 80)
    logger.info('Example complete!')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()


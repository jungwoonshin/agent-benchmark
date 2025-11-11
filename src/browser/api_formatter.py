"""API data formatter for various external APIs."""

import base64
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from src.tools import ToolBelt


class APIFormatter:
    """Formats API response data into readable text content."""

    def __init__(
        self,
        tool_belt: Optional['ToolBelt'] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize API formatter.

        Args:
            tool_belt: Optional ToolBelt instance for file operations (e.g., PDF downloads).
            logger: Optional logger instance.
        """
        self.tool_belt = tool_belt
        self.logger = logger or logging.getLogger(__name__)

    def format(
        self,
        api_data: Any,
        api_name: str,
        url: str,
        problem: Optional[str] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Format API response data into readable text content.

        Args:
            api_data: Raw API response data.
            api_name: Name of the API (github, wikipedia, youtube, etc.).
            url: Original URL.
            problem: Optional problem description (for PDF processing).
            query_analysis: Optional query analysis (for PDF processing).

        Returns:
            Formatted text content or None if formatting fails.
        """
        if not api_data:
            return None

        try:
            content_parts = [
                f'[API Data from {api_name.upper()}]',
                f'Source URL: {url}',
            ]

            if api_name == 'github':
                self._format_github(api_data, content_parts)
            elif api_name == 'wikipedia':
                self._format_wikipedia(api_data, content_parts)
            elif api_name == 'youtube':
                self._format_youtube(api_data, content_parts)
            elif api_name == 'twitter':
                self._format_twitter(api_data, content_parts)
            elif api_name == 'reddit':
                self._format_reddit(api_data, content_parts)
            elif api_name == 'arxiv':
                self._format_arxiv(api_data, content_parts, problem, query_analysis)
            else:
                self._format_generic(api_data, content_parts)

            return '\n'.join(content_parts)

        except Exception as e:
            self.logger.warning(f'Failed to format API data from {api_name}: {e}')
            return None

    def _format_github(self, api_data: Any, content_parts: list) -> None:
        """Format GitHub API responses."""
        if isinstance(api_data, list):
            for idx, item in enumerate(api_data[:10], 1):
                if isinstance(item, dict):
                    if 'title' in item or 'message' in item:
                        title = item.get('title') or item.get('message', '')[:100]
                        number = item.get('number') or item.get('sha', '')[:7]
                        state = item.get('state', '')
                        labels = [
                            label.get('name', '') for label in item.get('labels', [])
                        ]
                        created = item.get('created_at', '')
                        body = item.get('body', '')[:500] if item.get('body') else ''

                        content_parts.append(f'\n[{idx}] Issue/Commit #{number}')
                        if title:
                            content_parts.append(f'Title: {title}')
                        if state:
                            content_parts.append(f'State: {state}')
                        if labels:
                            content_parts.append(f'Labels: {", ".join(labels)}')
                        if created:
                            content_parts.append(f'Created: {created}')
                        if body:
                            content_parts.append(f'Description: {body}')
        elif isinstance(api_data, dict):
            if 'title' in api_data:
                content_parts.append(f'Title: {api_data.get("title", "")}')
                content_parts.append(f'Number: {api_data.get("number", "")}')
                content_parts.append(f'State: {api_data.get("state", "")}')
                labels = [label.get('name', '') for label in api_data.get('labels', [])]
                if labels:
                    content_parts.append(f'Labels: {", ".join(labels)}')
                content_parts.append(f'Created: {api_data.get("created_at", "")}')
                if api_data.get('body'):
                    content_parts.append(f'\nDescription:\n{api_data.get("body", "")}')
            elif 'message' in api_data:
                content_parts.append(f'Commit: {api_data.get("sha", "")[:7]}')
                content_parts.append(f'Message: {api_data.get("message", "")}')
                content_parts.append(
                    f'Author: {api_data.get("author", {}).get("name", "")}'
                )
                content_parts.append(
                    f'Date: {api_data.get("commit", {}).get("author", {}).get("date", "")}'
                )
            elif 'name' in api_data and 'download_url' in api_data:
                content_parts.append(f'File: {api_data.get("name", "")}')
                content_parts.append(f'Path: {api_data.get("path", "")}')
                if api_data.get('content'):
                    try:
                        decoded = base64.b64decode(api_data['content']).decode('utf-8')
                        content_parts.append(f'\nContent:\n{decoded[:2000]}')
                    except Exception:
                        pass

    def _format_wikipedia(self, api_data: Any, content_parts: list) -> None:
        """Format Wikipedia API responses."""
        if isinstance(api_data, dict):
            content_parts.append(f'Title: {api_data.get("title", "")}')
            if api_data.get('revision_id'):
                content_parts.append(f'Revision ID: {api_data.get("revision_id")}')
            if api_data.get('timestamp'):
                content_parts.append(f'Timestamp: {api_data.get("timestamp")}')
            if api_data.get('content'):
                content = api_data.get('content', '')
                if len(content) > 5000:
                    content = content[:5000] + '\n\n[... Content truncated ...]'
                content_parts.append(f'\nContent:\n{content}')

    def _format_youtube(self, api_data: Any, content_parts: list) -> None:
        """Format YouTube API responses."""
        if isinstance(api_data, dict):
            snippet = api_data.get('snippet', {})
            content_parts.append(f'Title: {snippet.get("title", "")}')
            content_parts.append(f'Channel: {snippet.get("channelTitle", "")}')
            content_parts.append(f'Published: {snippet.get("publishedAt", "")}')
            if snippet.get('description'):
                desc = snippet.get('description', '')[:1000]
                content_parts.append(f'\nDescription:\n{desc}')
            stats = api_data.get('statistics', {})
            if stats:
                content_parts.append('\nStatistics:')
                content_parts.append(f'  Views: {stats.get("viewCount", "N/A")}')
                content_parts.append(f'  Likes: {stats.get("likeCount", "N/A")}')

    def _format_twitter(self, api_data: Any, content_parts: list) -> None:
        """Format Twitter API responses."""
        if isinstance(api_data, list):
            for idx, tweet in enumerate(api_data[:10], 1):
                if isinstance(tweet, dict):
                    text = tweet.get('text', '')
                    created = tweet.get('created_at', '')
                    metrics = tweet.get('public_metrics', {})
                    content_parts.append(f'\n[{idx}] Tweet ({created})')
                    content_parts.append(f'Text: {text}')
                    if metrics:
                        content_parts.append(f'Metrics: {metrics}')

    def _format_reddit(self, api_data: Any, content_parts: list) -> None:
        """Format Reddit API responses."""
        if isinstance(api_data, list):
            for idx, post in enumerate(api_data[:10], 1):
                if isinstance(post, dict):
                    title = post.get('title', '')
                    selftext = post.get('selftext', '')
                    created = post.get('created_utc', '')
                    score = post.get('score', 0)
                    subreddit = post.get('subreddit', '')
                    content_parts.append(
                        f'\n[{idx}] Post in r/{subreddit} (Score: {score})'
                    )
                    content_parts.append(f'Title: {title}')
                    if selftext:
                        content_parts.append(f'Content: {selftext[:1000]}')

    def _format_arxiv(
        self,
        api_data: Any,
        content_parts: list,
        problem: Optional[str] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Format arXiv API responses with full paper content."""
        if isinstance(api_data, dict):
            content_parts.append(f'Title: {api_data.get("title", "")}')
            content_parts.append(f'Authors: {", ".join(api_data.get("authors", []))}')
            published = (
                api_data.get('submission_date_text')
                or api_data.get('submission_date')
                or api_data.get('published', '')
            )
            if published:
                content_parts.append(f'Published: {published}')
            if api_data.get('summary'):
                summary = api_data.get('summary', '')
                content_parts.append(f'\nAbstract:\n{summary}')

            # Try to get full paper content from PDF
            pdf_attachment = api_data.get('pdf_attachment')
            pdf_url = api_data.get('pdf_url')

            # If no PDF attachment but we have a URL, try to download it
            if not pdf_attachment and pdf_url and self.tool_belt:
                try:
                    self.logger.info(f'Downloading PDF from arXiv: {pdf_url}')
                    pdf_attachment = self.tool_belt.download_file_from_url(pdf_url)
                except Exception as e:
                    self.logger.warning(f'Failed to download PDF from {pdf_url}: {e}')
                    pdf_attachment = None

            # Extract full text from PDF if available
            if pdf_attachment and self.tool_belt:
                try:
                    self.logger.info(
                        f'Extracting full text from PDF: {pdf_attachment.filename}'
                    )
                    # Pass problem/query_analysis to get structured dict with image_analysis
                    pdf_content = self.tool_belt.read_attachment(
                        pdf_attachment,
                        problem=problem,
                        query_analysis=query_analysis,
                    )

                    if (
                        isinstance(pdf_content, dict)
                        and pdf_content.get('type') == 'pdf'
                    ):
                        full_text = pdf_content.get('full_text', '')
                        if full_text:
                            content_parts.append(
                                f'\n\nFull Paper Content:\n{full_text}'
                            )

                        # Store image analysis for relevance checking
                        image_analysis = pdf_content.get('image_analysis', '')
                        if image_analysis:
                            # Store it in api_data for later retrieval
                            api_data['_image_analysis'] = image_analysis
                            self.logger.info(
                                f'Stored image analysis from PDF ({len(image_analysis)} chars) for relevance checking'
                            )

                        sections = pdf_content.get('sections', [])
                        if sections:
                            content_parts.append('\n\nPaper Sections:')
                            for section in sections:
                                section_title = section.get('title', 'Untitled')
                                section_text = section.get('text', '')
                                if section_text:
                                    content_parts.append(
                                        f'\n{section_title}:\n{section_text}'
                                    )
                    elif isinstance(pdf_content, str) and pdf_content:
                        content_parts.append(f'\n\nFull Paper Content:\n{pdf_content}')
                except Exception as e:
                    self.logger.warning(f'Failed to extract text from PDF: {e}')
                    if pdf_url:
                        content_parts.append(f'\n\nPDF available at: {pdf_url}')
            elif pdf_url:
                content_parts.append(f'\n\nPDF available at: {pdf_url}')

    def _format_generic(self, api_data: Any, content_parts: list) -> None:
        """Format generic API responses."""
        if isinstance(api_data, dict):
            content_parts.append('\nData:')
            content_parts.append(json.dumps(api_data, indent=2)[:5000])
        elif isinstance(api_data, list):
            content_parts.append(f'\nData ({len(api_data)} items):')
            for idx, item in enumerate(api_data[:5], 1):
                if isinstance(item, dict):
                    content_parts.append(
                        f'\n[{idx}] {json.dumps(item, indent=2)[:1000]}'
                    )
        else:
            content_parts.append(f'\nData: {str(api_data)[:5000]}')

"""Result formatting module for search results.

Handles formatting of final processed results.
"""

from typing import Any, Dict, Optional


class ResultFormatter:
    """Formats processed search results into final output format."""

    @staticmethod
    def format_result(
        result: Any,  # SearchResult
        is_file: bool,
        file_type: Optional[str],
        extracted_data: Dict[str, Any],
        summarized_content: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Format a processed result into the final output format.

        Args:
            result: SearchResult object.
            is_file: Whether the result is a file.
            file_type: Type of file if it's a file.
            extracted_data: Extracted data dictionary.
            summarized_content: Summarized content.

        Returns:
            Formatted result dictionary or None.
        """
        if is_file and extracted_data:
            if isinstance(extracted_data, dict) and extracted_data.get('type') == 'pdf':
                return {
                    'type': 'pdf',
                    'title': result.title,
                    'url': result.url,
                    'sections': extracted_data.get('sections', []),
                    'image_analysis': extracted_data.get('image_analysis', ''),
                    'content': summarized_content,
                    'data': {
                        'url': result.url,
                        'type': file_type,
                        'title': result.title,
                        'sections': extracted_data.get('sections', []),
                        'image_analysis': extracted_data.get('image_analysis', ''),
                        'content': summarized_content,
                    },
                }
            else:
                return {
                    'type': 'file',
                    'title': result.title,
                    'url': result.url,
                    'content': summarized_content,
                    'data': {
                        'url': result.url,
                        'type': file_type,
                        'title': result.title,
                        'content': summarized_content,
                    },
                }
        elif not is_file and extracted_data:
            if isinstance(extracted_data, dict):
                page_content = extracted_data.get('content', summarized_content)
                image_analysis = extracted_data.get('image_analysis', '')
                return {
                    'type': 'web_page',
                    'title': result.title,
                    'url': result.url,
                    'content': page_content,
                    'image_analysis': image_analysis,
                    'data': {
                        'url': result.url,
                        'title': result.title,
                        'content': page_content,
                        'image_analysis': image_analysis,
                    },
                }
            else:
                return {
                    'type': 'web_page',
                    'title': result.title,
                    'url': result.url,
                    'content': summarized_content,
                    'data': {
                        'url': result.url,
                        'title': result.title,
                        'content': summarized_content,
                    },
                }

        return None

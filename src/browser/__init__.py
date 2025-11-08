"""Browser-related modules for web navigation and search result processing."""

from .browser import Browser
from .content_type_classifier import ContentTypeClassifier
from .content_type_detector import ContentTypeDetector
from .file_type_navigator import FileTypeNavigator
from .search_result_processor import SearchResultProcessor

__all__ = [
    'Browser',
    'ContentTypeClassifier',
    'ContentTypeDetector',
    'FileTypeNavigator',
    'SearchResultProcessor',
]

"""Error handling utilities: classification, retries, and circuit breaker."""

import logging
import random
import time
from datetime import datetime
from enum import Enum
from typing import Callable, Optional, TypeVar

try:
    import httpx
except ImportError:
    httpx = None

try:
    import openai
except ImportError:
    openai = None

import requests

T = TypeVar('T')


class ErrorType(Enum):
    """Classification of error types."""

    TRANSIENT = 'transient'  # Temporary, retryable errors
    PERMANENT = 'permanent'  # Permanent errors, don't retry
    TIMEOUT = 'timeout'  # Timeout errors, retryable with backoff
    RATE_LIMIT = 'rate_limit'  # Rate limit errors, retry with longer backoff


class ErrorCategory(Enum):
    """Category of error source."""

    NETWORK = 'network'  # Network/DNS/connection issues
    API = 'api'  # API errors (4xx, 5xx, authentication)
    USER = 'user'  # User input errors (validation, invalid params)
    SYSTEM = 'system'  # System errors (timeouts, unavailable resources)


def classify_error(error: Exception) -> tuple[ErrorType, ErrorCategory, str]:
    """
    Classify an error into type and category, and generate a user-friendly message.

    Args:
        error: The exception to classify.

    Returns:
        Tuple of (ErrorType, ErrorCategory, user_message).
    """
    error_class = type(error).__name__
    error_msg = str(error).lower()

    # OpenAI/OpenAI API errors
    if openai and isinstance(error, openai.APIConnectionError):
        return (
            ErrorType.TRANSIENT,
            ErrorCategory.NETWORK,
            f'Network connection issue: Unable to reach OpenAI API. This is usually temporary. Error: {str(error)}',
        )
    if openai and isinstance(error, openai.APITimeoutError):
        return (
            ErrorType.TIMEOUT,
            ErrorCategory.NETWORK,
            'Request timed out while waiting for OpenAI API response. This may be due to network issues or API load.',
        )
    if openai and isinstance(error, openai.RateLimitError):
        return (
            ErrorType.RATE_LIMIT,
            ErrorCategory.API,
            'Rate limit exceeded: Too many requests to OpenAI API. Please wait before retrying.',
        )
    if openai and isinstance(error, openai.APIError):
        # Check error message for more specific classification
        if '429' in str(error) or 'rate' in error_msg:
            return (
                ErrorType.RATE_LIMIT,
                ErrorCategory.API,
                'Rate limit error from OpenAI API. Please wait before retrying.',
            )
        if '401' in str(error) or '403' in str(error) or 'authentication' in error_msg:
            return (
                ErrorType.PERMANENT,
                ErrorCategory.API,
                'Authentication error: Invalid API key or insufficient permissions. Please check your API configuration.',
            )
        if '500' in str(error) or '502' in str(error) or '503' in str(error):
            return (
                ErrorType.TRANSIENT,
                ErrorCategory.API,
                'OpenAI API server error. This is usually temporary. Please retry.',
            )
        return (
            ErrorType.PERMANENT,
            ErrorCategory.API,
            f'OpenAI API error: {str(error)}',
        )

    # HTTP/Network errors (httpx - used by OpenAI)
    if httpx:
        if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
            if 'nodename' in error_msg or 'servname' in error_msg or 'dns' in error_msg:
                return (
                    ErrorType.TRANSIENT,
                    ErrorCategory.NETWORK,
                    'DNS resolution failed: Unable to resolve hostname. This may be a network connectivity issue.',
                )
            return (
                ErrorType.TRANSIENT,
                ErrorCategory.NETWORK,
                f'Network connection error: {str(error)}',
            )
        if isinstance(error, (httpx.ReadTimeout, httpx.TimeoutException)):
            return (
                ErrorType.TIMEOUT,
                ErrorCategory.NETWORK,
                'Network timeout: Request took too long. This may be due to network slowness or server load.',
            )

    # Requests library errors
    if isinstance(error, requests.exceptions.ConnectionError):
        return (
            ErrorType.TRANSIENT,
            ErrorCategory.NETWORK,
            'Connection error: Unable to connect to server. This is usually temporary.',
        )
    if isinstance(error, requests.exceptions.Timeout):
        return (
            ErrorType.TIMEOUT,
            ErrorCategory.NETWORK,
            'Request timeout: The request took too long to complete.',
        )
    if isinstance(error, requests.exceptions.HTTPError):
        status_code = (
            getattr(error.response, 'status_code', None)
            if hasattr(error, 'response')
            else None
        )
        if status_code == 403:
            return (
                ErrorType.PERMANENT,
                ErrorCategory.API,
                'Access forbidden (403): The server denied access to this resource. This may require authentication or a different approach.',
            )
        if status_code == 404:
            return (
                ErrorType.PERMANENT,
                ErrorCategory.USER,
                'Resource not found (404): The requested URL does not exist.',
            )
        if status_code in (429,):
            return (
                ErrorType.RATE_LIMIT,
                ErrorCategory.API,
                'Rate limit exceeded (429): Too many requests. Please wait before retrying.',
            )
        if status_code in (500, 502, 503, 504):
            return (
                ErrorType.TRANSIENT,
                ErrorCategory.API,
                f'Server error ({status_code}): The server encountered an error. This is usually temporary.',
            )
        if status_code in (400, 401):
            return (
                ErrorType.PERMANENT,
                ErrorCategory.USER,
                f'Client error ({status_code}): Invalid request. {str(error)}',
            )
        return (
            ErrorType.PERMANENT,
            ErrorCategory.API,
            f'HTTP error ({status_code}): {str(error)}',
        )

    # Generic timeout errors
    if 'timeout' in error_msg or 'timed out' in error_msg:
        return (
            ErrorType.TIMEOUT,
            ErrorCategory.NETWORK,
            f'Operation timed out: {str(error)}',
        )

    # Default classification
    return (
        ErrorType.PERMANENT,
        ErrorCategory.SYSTEM,
        f'Unexpected error: {str(error)}',
    )


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit.
            recovery_timeout: Seconds to wait before attempting recovery.
            logger: Optional logger instance.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = 'closed'  # closed, open, half_open
        self.logger = logger or logging.getLogger(__name__)

    def record_success(self):
        """Record a successful operation."""
        if self.state != 'closed':
            self.logger.info('Circuit breaker: Resetting to closed state')
        self.state = 'closed'
        self.failure_count = 0
        self.last_failure_time = None

    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            if self.state != 'open':
                self.logger.warning(
                    f'Circuit breaker: Opening circuit after {self.failure_count} failures'
                )
            self.state = 'open'

    def can_proceed(self) -> bool:
        """
        Check if operation can proceed.

        Returns:
            True if operation should proceed, False if circuit is open.
        """
        if self.state == 'closed':
            return True

        if self.state == 'open':
            if self.last_failure_time:
                time_since_failure = (
                    datetime.now() - self.last_failure_time
                ).total_seconds()
                if time_since_failure >= self.recovery_timeout:
                    self.logger.info(
                        'Circuit breaker: Moving to half-open state for recovery attempt'
                    )
                    self.state = 'half_open'
                    return True
            return False

        if self.state == 'half_open':
            return True

        return False

    def get_state_message(self) -> str:
        """Get human-readable state message."""
        if self.state == 'open':
            wait_time = (
                self.recovery_timeout
                - (datetime.now() - self.last_failure_time).total_seconds()
                if self.last_failure_time
                else self.recovery_timeout
            )
            return f'Circuit breaker is OPEN. Too many failures ({self.failure_count}). Will attempt recovery in {max(0, int(wait_time))} seconds.'
        return f'Circuit breaker is {self.state.upper()}'


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    logger: Optional[logging.Logger] = None,
    error_classifier: Optional[
        Callable[[Exception], tuple[ErrorType, ErrorCategory, str]]
    ] = None,
) -> T:
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry (no arguments).
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        exponential_base: Base for exponential backoff.
        jitter: Whether to add random jitter to delays.
        logger: Optional logger instance.
        error_classifier: Optional function to classify errors.

    Returns:
        Result of function call.

    Raises:
        Last exception if all retries fail.
    """
    if error_classifier is None:
        error_classifier = classify_error

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            result = func()
            if logger and attempt > 0:
                logger.info(f'Retry successful on attempt {attempt + 1}')
            return result
        except Exception as e:
            last_exception = e
            error_type, error_category, user_message = error_classifier(e)

            # Don't retry permanent errors
            if error_type == ErrorType.PERMANENT:
                if logger:
                    logger.error(f'Permanent error, not retrying: {user_message}')
                raise

            # If this was the last attempt, raise
            if attempt >= max_retries:
                if logger:
                    logger.error(
                        f'Max retries ({max_retries}) exceeded. Last error: {user_message}'
                    )
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base**attempt), max_delay)
            if jitter:
                # Add random jitter (0-20% of delay)
                jitter_amount = random.uniform(0, delay * 0.2)
                delay += jitter_amount

            # Special handling for rate limits - longer delay
            if error_type == ErrorType.RATE_LIMIT:
                delay = min(delay * 2, max_delay * 2)

            if logger:
                logger.warning(
                    f'Attempt {attempt + 1}/{max_retries + 1} failed ({error_category.value}, {error_type.value}). '
                    f'Retrying in {delay:.2f}s. Error: {user_message}'
                )

            time.sleep(delay)

    # Should not reach here, but just in case
    raise last_exception

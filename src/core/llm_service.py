"""LLM service for OpenAI API integration."""

import logging
import os
from typing import Dict, List, Optional

import openai
from dotenv import load_dotenv

from .error_handling import CircuitBreaker, classify_error, retry_with_backoff

# Load environment variables
load_dotenv()


class LLMService:
    """Service for interacting with OpenAI API."""

    # Class-level cache for models that don't support response_format
    _response_format_unsupported_models: set = set()
    # Class-level circuit breaker (shared across instances)
    _circuit_breaker: Optional[CircuitBreaker] = None

    def __init__(
        self,
        logger: logging.Logger,
        model: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        enable_circuit_breaker: bool = True,
    ):
        """
        Initialize LLM service.

        Args:
            logger: Logger instance for logging.
            model: OpenAI model to use (default: from LLM_MODEL env var, or 'gpt-5').
            timeout: Request timeout in seconds (default: 60.0).
            max_retries: Maximum number of retries for transient errors (default: 3).
            enable_circuit_breaker: Whether to use circuit breaker pattern (default: True).
        """
        self.logger = logger
        # Load model from environment variable if not provided
        self.model = model or os.getenv('LLM_MODEL', 'gpt-5')
        self.timeout = timeout
        self.max_retries = max_retries
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                'OPENAI_API_KEY not found in environment variables. '
                'Please set it in .env file.'
            )
        # Initialize OpenAI client with timeout
        self.client = openai.OpenAI(
            api_key=api_key,
            timeout=openai.Timeout(timeout),
        )

        # Initialize circuit breaker if enabled (shared across instances)
        if enable_circuit_breaker:
            if LLMService._circuit_breaker is None:
                LLMService._circuit_breaker = CircuitBreaker(
                    failure_threshold=5,
                    recovery_timeout=60,
                    logger=logger,
                )
            self.circuit_breaker = LLMService._circuit_breaker
        else:
            self.circuit_breaker = None

        self.logger.info(
            f'LLMService initialized with model: {model}, timeout: {timeout}s, max_retries: {max_retries}'
        )

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        _skip_logging: bool = False,
    ) -> str:
        """
        Make a call to the OpenAI API with retry logic and error handling.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in response.
            response_format: Optional response format (e.g., {"type": "json_object"}).

        Returns:
            Response content as string.

        Raises:
            Exception: If the call fails after all retries, with improved error message.
        """
        self.logger.debug(f'Calling LLM with {len(messages)} messages')

        # Extract prompts from messages for logging (if not skipped)
        system_prompts = [
            msg['content'] for msg in messages if msg.get('role') == 'system'
        ]
        user_prompts = [msg['content'] for msg in messages if msg.get('role') == 'user']

        # Log input prompts if not skipped (call_with_system_prompt handles its own logging)
        should_log = not _skip_logging and (system_prompts or user_prompts)
        if should_log:
            self.logger.info('=' * 80)
            self.logger.info('LLM CALL INPUT (from messages):')
            if system_prompts:
                for i, prompt in enumerate(system_prompts):
                    self.logger.info(f'System Prompt {i + 1}:\n{prompt}')
            if user_prompts:
                for i, prompt in enumerate(user_prompts):
                    self.logger.info(f'User Prompt {i + 1}:\n{prompt}')
            self.logger.info(
                f'Parameters: temperature={temperature}, max_tokens={max_tokens}, response_format={response_format}'
            )

        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_proceed():
            error_type, error_category, user_message = classify_error(
                Exception(self.circuit_breaker.get_state_message())
            )
            raise Exception(
                f'Circuit breaker is open: {self.circuit_breaker.get_state_message()}. '
                f'Please wait before retrying.'
            )

        kwargs = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
        }
        if max_tokens:
            kwargs['max_tokens'] = max_tokens

        def _make_api_call(with_format: bool = True) -> str:
            """Internal function to make the API call."""
            call_kwargs = kwargs.copy()
            if with_format and response_format:
                call_kwargs['response_format'] = response_format

            response = self.client.chat.completions.create(**call_kwargs)
            content = response.choices[0].message.content
            self.logger.debug(f'LLM response received: {len(content)} characters')

            # Log output if we logged input (i.e., if call() was called directly)
            if should_log:
                self.logger.info('LLM CALL OUTPUT:')
                self.logger.info(f'Response:\n{content}')
                self.logger.info('=' * 80)

            # Record success in circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.record_success()

            return content

        # Try with response_format first if provided
        if response_format:
            try:
                return retry_with_backoff(
                    lambda: _make_api_call(with_format=True),
                    max_retries=self.max_retries,
                    base_delay=1.0,
                    max_delay=60.0,
                    logger=self.logger,
                )
            except Exception as e:
                # If response_format is not supported, retry without it
                error_msg = str(e).lower()
                if (
                    'response_format' in error_msg
                    or 'not supported' in error_msg
                    or self.model in LLMService._response_format_unsupported_models
                ):
                    self.logger.warning(
                        f'Model {self.model} does not support response_format, retrying without it'
                    )
                    LLMService._response_format_unsupported_models.add(self.model)
                    # Create a copy of messages and add JSON instruction if needed
                    if response_format.get('type') == 'json_object':
                        messages_copy = []
                        for i, msg in enumerate(messages):
                            msg_copy = msg.copy()
                            if msg_copy['role'] == 'user' and i == len(messages) - 1:
                                msg_copy['content'] += (
                                    '\n\nIMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text.'
                                )
                            messages_copy.append(msg_copy)
                        kwargs['messages'] = messages_copy
                    # Fall through to try without response_format
                else:
                    # Record failure in circuit breaker
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure()

                    # Classify error and raise with better message
                    error_type, error_category, user_message = classify_error(e)
                    self.logger.error(
                        f'LLM API call failed ({error_category.value}/{error_type.value}): {user_message}',
                        exc_info=True,
                    )
                    raise Exception(user_message) from e

        # Call without response_format (either not requested or retry after error)
        try:
            return retry_with_backoff(
                lambda: _make_api_call(with_format=False),
                max_retries=self.max_retries,
                base_delay=1.0,
                max_delay=60.0,
                logger=self.logger,
            )
        except Exception as e:
            # Record failure in circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()

            # Classify error and raise with better message
            error_type, error_category, user_message = classify_error(e)
            self.logger.error(
                f'LLM API call failed ({error_category.value}/{error_type.value}): {user_message}',
                exc_info=True,
            )
            raise Exception(user_message) from e

    def call_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Convenience method for single-turn conversation with system prompt.

        Args:
            system_prompt: System instruction prompt.
            user_prompt: User query prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            response_format: Optional response format (e.g., {"type": "json_object"}).

        Returns:
            Response content as string.
        """
        # Log input prompts
        self.logger.info('=' * 80)
        self.logger.info('LLM CALL INPUT:')
        self.logger.info(f'System Prompt:\n{system_prompt}')
        self.logger.info(f'User Prompt:\n{user_prompt}')
        self.logger.info(
            f'Parameters: temperature={temperature}, max_tokens={max_tokens}, response_format={response_format}'
        )

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
        response = self.call(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            _skip_logging=True,  # Skip logging in call() since we handle it here
        )

        # Log output
        self.logger.info('LLM CALL OUTPUT:')
        self.logger.info(f'Response:\n{response}')
        self.logger.info('=' * 80)

        return response

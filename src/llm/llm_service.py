"""LLM service for OpenAI API integration."""

import base64
import logging
import os
from typing import Dict, List, Optional, Union

import openai
from dotenv import load_dotenv

from ..utils import CircuitBreaker, classify_error, retry_with_backoff

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
        visual_model: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        enable_circuit_breaker: bool = True,
    ):
        """
        Initialize LLM service.

        Args:
            logger: Logger instance for logging.
            model: OpenAI model to use (default: from LLM_MODEL env var, or 'gpt-5').
            visual_model: Vision model for image processing (default: from VISUAL_LLM_MODEL env var, or 'gpt-4o').
            timeout: Request timeout in seconds (default: 60.0).
            max_retries: Maximum number of retries for transient errors (default: 3).
            enable_circuit_breaker: Whether to use circuit breaker pattern (default: True).
        """
        self.logger = logger
        # Load model from environment variable if not provided
        self.model = model or os.getenv('LLM_MODEL', 'gpt-5')
        # Load visual model for image processing
        self.visual_model = visual_model or os.getenv('VISUAL_LLM_MODEL', 'gpt-4.1')
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
        # Initialize vision client with custom URL if gpt-4.1 is used and OPENAI_URL is set
        self.vision_client = None
        if self.visual_model == 'gpt-4.1':
            openai_url = os.getenv('OPENAI_URL')
            if openai_url:
                self.vision_client = openai.OpenAI(
                    api_key=api_key,
                    base_url=openai_url,
                    timeout=openai.Timeout(timeout),
                )
                self.logger.info(
                    f'Initialized vision client with custom URL: {openai_url}'
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
            f'LLMService initialized with model: {self.model}, visual_model: {self.visual_model}, timeout: {timeout}s, max_retries: {max_retries}'
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

        # Validate messages format - some custom API endpoints have strict requirements
        if not isinstance(messages, list):
            raise ValueError(f'messages must be a list, got {type(messages)}')

        if len(messages) > 10:
            self.logger.warning(
                f'Large messages array detected ({len(messages)} messages). '
                f'Some API endpoints may have strict format requirements. '
                f'First few message roles: {[msg.get("role") for msg in messages[:5]]}'
            )

        # Validate each message has required fields
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f'Message {i} must be a dict, got {type(msg)}')
            if 'role' not in msg:
                raise ValueError(f'Message {i} missing required "role" field')
            if 'content' not in msg:
                raise ValueError(f'Message {i} missing required "content" field')

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

        # Check if model is openai/gpt-oss-120b - this model doesn't support response_format
        # If so, add JSON instruction to prompts instead
        if response_format and self.model == 'openai/gpt-oss-120b':
            self.logger.debug(
                f'Model {self.model} does not support response_format, adding JSON instruction to prompts'
            )
            if response_format.get('type') == 'json_object':
                messages_copy = []
                for i, msg in enumerate(messages):
                    msg_copy = msg.copy()
                    # Add JSON instruction to the last user message
                    if msg_copy['role'] == 'user' and i == len(messages) - 1:
                        msg_copy['content'] += (
                            '\n\nIMPORTANT: Return your response as valid JSON only, without any markdown formatting or additional text.'
                        )
                    messages_copy.append(msg_copy)
                messages = messages_copy
            # Skip response_format for this model
            response_format = None

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

            # Some custom API endpoints have strict message format requirements
            # If we detect an unusually large messages array, log it for debugging
            messages_to_send = call_kwargs.get('messages', [])
            if len(messages_to_send) > 10:
                self.logger.error(
                    f'CRITICAL: Attempting to send {len(messages_to_send)} messages to API. '
                    f'This may cause API format errors. '
                    f'Message roles: {[msg.get("role") for msg in messages_to_send[:10]]}...'
                )
                # Log the first message structure for debugging
                if messages_to_send:
                    first_msg = messages_to_send[0]
                    self.logger.error(
                        f'First message structure: role={first_msg.get("role")}, '
                        f'content_type={type(first_msg.get("content"))}, '
                        f'content_length={len(str(first_msg.get("content", "")))}'
                    )

                # Try to fix: if we have too many messages, try to combine them
                # or take only the system and last user message
                if len(messages_to_send) > 2:
                    self.logger.warning(
                        f'Too many messages ({len(messages_to_send)}). '
                        f'Attempting to reduce to 2 messages (system + user)'
                    )
                    # Find system message if exists
                    system_msg = None
                    user_msgs = []
                    for msg in messages_to_send:
                        if msg.get('role') == 'system':
                            system_msg = msg
                        elif msg.get('role') == 'user':
                            user_msgs.append(msg)

                    # Combine all user messages into one
                    if user_msgs:
                        combined_user_content = '\n\n'.join(
                            str(msg.get('content', '')) for msg in user_msgs
                        )
                        combined_user_msg = {
                            'role': 'user',
                            'content': combined_user_content,
                        }

                        # Create new messages array with max 2 messages
                        fixed_messages = []
                        if system_msg:
                            fixed_messages.append(system_msg)
                        fixed_messages.append(combined_user_msg)

                        self.logger.info(
                            f'Reduced messages from {len(messages_to_send)} to {len(fixed_messages)}'
                        )
                        call_kwargs['messages'] = fixed_messages
                        messages_to_send = fixed_messages

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

    def call_with_images(
        self,
        messages: List[
            Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]
        ],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Make a vision API call with image inputs using the visual model.

        Args:
            messages: List of message dictionaries. Each message can contain:
                - 'role': 'user' or 'system'
                - 'content': List of content items (text strings or image dicts with 'type' and 'image_url')
                - OR 'content': string (for text-only messages)
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in response.
            response_format: Optional response format (e.g., {"type": "json_object"}).
            model: Override visual model (default: uses self.visual_model).

        Returns:
            Response content as string.

        Raises:
            Exception: If the call fails after all retries, with improved error message.
        """
        vision_model = model or self.visual_model
        self.logger.info(
            f'Calling vision LLM ({vision_model}) with {len(messages)} messages'
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
            'model': vision_model,
            'messages': messages,
            'temperature': temperature,
        }
        if max_tokens:
            kwargs['max_tokens'] = max_tokens
        if response_format:
            kwargs['response_format'] = response_format

        def _make_vision_api_call() -> str:
            """Internal function to make the vision API call."""
            # Use vision_client if model is gpt-4.1 and OPENAI_URL is set
            client_to_use = self.client
            if vision_model == 'gpt-4.1':
                openai_url = os.getenv('OPENAI_URL')
                if openai_url:
                    # Use existing vision_client if available, or create one
                    if self.vision_client:
                        client_to_use = self.vision_client
                    else:
                        # Create vision client on-the-fly if needed
                        api_key = os.getenv('OPENAI_API_KEY')
                        client_to_use = openai.OpenAI(
                            api_key=api_key,
                            base_url=openai_url,
                            timeout=openai.Timeout(self.timeout),
                        )
                        self.vision_client = client_to_use
                        self.logger.info(
                            f'Created vision client with custom URL: {openai_url}'
                        )
            response = client_to_use.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            self.logger.debug(
                f'Vision LLM response received: {len(content)} characters'
            )

            # Log output
            self.logger.info('=' * 80)
            self.logger.info('VISION LLM CALL OUTPUT:')
            self.logger.info(f'Response:\n{content}')
            self.logger.info('=' * 80)

            # Record success in circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.record_success()

            return content

        try:
            return retry_with_backoff(
                _make_vision_api_call,
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
                f'Vision LLM API call failed ({error_category.value}/{error_type.value}): {user_message}',
                exc_info=True,
            )
            raise Exception(user_message) from e

    @staticmethod
    def encode_image_to_base64(image_data: bytes) -> str:
        """
        Encode image bytes to base64 string for vision API.

        Args:
            image_data: Image data as bytes.

        Returns:
            Base64-encoded string.
        """
        return base64.b64encode(image_data).decode('utf-8')

    @staticmethod
    def create_image_content(
        image_data: bytes, image_format: str = 'auto'
    ) -> Dict[str, Union[str, Dict[str, str]]]:
        """
        Create image content dict for vision API messages.

        Args:
            image_data: Image data as bytes.
            image_format: Image format hint ('auto', 'png', 'jpeg', etc.). Default: 'auto'.

        Returns:
            Content dict with 'type' and 'image_url' keys.
        """
        base64_image = LLMService.encode_image_to_base64(image_data)

        # Auto-detect format from image data if not specified
        if image_format == 'auto':
            # Check PNG signature (starts with PNG magic bytes)
            if image_data.startswith(b'\x89PNG\r\n\x1a\n'):
                mime_type = 'image/png'
            # Check JPEG signature (starts with FF D8)
            elif image_data.startswith(b'\xff\xd8'):
                mime_type = 'image/jpeg'
            # Check GIF signature
            elif image_data.startswith(b'GIF'):
                mime_type = 'image/gif'
            # Check WebP signature
            elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]:
                mime_type = 'image/webp'
            else:
                # Default to PNG for unknown formats
                mime_type = 'image/png'
        else:
            # Use specified format
            format_map = {
                'png': 'image/png',
                'jpeg': 'image/jpeg',
                'jpg': 'image/jpeg',
                'gif': 'image/gif',
                'webp': 'image/webp',
            }
            mime_type = format_map.get(image_format.lower(), 'image/png')

        return {
            'type': 'image_url',
            'image_url': {
                'url': f'data:{mime_type};base64,{base64_image}',
            },
        }

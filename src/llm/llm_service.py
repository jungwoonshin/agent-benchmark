"""LLM service for OpenAI API integration."""

import base64
import logging
import os
from typing import Dict, List, Optional, Union

import openai
from dotenv import load_dotenv

from ..utils import classify_error, retry_with_backoff

# Load environment variables
load_dotenv()


class LLMService:
    """Service for interacting with OpenAI API."""

    MODEL_CONFIG = {
        'gpt-4.1': {
            'base_url_env': 'OPENAI_URL',
            'is_reasoning_model': True,
            'use_max_completion_tokens': True,
            'supports_response_format': False,
        },
        'openai/gpt-oss-120b': {
            'base_url_env': 'OPENAI_BASE_URL',
            'supports_response_format': False,
            'is_reasoning_model': False,
            'use_max_completion_tokens': True,  # Use max_completion_tokens instead of max_tokens
        },
    }

    # Class-level cache for models that don't support response_format
    _response_format_unsupported_models: set = set()

    def __init__(
        self,
        logger: logging.Logger,
        model: str = None,
        visual_model: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize LLM service.

        Args:
            logger: Logger instance for logging.
            model: OpenAI model to use (default: from LLM_MODEL env var, or 'openai/gpt-oss-120b').
            visual_model: Vision model for image processing (default: from VISUAL_LLM_MODEL env var, or 'gpt-4o').
            timeout: Request timeout in seconds (default: 60.0).
            max_retries: Maximum number of retries for transient errors (default: 3).
        """
        self.logger = logger
        self.model = model or os.getenv('LLM_MODEL', 'openai/gpt-oss-120b')
        self.visual_model = visual_model or os.getenv('VISUAL_LLM_MODEL', 'gpt-4.1')
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                'OPENAI_API_KEY not found in environment variables. '
                'Please set it in .env file.'
            )

        self.clients = {}
        self._initialize_clients()
        self.DEFAULT_MAX_TOKENS = 8192

        self.logger.info(
            f'LLMService initialized with model: {self.model}, visual_model: {self.visual_model}, timeout: {timeout}s, max_retries: {max_retries}'
        )

    def _initialize_clients(self):
        """Initialize OpenAI clients based on model configurations."""
        self.clients['default'] = openai.OpenAI(
            api_key=self.api_key,
            timeout=openai.Timeout(self.timeout),
        )

        all_models = {self.model, self.visual_model}
        for model_name in all_models:
            if model_name in self.MODEL_CONFIG:
                config = self.MODEL_CONFIG[model_name]
                base_url_env = config.get('base_url_env')
                if base_url_env:
                    base_url = os.getenv(base_url_env)
                    if base_url:
                        self.clients[model_name] = openai.OpenAI(
                            api_key=self.api_key,
                            base_url=base_url,
                            timeout=openai.Timeout(self.timeout),
                        )
                        self.logger.info(
                            f'Initialized client for {model_name} with custom URL from {base_url_env}'
                        )

    def _get_client_for_model(self, model_name: str) -> openai.OpenAI:
        """Get the appropriate OpenAI client for a given model."""
        return self.clients.get(model_name, self.clients['default'])

    def _validate_messages(self, messages: List[Dict[str, str]]):
        """Validate the format of messages."""
        if not isinstance(messages, list):
            raise ValueError(f'messages must be a list, got {type(messages)}')
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f'Message {i} must be a dict, got {type(msg)}')
            if 'role' not in msg:
                raise ValueError(f'Message {i} missing required "role" field')
            if 'content' not in msg:
                raise ValueError(f'Message {i} missing required "content" field')

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
        self._validate_messages(messages)

        # Log input prompts if not skipped
        if not _skip_logging:
            self._log_call_inputs(messages, temperature, max_tokens, response_format)

        # Consolidate messages if there are too many
        if len(messages) > 10:
            raise Exception('Too many messages. Please consolidate messages.')

        try:
            content = self._call_with_response_format_fallback(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

            # Log outputs if not skipped
            if not _skip_logging:
                self._log_call_outputs(content)

            return content
        except Exception as e:
            # Avoid re-wrapping exception if it's already classified
            if 'LLM API call failed' in str(e):
                raise e

            error_type, error_category, user_message = classify_error(e)
            self.logger.error(
                f'LLM API call failed ({error_category.value}/{error_type.value}): {user_message}',
                exc_info=True,
            )
            raise Exception(f'LLM API call failed: {user_message}') from e

    def _call_with_response_format_fallback(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, str]],
    ) -> str:
        """Attempt API call with response_format and fall back if not supported."""

        model_supports_format = self.MODEL_CONFIG.get(self.model, {}).get(
            'supports_response_format', True
        )

        # Handle models that are known not to support response_format
        if response_format and (
            not model_supports_format
            or self.model in self._response_format_unsupported_models
        ):
            self.logger.warning(
                f'Model {self.model} does not support response_format, skipping.'
            )
            messages = self._add_json_instruction(messages)
            response_format = None

        if response_format:
            try:
                return self._make_api_call(
                    messages, temperature, max_tokens, response_format
                )
            except Exception as e:
                error_msg = str(e).lower()
                if 'response_format' in error_msg or 'not supported' in error_msg:
                    self.logger.warning(
                        f'Model {self.model} does not support response_format, retrying without it.'
                    )
                    self._response_format_unsupported_models.add(self.model)
                    messages = self._add_json_instruction(messages)
                else:
                    raise e

        return self._make_api_call(messages, temperature, max_tokens, None)

    def _make_api_call(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, str]],
    ) -> str:
        """Make a single attempt to call the OpenAI API."""
        self.logger.info(f'Making API call with model: {self.model}')
        model_config = self.MODEL_CONFIG.get(self.model, {})
        is_reasoning_model = model_config.get('is_reasoning_model', False)

        # For reasoning models, ensure only user messages (no system messages)
        if is_reasoning_model and len(messages) > 1:
            # Consolidate all messages into a single user message
            combined_content = []
            for msg in messages:
                if msg.get('role') == 'system':
                    combined_content.append(f'Instructions: {msg.get("content", "")}')
                else:
                    combined_content.append(msg.get('content', ''))
            messages = [{'role': 'user', 'content': '\n\n'.join(combined_content)}]
            self.logger.info(
                f'Consolidated {len(messages)} messages for reasoning model'
            )

        kwargs = {
            'model': self.model,
            'messages': messages,
        }

        # Add temperature only if model supports it
        supports_temperature = model_config.get('supports_temperature', True)
        if supports_temperature:
            kwargs['temperature'] = temperature
        else:
            self.logger.debug(
                f'Model {self.model} does not support temperature parameter'
            )

        if response_format:
            kwargs['response_format'] = response_format

        def api_call_fn():
            client = self._get_client_for_model(self.model)
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        return retry_with_backoff(
            api_call_fn,
            max_retries=self.max_retries,
            base_delay=1.0,
            max_delay=60.0,
            logger=self.logger,
        )

    def _add_json_instruction(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Add instruction to the last user message to return JSON."""
        messages_copy = [msg.copy() for msg in messages]
        for i in reversed(range(len(messages_copy))):
            if messages_copy[i]['role'] == 'user':
                messages_copy[i]['content'] += (
                    '\n\nIMPORTANT: Return your response as valid JSON only, '
                    'without any markdown formatting or additional text.'
                )
                break
        return messages_copy

    def _log_call_inputs(self, messages, temperature, max_tokens, response_format):
        """Log the inputs to an LLM call."""
        self.logger.info('=' * 80)
        self.logger.info('LLM CALL INPUT:')
        system_prompts = [
            msg['content'] for msg in messages if msg.get('role') == 'system'
        ]
        user_prompts = [msg['content'] for msg in messages if msg.get('role') == 'user']
        if system_prompts:
            for i, prompt in enumerate(system_prompts):
                self.logger.info(f'System Prompt {i + 1}:\n{prompt}')
        if user_prompts:
            for i, prompt in enumerate(user_prompts):
                self.logger.info(f'User Prompt {i + 1}:\n{prompt}')
        self.logger.info(
            f'Parameters: temperature={temperature}, max_tokens={max_tokens}, response_format={response_format}'
        )

    def _log_call_outputs(self, content: str):
        """Log the outputs from an LLM call."""
        self.logger.info('LLM CALL OUTPUT:')
        self.logger.info(f'Response:\n{content}')
        self.logger.info('=' * 80)

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
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
        self._log_call_inputs(messages, temperature, max_tokens, response_format)

        response = self.call(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            _skip_logging=True,  # Skip logging in call() since we handle it here
        )

        # Log output
        self._log_call_outputs(response)
        return response

    def _consolidate_vision_messages(self, messages):
        """Consolidate vision messages to a system and a user message."""
        if len(messages) > 2:
            self.logger.warning(
                f'Too many vision messages ({len(messages)}). Consolidating to 2 messages.'
            )
            system_msg = next(
                (msg for msg in messages if msg['role'] == 'system'), None
            )
            user_msgs = [msg for msg in messages if msg['role'] == 'user']

            if user_msgs:
                combined_content = []
                for msg in user_msgs:
                    content = msg.get('content', [])
                    if isinstance(content, list):
                        combined_content.extend(content)
                    else:
                        combined_content.append({'type': 'text', 'text': str(content)})

                new_messages = []
                if system_msg:
                    new_messages.append(system_msg)
                new_messages.append({'role': 'user', 'content': combined_content})
                return new_messages
        return messages

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
        
        # Count images in the request
        image_count = 0
        text_length = 0
        for msg in messages:
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if isinstance(item, dict):
                        if item.get('type') == 'image_url':
                            image_count += 1
                        elif item.get('type') == 'text':
                            text_length += len(item.get('text', ''))
            elif isinstance(msg.get('content'), str):
                text_length += len(msg['content'])
        
        self.logger.info(
            f'[Visual LLM] Calling vision model: {vision_model}, '
            f'messages: {len(messages)}, images: {image_count}, '
            f'text length: {text_length} chars, temperature: {temperature}, '
            f'max_tokens: {max_tokens}'
        )

        messages = self._consolidate_vision_messages(messages)

        try:
            response = self._make_vision_api_call(
                model=vision_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            self.logger.info(
                f'[Visual LLM] API call successful: {len(response)} chars returned'
            )
            return response
        except Exception as e:
            # Avoid re-wrapping exception if it's already classified
            if 'LLM API call failed' in str(e):
                raise e

            error_type, error_category, user_message = classify_error(e)
            self.logger.error(
                f'[Visual LLM] API call failed ({error_category.value}/{error_type.value}): {user_message}',
                exc_info=True,
            )
            raise Exception(f'LLM API call failed: {user_message}') from e

    def _make_vision_api_call(
        self,
        model: str,
        messages: List[Dict[str, any]],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, str]],
    ):
        kwargs = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
        }
        # Validate max_tokens: must be at least 1 if provided
        if max_tokens is not None:
            if max_tokens < 1:
                self.logger.warning(
                    f'[Visual LLM] Invalid max_tokens value for vision API: {max_tokens}. Must be at least 1. '
                    f'Setting max_tokens to None (will use model default).'
                )
                max_tokens = None

        if max_tokens and max_tokens >= 1:
            kwargs['max_tokens'] = max_tokens
        if response_format:
            kwargs['response_format'] = response_format

        self.logger.debug(
            f'[Visual LLM] Making API call with model={model}, '
            f'temperature={temperature}, max_tokens={max_tokens}, '
            f'response_format={response_format}'
        )

        def api_call_fn():
            client_to_use = self._get_client_for_model(model)
            self.logger.debug(f'[Visual LLM] Executing API request to {model}')
            response = client_to_use.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        return retry_with_backoff(
            api_call_fn,
            max_retries=self.max_retries,
            base_delay=1.0,
            max_delay=60.0,
            logger=self.logger,
        )

    @staticmethod
    def encode_image_to_base64(image_data: bytes) -> str:
        """Encode image bytes to base64 string."""
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

        mime_type = 'image/png'  # Default MIME type
        if image_format == 'auto':
            # Simple magic byte checks for common formats
            if image_data.startswith(b'\x89PNG\r\n\x1a\n'):
                mime_type = 'image/png'
            elif image_data.startswith(b'\xff\xd8'):
                mime_type = 'image/jpeg'
            elif image_data.startswith(b'GIF87a') or image_data.startswith(b'GIF89a'):
                mime_type = 'image/gif'
            elif image_data.startswith(b'RIFF') and image_data[8:12] == b'WEBP':
                mime_type = 'image/webp'
        else:
            mime_type = f'image/{image_format.lower()}'

        return {
            'type': 'image_url',
            'image_url': {
                'url': f'data:{mime_type};base64,{base64_image}',
            },
        }

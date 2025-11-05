"""Test script for custom LLM API endpoint."""

import json
import os

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
API_URL = 'http://114.110.129.108:3163/v1/chat/completions'
MODEL = 'openai/gpt-oss-120b'
API_KEY = os.getenv('OPENAI_API_KEY', 'test-key')


def test_with_response_format():
    """Test if the API accepts response_format parameter."""
    print('=' * 80)
    print("Testing LLM API with response_format={'type': 'json_object'}")
    print('=' * 80)
    print(f'URL: {API_URL}')
    print(f'Model: {MODEL}')
    print(f'API Key: {API_KEY[:20]}...' if len(API_KEY) > 20 else f'API Key: {API_KEY}')
    print()

    # Initialize OpenAI client with custom base URL
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url=API_URL.replace(
            '/chat/completions', ''
        ),  # Remove /chat/completions from base_url
        timeout=60.0,
    )

    # Test message - try with just user message first (API might expect 2 messages total)
    messages = [
        {
            'role': 'user',
            'content': "Return a JSON object with keys: 'test' and 'status', where test='response_format_support' and status='success' or 'failed'.",
        }
    ]

    print("Making API call with response_format={'type': 'json_object'}...")
    print()

    try:
        # Try with response_format
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            response_format={'type': 'json_object'},
        )

        content = response.choices[0].message.content
        print('✓ API call succeeded with response_format!')
        print()
        print('Response:')
        print(content)
        print()

        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            print('✓ Response is valid JSON!')
            print(f'Parsed JSON: {json.dumps(parsed, indent=2)}')
            return True
        except json.JSONDecodeError as e:
            print(f'✗ Response is NOT valid JSON: {e}')
            print(f'Response content: {content[:500]}')
            return False

    except Exception as e:
        error_msg = str(e).lower()
        print(f'✗ API call failed: {e}')
        print()

        # Check if it's a response_format error
        if 'response_format' in error_msg or 'not supported' in error_msg:
            print('→ This model does NOT support response_format parameter')
            print()
            print('Testing without response_format...')
            try:
                # Retry without response_format
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.3,
                )
                content = response.choices[0].message.content
                print('✓ API call succeeded without response_format!')
                print()
                print('Response:')
                print(content)
                print()

                # Try to extract JSON
                import re

                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        print('✓ Found valid JSON in response!')
                        print(f'Parsed JSON: {json.dumps(parsed, indent=2)}')
                        return False  # response_format not supported, but JSON found
                    except json.JSONDecodeError:
                        print('✗ Could not parse JSON from response')
                        return False
                else:
                    print('✗ No JSON found in response')
                    return False
            except Exception as e2:
                print(f'✗ API call failed even without response_format: {e2}')
                return False
        else:
            print(f'→ Error type: {type(e).__name__}')
            print(f'→ Error details: {str(e)}')
            return False


def test_without_response_format():
    """Test the API without response_format parameter."""
    print('=' * 80)
    print('Testing LLM API without response_format (for comparison)')
    print('=' * 80)
    print()

    client = openai.OpenAI(
        api_key=API_KEY,
        base_url=API_URL.replace('/chat/completions', ''),
        timeout=60.0,
    )

    messages = [
        {
            'role': 'user',
            'content': "Return a JSON object with keys: 'test' and 'status', where test='no_response_format' and status='success' or 'failed'.",
        }
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
        )

        content = response.choices[0].message.content
        print('✓ API call succeeded!')
        print()
        print('Response:')
        print(content)
        print()

        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            print('✓ Response is valid JSON!')
            print(f'Parsed JSON: {json.dumps(parsed, indent=2)}')
            return True
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or text
            import re

            json_match = re.search(
                r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL
            )
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    print('✓ Found valid JSON in markdown code block!')
                    print(f'Parsed JSON: {json.dumps(parsed, indent=2)}')
                    return True
                except json.JSONDecodeError:
                    pass

            # Try simple regex
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    print('✓ Found valid JSON in response!')
                    print(f'Parsed JSON: {json.dumps(parsed, indent=2)}')
                    return True
                except json.JSONDecodeError as e:
                    print(f'✗ Response contains JSON-like text but is not valid: {e}')
                    return False

            print('✗ Response is NOT valid JSON')
            return False

    except Exception as e:
        print(f'✗ API call failed: {e}')
        return False


if __name__ == '__main__':
    print('\n')
    print('LLM API Test Script')
    print('=' * 80)
    print()

    # Test with response_format first
    supports_response_format = test_with_response_format()

    print()
    print()

    # Test without response_format for comparison
    works_without_format = test_without_response_format()

    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'response_format support: {"✓ YES" if supports_response_format else "✗ NO"}')
    print(
        f'Works without response_format: {"✓ YES" if works_without_format else "✗ NO"}'
    )
    print()

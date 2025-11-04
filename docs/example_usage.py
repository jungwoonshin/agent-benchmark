"""Example usage of the Agent system with LLM-based decision making."""

import logging
import os
import sys

from dotenv import load_dotenv

from src.core import Agent, Attachment, ToolBelt

# Load environment variables
load_dotenv()


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/log.txt', mode='a'),
        ],
    )
    return logging.getLogger('agent_system')


def main():
    """Example usage of the Agent system."""
    # Setup logging
    logger = setup_logging()

    # Create ToolBelt
    tool_belt = ToolBelt()

    # Create Agent with LLM (model from LLM_MODEL env var or defaults to gpt-5)
    # Make sure OPENAI_API_KEY is set in .env file
    try:
        llm_model = os.getenv('LLM_MODEL', 'gpt-5')
        agent = Agent(tool_belt=tool_belt, logger=logger, llm_model=llm_model)
    except ValueError as e:
        print(f'Error: {e}')
        print('\nPlease create a .env file with your OPENAI_API_KEY:')
        print('OPENAI_API_KEY=your_key_here')
        return

    # Example 1: Simple computational problem
    print('=' * 80)
    print('Example 1: Computational Problem')
    print('=' * 80)
    problem1 = 'What is 2 + 2? Show your work.'
    try:
        final_answer, monologue = agent.solve(problem1)
        print(f'\nFinal Answer: {final_answer}')
        print(f'\nReasoning Monologue:\n{monologue}')
    except Exception as e:
        print(f'Error solving problem: {e}')
        logger.error(f'Error: {e}', exc_info=True)

    # Example 2: Problem with attachment
    print('\n' + '=' * 80)
    print('Example 2: Problem with Attachment')
    print('=' * 80)
    attachment = Attachment(filename='e14448e9.jpg', data=b'fake_image_data')
    problem2 = 'What animal is in this image? Describe it.'
    try:
        final_answer2, monologue2 = agent.solve(problem2, attachments=[attachment])
        print(f'\nFinal Answer: {final_answer2}')
        print(f'\nReasoning Monologue:\n{monologue2}')
    except Exception as e:
        print(f'Error solving problem: {e}')
        logger.error(f'Error: {e}', exc_info=True)

    print('\n' + '=' * 80)
    print('Check logs/log.txt for detailed logging output')
    print('=' * 80)


if __name__ == '__main__':
    main()

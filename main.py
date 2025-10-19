import asyncio
import logging
import os
from dotenv import load_dotenv

from agent import ProgressiveDisclosureAgent

# Load environment variables from a .env file (if present) into os.environ
load_dotenv()

# Configure OpenRouter via OpenAI-compatible settings
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Please export your OpenRouter API key to the environment."
    )

logger = logging.getLogger(__name__)
# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

async def main():
    agent = ProgressiveDisclosureAgent(OPENROUTER_API_KEY, "./skills")
    logger.info(f"Built agent prompt:")
    logger.info(agent.prompt)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
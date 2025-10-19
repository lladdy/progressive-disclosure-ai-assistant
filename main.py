import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from skill_loader import locate_skills, get_skill_content

# Load environment variables from a .env file (if present) into os.environ
load_dotenv()

# Configure OpenRouter via OpenAI-compatible settings
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Please export your OpenRouter API key to the environment."
    )

# Choose any OpenRouter-supported model; adjust as needed
# See https://openrouter.ai/models for options
model = OpenAIChatModel(
    "anthropic/claude-3.5-sonnet",
    provider=OpenAIProvider(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"),
)

base_prompt = """
You are a helpful assistant with access to a list of skills. Try to answer the user's messages and use the skills when
they are relevant to the user's request.
You can access more information about each skill by calling the list_skill_files and load_file_content tools.
Always prioritize using the SKILL.md file for each skill first, as it contains the main documentation.

You have the following skills available:

"""

# Build the rest of the prompt using locate_skills
skills = locate_skills("./skills")
skills_list = "\n".join([f"- Name: {skill.name} | Description: {skill.description}" for skill in skills])
full_prompt = f"{base_prompt}{skills_list}\n"

agent = Agent(
    instructions=full_prompt,
    model=model,
)


@agent.tool
def list_skill_files(ctx: RunContext[int], skill_name: str) -> str:
    """
    List all files in the folder for the skill with the given name.

    Args:
        ctx: RunContext (unused here, but required by the tool interface)
        skill_name: The human-readable name field from the skill's frontmatter

    Returns:
        A newline-separated list of file paths contained in the skill's folder,
        or a helpful message if the skill is not found or fails to load.
    """
    # Prefer exact match on name, then fallback to case-insensitive match
    target = None
    for s in skills:
        if s.name == skill_name:
            target = s
            break
    if target is None:
        lowered = skill_name.lower().strip()
        for s in skills:
            if s.name.lower().strip() == lowered:
                target = s
                break

    if target is None:
        available = ", ".join(sorted({s.name for s in skills}))
        return f"Skill '{skill_name}' not found. Available skills: {available}"

    try:
        content = get_skill_content(target.skill_path)
        # Include SKILL.md explicitly along with other files discovered recursively
        all_files = [str(target.skill_path / "SKILL.md")] + content.other_files
        return "\n".join(all_files)
    except Exception as e:
        return f"Failed to list files for '{skill_name}': {e}"


@agent.tool
def load_file_content(ctx: RunContext[int], file_path: str) -> str:
    """
    Load and return the content of the file at the given path.

    Args:
        ctx: RunContext (unused by this tool)
        file_path: Absolute or relative path to a file

    Returns:
        The textual content of the file, or an error message if the file
        doesn't exist, isn't a file, is too large, or can't be decoded.
    """
    try:
        p = Path(file_path)
        if not p.exists():
            return f"File not found: {file_path}"
        if not p.is_file():
            return f"Path is not a file: {file_path}"
        # Guard against accidentally loading very large files
        try:
            size = p.stat().st_size
            if size > 5 * 1024 * 1024:  # 5 MB limit
                return f"File is too large to display ({size} bytes): {file_path}"
        except Exception:
            # If stat fails, continue and try reading
            pass
        # Read as text; replace undecodable bytes to avoid exceptions
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Failed to read file '{file_path}': {e}"


async def main():
    await agent.to_cli()


if __name__ == "__main__":
    asyncio.run(main())
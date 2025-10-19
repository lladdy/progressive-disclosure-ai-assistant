
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from skill_loader import locate_skills, get_skill_content

BASE_PROMPT = """
You are a concise, helpful assistant with access to a set of skills. Use these rules when answering:

- Goal: Answer the user's question as directly and helpfully as possible. Use skills only when the user's request depends on skill-specific content or when the user explicitly asks you to use a skill.
- Tools available:
  - list_skill_files(skill_name): returns a newline-separated list of file paths in the given skill's folder.
  - load_file_content(file_path): returns the textual contents of the specified file.
- Always prefer the SKILL.md file for a skill as the canonical documentation. Use list_skill_files to discover SKILL.md and other files, then use load_file_content to read any file you need.
- Do not invent file names or claim to have read files you have not loaded. If you need information from a skill, explicitly call the appropriate tools.
- Before calling any tool, write a one-line "Tool plan:" that states which tool(s) you will call and why.
- After using tools, produce a concise final answer. If you used skill files, include a short "Sources:" section listing the file paths you consulted and 1–2 short quoted snippets (if helpful).
- If the user's request is ambiguous or missing critical details, ask the user for clarification instead of making assumptions.
- If a skill is not relevant, answer directly without calling tools.
- Keep responses brief, avoid unnecessary detail, and avoid unsafe or disallowed instructions.

You have the following skills available:
"""

class ProgressiveDisclosureAgent:
    def __init__(self, openrouter_api_key, skills_path):
        # Choose any OpenRouter-supported model; adjust as needed
        # See https://openrouter.ai/models for options
        model = OpenAIChatModel(
            "anthropic/claude-3.5-sonnet",
            provider=OpenAIProvider(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"),
        )

        # Build the rest of the prompt using locate_skills
        skills = locate_skills(skills_path)
        skills_list = "\n".join([f"- Name: {skill.name} | Description: {skill.description}" for skill in skills])
        full_prompt = f"{BASE_PROMPT}{skills_list}\n"

        self._prompt = full_prompt

        server = MCPServerStdio('uvx', args=['mcp-run-python@latest', 'stdio'], timeout=10)

        agent = Agent(
            instructions=full_prompt,
            model=model,
            toolsets=[server],
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
                except Exception as e:
                    # If stat fails, continue and try reading
                    return f"Failed to call stat() on file '{file_path}': {e}"
                # Read as text; replace undecodable bytes to avoid exceptions
                return p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return f"Failed to read file '{file_path}': {e}"

        self._agent = agent

    @property
    def prompt(self):
        return self._prompt

    async def run(self):
        async with self._agent:
            await self._agent.to_cli()

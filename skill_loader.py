"""
Skill Loader Utility

This module provides functions to discover and load Claude Skills from a directory structure.
Skills follow the format defined in claude-cookbooks/skills with YAML frontmatter metadata.

Architecture:
- Skills are directories containing a SKILL.md file with YAML frontmatter
- Frontmatter must include 'name' and 'description' fields
- Skills may include resources and REFERENCE.md files
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import re


class SkillMetadata:
    """Represents the metadata/header of a skill."""

    def __init__(self, skill_path: Path, name: str, description: str):
        self.skill_path = skill_path
        self.skill_id = skill_path.name
        self.name = name
        self.description = description

    def to_dict(self) -> Dict[str, str]:
        """Convert metadata to dictionary format."""
        return {
            'skill_id': self.skill_id,
            'skill_path': str(self.skill_path),
            'name': self.name,
            'description': self.description
        }

    def __repr__(self) -> str:
        return f"SkillMetadata(skill_id='{self.skill_id}', name='{self.name}')"


class SkillContent:
    """Represents the full content of a skill."""

    def __init__(
        self,
        metadata: SkillMetadata,
        skill_md_content: str,
        other_files: List[str],
    ):
        self.metadata = metadata
        self.skill_md_content = skill_md_content
        self.other_files = other_files

    def to_dict(self) -> Dict:
        """Convert skill content to dictionary format."""
        return {
            'metadata': self.metadata.to_dict(),
            'skill_md': self.skill_md_content,
            'other_files': self.other_files,
        }


def extract_yaml_frontmatter(content: str) -> tuple[Optional[Dict], str]:
    """
    Extract YAML frontmatter from markdown content.

    Args:
        content: The full markdown content including frontmatter

    Returns:
        Tuple of (frontmatter_dict, remaining_content)
        Returns (None, content) if no valid frontmatter found
    """
    # Match YAML frontmatter pattern: --- at start, content, --- at end
    pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return None, content

    try:
        frontmatter_str = match.group(1)
        remaining_content = match.group(2)
        frontmatter = yaml.safe_load(frontmatter_str)
        return frontmatter, remaining_content
    except yaml.YAMLError:
        return None, content


def is_valid_skill_directory(path: Path) -> bool:
    """
    Check if a directory contains a valid skill (has SKILL.md with required metadata).

    Args:
        path: Path to the directory to check

    Returns:
        True if the directory contains a valid skill
    """
    if not path.is_dir():
        return False

    skill_md = path / "SKILL.md"
    if not skill_md.exists():
        return False

    try:
        content = skill_md.read_text(encoding='utf-8')
        frontmatter, _ = extract_yaml_frontmatter(content)

        if frontmatter is None:
            return False

        # Check for required fields
        return 'name' in frontmatter and 'description' in frontmatter
    except Exception:
        return False


def locate_skills(root_folder: str | Path) -> List[SkillMetadata]:
    """
    Locate all valid skills in a given folder and return their metadata headers.

    This function scans the root folder for subdirectories containing SKILL.md files
    with valid YAML frontmatter (name and description fields).

    Args:
        root_folder: Path to the folder to search for skills

    Returns:
        List of SkillMetadata objects containing skill headers

    Example:
        >>> skills = locate_skills("./custom_skills")
        >>> for skill in skills:
        ...     print(f"{skill.name}: {skill.description}")
    """
    root_path = Path(root_folder)

    if not root_path.exists():
        raise FileNotFoundError(f"Root folder not found: {root_folder}")

    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_folder}")

    skills = []

    # Search for all directories in root folder
    for item in root_path.iterdir():
        if not item.is_dir():
            continue

        if not is_valid_skill_directory(item):
            continue

        # Read and parse SKILL.md
        skill_md = item / "SKILL.md"
        try:
            content = skill_md.read_text(encoding='utf-8')
            frontmatter, _ = extract_yaml_frontmatter(content)

            if frontmatter:
                metadata = SkillMetadata(
                    skill_path=item,
                    name=frontmatter['name'],
                    description=frontmatter['description']
                )
                skills.append(metadata)
        except Exception as e:
            # Skip skills with read errors
            print(f"Warning: Could not load skill from {item}: {e}")
            continue

    return skills


def get_skill_content(skill_path: str | Path) -> SkillContent:
    """
    Retrieve the complete content of a skill for LLM consumption.

    This function loads the full skill information including:
    - Metadata from YAML frontmatter
    - Full SKILL.md content
    - List of resource files

    Args:
        skill_path: Path to the skill directory

    Returns:
        SkillContent object containing all skill information

    Raises:
        FileNotFoundError: If skill directory or SKILL.md doesn't exist
        ValueError: If SKILL.md doesn't have valid frontmatter

    Example:
        >>> content = get_skill_content("./custom_skills/financial_analyzer")
        >>> print(content.metadata.name)
        >>> print(content.skill_md_content)
    """
    skill_path = Path(skill_path)

    if not skill_path.exists():
        raise FileNotFoundError(f"Skill directory not found: {skill_path}")

    if not skill_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {skill_path}")

    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md not found in {skill_path}")

    # Read SKILL.md
    skill_md_content = skill_md.read_text(encoding='utf-8')
    frontmatter, _ = extract_yaml_frontmatter(skill_md_content)

    if not frontmatter or 'name' not in frontmatter or 'description' not in frontmatter:
        raise ValueError(f"SKILL.md in {skill_path} missing required frontmatter fields (name, description)")

    # Create metadata
    metadata = SkillMetadata(
        skill_path=skill_path,
        name=frontmatter['name'],
        description=frontmatter['description']
    )

    # Collect other files in the skill directory (recursively), excluding SKILL.md
    other_files = sorted(
        str(p)
        for p in skill_path.rglob('*')
        if p.is_file() and p.name != 'SKILL.md'
    )

    return SkillContent(
        metadata=metadata,
        skill_md_content=skill_md_content,
        other_files=other_files,
    )



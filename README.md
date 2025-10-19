Progressive Disclosure AI Assistant
====================================

A small demonstration AI assistant that uses a "skills" pattern (Claude-like skills) to show progressive disclosure: the assistant exposes specialized capabilities as separate, focused skills.

Quick overview
--------------
- Skills live in the `skills/` directory (each skill has its own subfolder and implementation files).
- `main.py` contains the assistant runner and `skill_loader.py` handles loading available skills.

How it demonstrates progressive disclosure
-----------------------------------------
Each skill represents a discrete capability. The assistant can reveal or invoke these capabilities as needed, keeping interactions focused and avoiding overwhelming users with details unless they request them.

Reference
---------
This project follows a skills-style approach similar to Anthropic's description of skills: https://www.anthropic.com/news/skills

Notes
-----
- To run the assistant, try `python main.py` from the repository root (assumes Python 3.10+ and any required dependencies).
- Add new skills under `skills/` following the existing patterns.


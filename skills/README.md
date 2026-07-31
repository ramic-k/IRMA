# Agent skills

`skills/irma/` is an assistant skill for AI coding agents (Claude
Code, Codex, Gemini CLI, and compatible tools): it teaches an agent to
drive IRMA calculations, route users to the right documentation and
example, and answer physics questions from the code with citations.
It is plain markdown and duplicates nothing; the `docs/` tree stays
authoritative.

Installation is opt-in. Either:

- copy or symlink `skills/irma` into your agent's skills directory
  (Claude Code: `~/.claude/skills/irma`, or `.claude/skills/irma`
  inside a project); or
- point your agent's project-instructions file (`AGENTS.md`,
  `CLAUDE.md`, `GEMINI.md`, or equivalent) at
  `skills/irma/SKILL.md` with an instruction to read and follow it.

The YAML header in `SKILL.md` is invocation metadata for agents that
use it; others can ignore it.

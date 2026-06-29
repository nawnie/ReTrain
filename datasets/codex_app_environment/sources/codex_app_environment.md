# Codex App Environment Guide

Codex is a coding agent environment with local files, project instructions, skills, plugins, apps, MCP tools, browser tools, and execution tools. A model trained for this environment must inspect what is available in the active session instead of assuming every tool exists.

## Core Concepts

Skills are local instruction packs stored as `SKILL.md` files. When a user names a skill, or the task clearly matches a skill description, the agent must read that skill before acting. If the skill points to required reference files, read only the relevant references and apply them.

Plugins are bundles that can provide skills, MCP servers, apps, and other capabilities. Plugins are not called directly. Use the skills and tools exposed by the plugin.

MCP tools are callable functions exposed by MCP servers. MCP resources are read-only context objects shared by servers. Prefer MCP resources over web search when the resource is available and relevant. Validate tool schemas before calling MCP tools.

Apps and connectors expose user data or external services through MCP tools. Use `tool_search` to discover deferred tools before asking to install anything. Request a plugin or connector install only when the user explicitly asks for that exact tool and discovery shows a matching candidate.

Local tools include shell commands, `apply_patch`, plan updates, image viewing, and other runtime helpers. Use shell for inspection and verification. Use `apply_patch` for manual file edits. Do not use destructive git or filesystem operations unless the user explicitly asks.

Browser and web tools are for current or source-grounded information, screenshots, and product docs. Current facts, recommendations that affect money or time, laws, prices, schedules, software docs, and OpenAI product behavior need verification from primary or official sources.

## Behavior To Train

Start with project instructions. In this repo, read `HANDOFF.md` before work. Then inspect the live file tree and current app state.

Choose a small skill set. Announce the router when required, read selected skill files, and avoid loading unrelated skills.

Use tools in the right channel and with the right schema. Do not invent tool names, hidden parameters, or MCP server names. If a tool is not exposed, use discovery or state that it is unavailable.

Keep edits scoped. Protect user changes in dirty worktrees. Do not revert unrelated edits.

Verify with targeted commands. For this project, validate dataset JSONL, run Python compile checks, run frontend builds after React changes, and smoke API endpoints when backend behavior changes.

Keep secrets out of training data. Do not write API keys, tokens, private emails, private chat logs, or personal contact details into corpora.

When memory is used, cite memory in the final response. If memory was not refreshed in the current turn and may be stale, say that plainly.

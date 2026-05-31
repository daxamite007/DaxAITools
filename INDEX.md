# Project Index

Entry point for the **Tasklet** repo. **Read this file first, every session.**

Tasklet is a small REST API for creating and assigning tasks across teams.

## How to use this index
1. Always read this file and `CURRENT_STATE.md` at the start of a session.
2. Read the other files **only when the task calls for it** — use the "Read when" column to decide. Don't pull everything in by default.

## Map
| File | Read when you need to… |
|------|------------------------|
| `CURRENT_STATE.md` | Know what's in progress right now, recent changes, blockers, or what to do next. **Read every session.** |
| `architecture.md` | Understand how the system is structured — components, data flow, runtime, and why design choices were made. Read before changing how parts fit together. |
| `conventions.md` | Write or review code: naming, formatting, commit style, testing rules. *(stub — add your own)* |
| `api-reference.md` | Look up endpoint shapes, request/response formats, or status codes. *(stub — add your own)* |
| `decisions/` | Trace why a past decision was made (ADR-style log). *(stub — add your own)* |

## Keeping this current
- When you add a doc, add a row here with a **specific** "Read when" description — that description is what tells the agent whether to open the file.
- Keep this file short. If it grows past ~1 page, the detail belongs in a linked file instead.

> Tip: many harnesses auto-discover a file named `README.md` or `CLAUDE.md`. If yours does, rename this file to match so it loads without being told to.

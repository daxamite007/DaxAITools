# Current State

**Last updated:** 2026-05-30
**Status:** Active development — auth migration in progress.

> This file holds **volatile status only.** Anything stable (how the system is built, conventions, API shapes) belongs in a reference file linked from `INDEX.md`, not here.

## Active work
- Migrating authentication from server-side sessions to JWT. ~60% complete: login and refresh are done; logout and token revocation remain.
- Current sprint codename: **"Marigold."**

## Recent changes
*Most recent first.*
- Added the `/auth/refresh` endpoint.
- Switched the task queue from in-memory to Redis-backed.
- Removed the legacy `/v0` routes.

## Known issues / blockers
- Token revocation has no storage yet — logout currently does nothing server-side.
- Flaky integration test `test_assign_across_teams` fails roughly 1 in 10 runs; cause unknown.

## Next steps
- Implement a revocation list (likely Redis-backed) so logout actually invalidates tokens.
- Backfill tests for the refresh flow.

## Open questions
- Should refresh tokens rotate on every use, or carry a fixed lifetime? Undecided.

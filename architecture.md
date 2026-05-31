# Architecture

> Stable reference. Update only when the actual structure changes — not for day-to-day status (that goes in `CURRENT_STATE.md`).

## Overview
Tasklet is a single REST service backed by Postgres, with Redis for the task queue and (soon) token revocation.

## Components
- **API layer** — handles HTTP, validation, and auth. Stateless.
- **Service layer** — business logic; the only layer permitted to touch the database.
- **Postgres** — source of truth for users, teams, and tasks.
- **Redis** — task queue and ephemeral state.

## Data flow
A request hits the API layer, is validated and authenticated, then passed to the service layer, which reads/writes Postgres and enqueues any async work in Redis before a response is returned. The API layer never queries the database directly.

## Runtime
- The service listens on **port 8420** in every environment.
- One process per container; horizontal scaling is by adding containers behind the load balancer.

## Key decisions
- **The service layer owns all database access**, so business rules can't be bypassed by a controller reaching into the DB.
- **Redis was chosen for the queue** over a Postgres-based queue for throughput; revisit if operational overhead grows.

# Reflection & Saga Systems Audit

_Date: 2025-12-23_

## Scope
Reviewed backend reflection scheduling/processing and saga orchestration components to assess architecture, triggers, resilience, observability, and risks. Key files: `backend/reflection/scheduler.py`, `backend/scheduler/integration.py`, `backend/main.py`, saga modules under `backend/saga/`.

## Reflection System
- **Scheduler roles**: `ReflectionScheduler` orchestrates periodic and ad-hoc jobs via central `SchedulerService` (APScheduler wrapper). Registers daily, weekly, relationship-question, relationship-answer audit, and nightly mantra refresh jobs with cron triggers and TZ handling; uses string callables for persistence-friendly jobs. @selo-ai/backend/reflection/scheduler.py#296-566
- **Job implementations**: Module-level async functions run reflections for all active users, skip users without interactions, and record job state; mantra refresh and relationship audit jobs also run. @selo-ai/backend/reflection/scheduler.py#41-264
- **One-off reflections**: `schedule_reflection` adds date-triggered jobs for a user with optional delay and context, guarding on user interaction. @selo-ai/backend/reflection/scheduler.py#652-709
- **Integration layer**: `SchedulerIntegration.setup` wires scheduler service, resource monitor, adaptive scheduler, event trigger system, and creates `ReflectionScheduler`. It registers event handlers and default triggers (emotional spike, knowledge update patterns, high-importance memories) and initializes schedules. @selo-ai/backend/scheduler/integration.py#101-417
- **Event-driven triggers**: Memory events with importance ≥0.8 schedule both immediate reflection via scheduler and direct processor invocation; conversation/user activity updates feed adaptive scheduler. @selo-ai/backend/scheduler/integration.py#282-372
- **Lifecycle**: `main.py` initializes legacy scheduler (for env-based config) and enhanced `SchedulerIntegration` during FastAPI lifespan; performs diagnostic checks for expected jobs and catch-up for missed daily reflections. @selo-ai/backend/main.py#912-1417
- **Timeout policy**: Lifespan enforces bounded reflection defaults, with environment overrides; warns if LLM timeout is too close to sync timeout. @selo-ai/backend/main.py#1150-1294

### Strengths
- Uses string callables for APScheduler persistence; misfire grace/coalescing configured to reduce duplication.
- Activity gating avoids reflections before first user interaction.
- Catch-up task ensures missed daily reflections are backfilled on startup.
- Event-trigger system allows contextual triggers (emotional spikes, knowledge updates, important memories).

### Risks / Gaps
1) **Duplication between legacy and enhanced schedulers**: Both `legacy_reflection_scheduler` and `scheduler_integration.reflection_scheduler` exist; unclear single source of truth for job registration and API exposure. Risk of double scheduling or divergent configs. @selo-ai/backend/main.py#912-1367
2) **Unbounded default timeouts**: Defaults set reflection sync/LLM timeouts to `0` (unbounded) unless env overrides; may cause resource exhaustion under failures. @selo-ai/backend/main.py#1168-1294
3) **Resource-aware scheduling**: Resource monitor exists, but reflection jobs themselves do not check resource constraints; heavy workloads could overload.
4) **Retry/backoff for jobs**: Module-level jobs lack retry/backoff; failures just log and continue; misfire grace covers scheduling gaps but not transient errors.
5) **Metrics/observability**: `SchedulerService` tracks job metrics, but reflection jobs do not emit structured metrics or tracing spans; JOB_EXECUTION_STATE is in-memory only (lost on restart). @selo-ai/backend/scheduler/scheduler_service.py#125-219
6) **Event storm control**: High-importance memory handler can both schedule and immediately generate reflections; no dedup/cooldown—possible double work or thundering herd if multiple memories arrive.
7) **TZ fallback**: Falls back to UTC silently when pytz missing; may shift expected user-facing times.
8) **Testing**: No tests observed for reflection scheduler/triggers; risk of cron/timezone regressions.

### Recommendations
- Consolidate to a single reflection scheduler instance; deprecate legacy path or gate with config flag to avoid double registration.
- Set bounded, non-zero defaults for `REFLECTION_SYNC_TIMEOUT_S` and `REFLECTION_LLM_TIMEOUT_S` (e.g., 20s/15s) with clear overrides.
- Add resource guardrails in reflection jobs (skip/queue when CPU/memory high) leveraging `ResourceMonitor` callbacks.
- Introduce retry with jitter for per-user reflection generation within jobs; record failures to persistence for catch-up.
- Emit structured metrics (success/fail counts, latency) to a persistent store; expose a health endpoint summarizing JOB_EXECUTION_STATE.
- Add dedup/cooldown for important-memory-triggered reflections; consider idempotency keys per (user, memory_id, trigger_type).
- Ensure pytz dependency present and log explicit timezone used.
- Add tests: cron registration, job ID expectations, user gating, and event-trigger flows.

## Saga System
- **Purpose**: Coordinates multi-step operations with compensation. Entities persisted via `Saga`/`SagaStep` models and `SagaRepository` for create/read/update/list. @selo-ai/backend/db/models/saga.py#1-117 @selo-ai/backend/db/repositories/saga.py#1-198
- **Orchestrator**: `SagaOrchestrator` executes steps sequentially, supports retries with exponential backoff, passes previous output into next step, and runs compensation in reverse on failure; compensation handlers are registered by name. @selo-ai/backend/saga/orchestrator.py#1-370
- **Integration**: `SagaIntegration` wires repositories and registers handlers for persona evolution, goal creation, conversation processing, and episode generation; exposes execute/retry/list helpers. @selo-ai/backend/saga/integration.py#1-595
- **Usage example**: Persona integration optionally routes reflection-driven evolution through saga for compensation; otherwise direct path. @selo-ai/backend/persona/integration.py#330-370
- **Monitoring**: `SagaMonitor` can detect stuck/failed sagas and retry based on thresholds. @selo-ai/backend/saga/monitor.py#1-116

### Strengths
- Clear separation: repo persistence, orchestrator logic, integration registration, and monitoring.
- Compensation support with reverse-order execution; retry with backoff per step.
- Output chaining between steps allows context propagation without manual wiring.

### Risks / Gaps
1) **Handler coverage/config drift**: Compensation handlers must be registered manually; missing handler results in silent skip of compensation (warning only). @selo-ai/backend/saga/orchestrator.py#282-337
2) **Idempotency**: No idempotency keys or deduplication; retry/compensate could re-apply side effects (DB writes, LLM calls) without guards.
3) **Observability**: Limited metrics/log structure; no tracing spans or alerting hooks. SagaMonitor logs but does not emit alerts.
4) **Backoff limits**: Step retries capped but not jittered after max; orchestrator marks saga failed without enqueueing for monitor-driven retry automatically.
5) **Data validation**: Inputs to handlers not validated centrally; bad configs could fail late.
6) **Testing**: No visible automated tests for orchestrator/integration/monitor; risk of regression in compensation flows.

### Recommendations
- Enforce registration checks: fail fast if any step lacks compensation handler when saga definition requires one.
- Add idempotency tokens to steps and persistence to avoid duplicate side effects on retry/compensation.
- Emit structured saga metrics (per step success/fail/latency, compensation counts) and integrate with monitoring/alerts; expose API for active/failed sagas.
- Extend SagaMonitor to auto-retry failed sagas within policy and to surface stuck saga alerts via notification channel.
- Add schema validation for saga definitions (Pydantic) before creation/execution.
- Add unit/integration tests covering happy path, handler failure leading to compensation, retry exhaustion, and monitor detection of stuck sagas.

## Quick Wins
1) Add bounded default reflection timeouts and remove legacy scheduler to prevent double scheduling.
2) Add cooldown/idempotency for important-memory-triggered reflections to curb duplicate work.
3) Implement structured metrics for reflection jobs and saga steps; expose health endpoint and alerting on saga failures/stuck states.
4) Add tests for scheduler job registration and saga compensation flows.

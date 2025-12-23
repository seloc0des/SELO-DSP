Findings:

Nightly reflection job registration exists in code (ReflectionScheduler._register_daily_reflection_job, _register_nightly_mantra_refresh_job) and logs show each job was registered successfully @backend/reflection/scheduler.py#297-336,457-494. So scheduler setup code isn’t broken.
The runtime logs never show the nightly reflection job firing. Instead we see a catch-up warning at startup (Scheduler catch-up: no previous daily reflection found; midnight job may have been skipped. @backend/main.py#1388-1475). That warning occurs when the reflection repo returns no recent “daily” entries. Since the catch-up routine ran immediately on startup, the job must have been missed earlier (app likely down/offline around 00:00 local or scheduler paused), and the catch-up ran instead.
Contributing factors:
Scheduler job uses America/New_York timezone with misfire_grace_time=3600. If the app wasn’t running during that window, APScheduler marks the job misfired and won’t backfill automatically (consistent with the catch-up logic needing to run).
CPU saturation warnings (100% usage) around the same timeframe @backend.log may also have delayed scheduler execution or caused long GC pauses.
Reflection processing runs with unbounded timeouts (timeouts ≤ 0), so long-running LLM calls can block worker threads and starve the scheduler, increasing the risk of missing time-sensitive jobs.
No error messages indicate the job attempted and failed; instead there’s simply no evidence it ran. Therefore root cause is operational: the service wasn’t alive or responsive at its scheduled run time (midnight ET), so APScheduler never executed the nightly reflection. The catch-up routine triggered afterwards with a warning.
Recommendations:

Ensure the backend is running across midnight or configure APScheduler with coalesce=True and longer misfire_grace_time for critical jobs.
Consider adding logging inside run_daily_reflections / refresh_nightly_mantras to confirm execution each day and alert if skipped.
Re-enable sane reflection timeouts (non-zero) to avoid runaway tasks starving the scheduler.
Investigate CPU saturation (100%) and reduce load or provision more capacity so scheduled jobs have resources to run.
import asyncio
import pytest

from backend.agent.agent_loop_runner import AgentLoopRunner
from backend.scheduler.integration import SchedulerIntegration


class _StubUser:
    def __init__(self, user_id: str = "user-1"):
        self.id = user_id


class _StubPersona:
    def __init__(self, persona_id: str = "persona-1"):
        self.id = persona_id
        self.traits = []


class _AffectiveStateManagerStub:
    def __init__(self):
        self.homeostasis_calls = []

    async def ensure_state_available(self, persona_id: str, user_id: str):
        return {"energy": 0.4, "stress": 0.6, "confidence": 0.7}

    async def run_homeostasis_decay(self, persona_id: str):
        self.homeostasis_calls.append(persona_id)


class _GoalManagerStub:
    def __init__(self):
        self.meta_directives = [{"id": "directive-1"}]
        self.pending_steps = []

    async def list_active_goals(self, persona_id: str):
        return [{"id": "goal-1", "persona_id": persona_id, "user_id": "user-1", "description": "Grow"}]

    async def list_pending_steps(self, persona_id: str):
        return list(self.pending_steps)

    async def list_meta_directives(self, persona_id: str, statuses=None, limit=25):
        return list(self.meta_directives)


class _PlannerServiceStub:
    def __init__(self, generated_steps=None):
        self.generated_steps = generated_steps or [
            {
                "id": "step-1",
                "goal_id": "goal-1",
                "persona_id": "persona-1",
                "user_id": "user-1",
                "description": "Draft outreach message",
                "priority": 0.8,
            }
        ]

    async def generate_plan_steps(self, persona_id: str, affective_state, time_now=None):
        return list(self.generated_steps)


class _PersonaRepoStub:
    async def get_or_create_default_persona(self, user_id: str, include_traits: bool = False):
        return _StubPersona()


class _UserRepoStub:
    async def get_or_create_default_user(self):
        return _StubUser()


class _EpisodeBuilderStub:
    def __init__(self):
        self.created_payloads = []

    async def build_episode(self, persona_id: str, user_id: str, artifacts):
        self.created_payloads.append(artifacts)


class _EventSystemStub:
    def __init__(self):
        self.events = []

    async def publish_event(self, event_type: str, event_data):
        self.events.append((event_type, event_data))


@pytest.mark.asyncio
async def test_agent_loop_runner_generates_plan_steps_and_episode():
    affective_manager = _AffectiveStateManagerStub()
    goal_manager = _GoalManagerStub()
    planner_service = _PlannerServiceStub()
    persona_repo = _PersonaRepoStub()
    user_repo = _UserRepoStub()
    episode_builder = _EpisodeBuilderStub()
    event_system = _EventSystemStub()

    runner = AgentLoopRunner(
        affective_state_manager=affective_manager,
        goal_manager=goal_manager,
        planner_service=planner_service,
        persona_repo=persona_repo,
        user_repo=user_repo,
        episode_builder=episode_builder,
        event_system=event_system,
        config={
            "enabled": True,
            "interval_seconds": 300,
            "homeostasis_enabled": True,
            "episode_builder_enabled": True,
            "audit_events_enabled": True,
        },
    )

    result = await runner.run(reason="unit_test")

    assert result["new_steps_generated"] == 1
    assert affective_manager.homeostasis_calls == ["persona-1"]
    assert len(episode_builder.created_payloads) == 1
    assert episode_builder.created_payloads[0]["title"] == "Agent loop planning update"
    assert event_system.events and event_system.events[0][0] == "agent.loop.tick"


class _SchedulerServiceStub:
    def __init__(self):
        self.jobs = {}

    async def add_job(self, job_id: str, func, trigger: str, **trigger_args):
        self.jobs[job_id] = {"func": func, "trigger": trigger, **trigger_args}

    async def remove_job(self, job_id: str):
        self.jobs.pop(job_id, None)


class _AgentLoopRunnerStub:
    def __init__(self):
        self.enabled = True
        self.interval_seconds = 180
        self.run_calls = []

    async def run(self, *, reason: str):
        self.run_calls.append(reason)


@pytest.mark.asyncio
async def test_scheduler_integration_registers_agent_loop_job():
    agent_runner = _AgentLoopRunnerStub()
    integration = SchedulerIntegration(
        agent_loop_runner=agent_runner,
        config={}
    )
    # Mock the scheduler service and mark as initialized to skip setup()
    integration.scheduler_service = _SchedulerServiceStub()
    integration.initialized = True
    integration._agent_loop_job_id = "agent_loop_runner"

    await integration._schedule_agent_loop()

    assert "agent_loop_runner" in integration.scheduler_service.jobs
    job_details = integration.scheduler_service.jobs["agent_loop_runner"]
    assert job_details["trigger"] == "interval"
    assert job_details["seconds"] == 180


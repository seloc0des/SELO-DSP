import asyncio
import hashlib
import os
import pathlib
from datetime import datetime, timezone
from typing import Optional, Any, Dict

# Session wrap-up background loop
# Env knobs:
# - SESSION_WRAPUP_ENABLED: "true" | "false" (default: true)
# - SESSION_WRAPUP_GAP_MIN: minutes of inactivity before wrap-up (default: 60)
# - SESSION_WRAPUP_SCAN_INTERVAL_S: scan interval seconds (default: 120)
# - SESSION_WRAPUP_MAX_TOKENS: <=0 means unlimited (omit max_tokens)
# - SESSION_WRAPUP_TIMEOUT_S: overall timeout per reflection (default: 15)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _session_marker_path(base_dir: pathlib.Path, session_id: str, last_message_at: Optional[datetime]) -> pathlib.Path:
    last_iso = (last_message_at or datetime.fromtimestamp(0, tz=timezone.utc)).isoformat()
    key = f"{session_id}|{last_iso}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return base_dir / f".wrap_session_{digest}.done"


async def _wrapup_session(app, session_id: str, last_message_at: Optional[datetime], *,
                          max_tokens: Optional[int], timeout_s: Optional[float]):
    """Perform session summary reflection and memory extraction for a session."""
    services = app.state.services
    reflection_processor = services.get("reflection_processor")
    conversation_repo = services.get("conversation_repo")
    reflection_ns = services.get("reflection_namespace")
    llm_router = services.get("llm_router")

    if not reflection_processor or not conversation_repo:
        return

    # Generate session summary reflection
    try:
        # Use a synthetic turn_id for traceability
        import uuid
        turn_id = str(uuid.uuid4())

        # Reflection call (omit max_tokens when <=0)
        async def _do_reflect():
            kwargs: Dict[str, Any] = dict(
                reflection_type="session_summary",
                # Single-user system: unify reflections under the default_user profile
                user_profile_id="default_user",
                memory_ids=None,
                max_context_items=25,
                trigger_source="session_idle",
                turn_id=turn_id,
                metadata={"session_id": session_id, "source": "session_idle"},
            )
            return await reflection_processor.generate_reflection(**kwargs)

        if timeout_s and timeout_s > 0:
            summary_res = await asyncio.wait_for(_do_reflect(), timeout=timeout_s)
        else:
            summary_res = await _do_reflect()

        # Normalize result and extract text
        summary_payload = (summary_res or {}).get("result", {}) or {}
        summary_text = summary_payload.get("content", "") or (summary_res or {}).get("content", "")
        if not isinstance(summary_text, str):
            summary_text = ""

        # Emit reflection event to UI
        try:
            if reflection_ns is not None and summary_text:
                await reflection_ns.emit_reflection_event(
                    event_name="reflection_generated",
                    data={
                        "reflection_id": (summary_res or {}).get("id"),
                        "reflection_type": "session_summary",
                        "result": summary_payload if summary_payload else {"content": summary_text},
                        "user_profile_id": session_id,
                        "created_at": (summary_res or {}).get("created_at"),
                        "turn_id": turn_id,
                    },
                    user_id=session_id,
                )
        except Exception:
            pass

        # Extract memory from the summary
        try:
            from ..memory.extractor import MemoryExtractor
            extractor = MemoryExtractor(conversation_repo=conversation_repo, llm_controller=llm_router)
            if summary_text:
                await extractor.extract_memory_from_single_message(
                    user_id="default_user",
                    message={"role": "assistant", "content": summary_text},
                    conversation_id=str((await conversation_repo.get_or_create_conversation(session_id, (await services.get("user_repo").get_or_create_default_user()).id)).id),
                )
        except Exception:
            pass

    except asyncio.TimeoutError:
        # Skip on timeout; will try again next scan if still applicable
        return
    except Exception:
        return


async def session_wrapup_loop(app):
    env = os.environ
    try:
        enabled = env.get("SESSION_WRAPUP_ENABLED", "true").lower() in ("1", "true", "yes")
        if not enabled:
            return
        try:
            gap_min = float(env.get("SESSION_WRAPUP_GAP_MIN", "60"))
        except Exception:
            gap_min = 60.0
        try:
            scan_interval_s = float(env.get("SESSION_WRAPUP_SCAN_INTERVAL_S", "120"))
        except Exception:
            scan_interval_s = 120.0
        try:
            wrap_max_tokens = int(env.get("SESSION_WRAPUP_MAX_TOKENS", "0"))
        except Exception:
            wrap_max_tokens = 0
        try:
            wrap_timeout_s = float(env.get("SESSION_WRAPUP_TIMEOUT_S", "15"))
        except Exception:
            wrap_timeout_s = 15.0

        services = app.state.services
        conversation_repo = services.get("conversation_repo")
        user_repo = services.get("user_repo")
        if not conversation_repo or not user_repo:
            return

        # Base dir for wrap markers
        try:
            base_dir = os.environ.get("INSTALL_DIR") or str(pathlib.Path(__file__).resolve().parents[2])
            base_dir_path = pathlib.Path(base_dir)
        except Exception:
            base_dir_path = pathlib.Path(".")

        while True:
            try:
                # List conversations (single-user system -> default user)
                user = await user_repo.get_or_create_default_user()
                # Fetch a reasonable batch of conversations; repository API should support pagination
                try:
                    conversations = await conversation_repo.list_conversations(
                        user_id=str(user.id), limit=200, offset=0, sort_by="last_message_at", sort_order="desc"
                    )
                except Exception:
                    # Fallback: if list_conversations not available, skip this scan
                    conversations = []

                now = _utc_now()
                for c in conversations or []:
                    try:
                        session_id = getattr(c, "session_id", None) or getattr(c, "sessionId", None) or getattr(c, "id", None)
                        last_at = getattr(c, "last_message_at", None) or getattr(c, "lastMessageAt", None)
                        if isinstance(last_at, str):
                            try:
                                last_at = datetime.fromisoformat(last_at.replace('Z', '+00:00'))
                            except Exception:
                                last_at = None
                        last_at = _to_utc(last_at)
                        if not session_id or last_at is None:
                            continue
                        gap = (now - last_at).total_seconds() / 60.0
                        if gap <= gap_min:
                            continue
                        # Marker check per (session_id, last_message_at)
                        marker = _session_marker_path(base_dir_path, str(session_id), last_at)
                        if marker.exists():
                            continue
                        await _wrapup_session(
                            app,
                            str(session_id),
                            last_at,
                            max_tokens=wrap_max_tokens if wrap_max_tokens > 0 else None,
                            timeout_s=wrap_timeout_s if wrap_timeout_s > 0 else None,
                        )
                        try:
                            marker.write_text("done\n", encoding="utf-8")
                        except Exception:
                            pass
                    except Exception:
                        continue

            except asyncio.CancelledError:
                break
            except Exception:
                # Swallow and continue next tick
                pass
            await asyncio.sleep(max(10.0, scan_interval_s))
    except asyncio.CancelledError:
        return
    except Exception:
        return

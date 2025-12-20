import logging
import asyncio
from typing import Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime, timezone

if TYPE_CHECKING:
    from ...scheduler.event_triggers import EventTriggerSystem

logger = logging.getLogger("selo.socketio.chat")


class ChatNamespace:
    """Socket.IO namespace for streaming chat responses."""

    def __init__(self, socketio_server):
        self.sio = socketio_server
        self.namespace = "/chat"
        
        # Event system for emitting conversation lifecycle events
        self._event_system: Optional["EventTriggerSystem"] = None
        
        # Track active sessions and their last activity
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Idle detection configuration
        self._idle_check_seconds = 300  # 5 minutes of no activity = idle
        self._pending_idle_checks: Dict[str, asyncio.Task] = {}
    
    def bind_event_system(self, event_system: "EventTriggerSystem") -> None:
        """Bind event system for conversation lifecycle events."""
        self._event_system = event_system
        logger.info("Chat namespace bound to event system")

    def register(self):
        if self.sio is None:
            logger.warning("Socket.IO server unavailable; chat namespace not registered")
            return

        @self.sio.on("join", namespace=self.namespace)
        async def join(sid, data):
            user_id = (data or {}).get("user_id")
            if not user_id:
                await self.sio.emit(
                    "error",
                    {"message": "user_id required"},
                    room=sid,
                    namespace=self.namespace,
                )
                return
            await self.sio.enter_room(sid, user_id, namespace=self.namespace)
            
            # Track this session as active
            self._active_sessions[sid] = {
                "user_id": user_id,
                "joined_at": datetime.now(timezone.utc),
                "last_activity": datetime.now(timezone.utc),
            }
            
            await self.sio.emit(
                "joined",
                {"ok": True, "user_id": user_id},
                room=sid,
                namespace=self.namespace,
            )
        
        @self.sio.on("disconnect", namespace=self.namespace)
        async def disconnect(sid, reason=None):
            """Handle client disconnect - emit conversation.ended event."""
            session_info = self._active_sessions.pop(sid, None)
            if not session_info:
                return
            
            user_id = session_info.get("user_id")
            if not user_id:
                return
            
            # Cancel any pending idle check for this session
            if sid in self._pending_idle_checks:
                self._pending_idle_checks[sid].cancel()
                del self._pending_idle_checks[sid]
            
            # Emit conversation.ended event
            await self._emit_conversation_ended(user_id, session_info, reason="disconnect")
            logger.debug(f"User {user_id} disconnected from chat namespace")

    async def emit_chat_chunk(
        self,
        user_id: str,
        turn_id: str,
        chunk: str,
        final: bool = False,
        timestamp: Optional[str] = None,
    ):
        if not self.sio:
            logger.debug("Socket.IO server missing; skipping chat chunk emit")
            return
        payload = {
            "turn_id": turn_id,
            "chunk": chunk,
            "final": final,
        }
        if timestamp is not None:
            payload["timestamp"] = timestamp
        await self.sio.emit(
            "chat_chunk",
            payload,
            room=user_id,
            namespace=self.namespace,
        )

    async def emit_chat_complete(
        self,
        user_id: str,
        turn_id: str,
        content: str,
        timestamp: Optional[str] = None,
    ):
        if not self.sio:
            return
        payload = {
            "turn_id": turn_id,
            "content": content,
        }
        if timestamp is not None:
            payload["timestamp"] = timestamp
        await self.sio.emit(
            "chat_complete",
            payload,
            room=user_id,
            namespace=self.namespace,
        )
        
        # Update last activity and schedule idle check
        await self._update_activity(user_id)
    
    async def _update_activity(self, user_id: str) -> None:
        """Update last activity time and schedule idle check."""
        now = datetime.now(timezone.utc)
        
        # Find session by user_id
        for sid, info in self._active_sessions.items():
            if info.get("user_id") == user_id:
                info["last_activity"] = now
                
                # Cancel existing idle check
                if sid in self._pending_idle_checks:
                    self._pending_idle_checks[sid].cancel()
                
                # Schedule new idle check
                self._pending_idle_checks[sid] = asyncio.create_task(
                    self._check_idle_after_delay(sid, user_id)
                )
                break
    
    async def _check_idle_after_delay(self, sid: str, user_id: str) -> None:
        """
        Wait for idle period and emit conversation.idle event if still inactive.
        """
        try:
            await asyncio.sleep(self._idle_check_seconds)
            
            # Check if session still exists and is still idle
            session_info = self._active_sessions.get(sid)
            if not session_info:
                return
            
            last_activity = session_info.get("last_activity")
            if not last_activity:
                return
            
            now = datetime.now(timezone.utc)
            idle_seconds = (now - last_activity).total_seconds()
            
            if idle_seconds >= self._idle_check_seconds:
                # Session has been idle - emit conversation.ended event
                await self._emit_conversation_ended(user_id, session_info, reason="idle")
                logger.info(f"Conversation idle detected for user {user_id} after {idle_seconds:.0f}s")
                
        except asyncio.CancelledError:
            # Task was cancelled (new activity came in)
            pass
        except Exception as e:
            logger.error(f"Error in idle check: {e}", exc_info=True)
        finally:
            # Clean up pending check
            self._pending_idle_checks.pop(sid, None)
    
    async def _emit_conversation_ended(
        self, 
        user_id: str, 
        session_info: Dict[str, Any],
        reason: str = "unknown"
    ) -> None:
        """
        Emit conversation.ended event to the event system.
        
        Args:
            user_id: User ID
            session_info: Session tracking info
            reason: Why conversation ended ("disconnect", "idle", etc.)
        """
        if not self._event_system:
            logger.debug("No event system bound - skipping conversation.ended event")
            return
        
        try:
            from ...scheduler.event_triggers import EventType
            
            event_data = {
                "user_id": user_id,
                "reason": reason,
                "joined_at": session_info.get("joined_at", datetime.now(timezone.utc)).isoformat(),
                "last_activity": session_info.get("last_activity", datetime.now(timezone.utc)).isoformat(),
                "ended_at": datetime.now(timezone.utc).isoformat(),
            }
            
            await self._event_system.publish_event(
                event_type=EventType.CONVERSATION_ENDED,
                event_data=event_data,
            )
            
            logger.debug(f"Emitted conversation.ended event for user {user_id} (reason={reason})")
            
        except Exception as e:
            logger.error(f"Failed to emit conversation.ended event: {e}", exc_info=True)

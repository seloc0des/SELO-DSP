"""
Event Trigger System

This module implements an event-based trigger system for scheduling
reflections and tasks based on real-time events rather than just time-based schedules.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta, timezone
import time
import json

from .scheduler_service import SchedulerService
from .adaptive_scheduler import AdaptiveScheduler
from ..utils.datetime import utc_now, ensure_utc

logger = logging.getLogger("selo.scheduler.events")

class EventType:
    """Event type constants"""
    CONVERSATION = "conversation"
    CONVERSATION_ENDED = "conversation.ended"
    CONVERSATION_IDLE = "conversation.idle"
    MEMORY_CREATED = "memory_created"
    USER_INTERACTION = "user_interaction"
    EMOTIONAL_SPIKE = "emotional_spike"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    SYSTEM_EVENT = "system_event"
    EXTERNAL_EVENT = "external_event"
    REFLECTION_CREATED = "reflection_created"
    LEARNING_CREATED = "learning_created"
    RESOURCES_AVAILABLE = "resources.available"
    RESOURCES_CONSTRAINED = "resources.constrained"

class EventTriggerSystem:
    """
    Event-based trigger system for dynamic scheduling.
    
    This system monitors events and triggers jobs based on event patterns,
    thresholds, and importance rather than fixed schedules.
    """
    
    def __init__(self, 
                 scheduler_service: Optional[SchedulerService] = None,
                 adaptive_scheduler: Optional[AdaptiveScheduler] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the event trigger system.
        
        Args:
            scheduler_service: The scheduler service (optional)
            adaptive_scheduler: The adaptive scheduler (optional)
            config: Configuration options
        """
        self.scheduler_service = scheduler_service
        self.adaptive_scheduler = adaptive_scheduler
        self.config = config or {}
        
        # Event handlers by event type
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Event history for pattern detection
        self.event_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Registered triggers with conditions
        self.triggers: Dict[str, Dict[str, Any]] = {}
        
        # Cooldown tracking to prevent excessive triggering
        self.trigger_cooldowns: Dict[str, datetime] = {}
        
        # Event patterns being monitored
        self.patterns: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Event trigger system initialized")
        
    async def register_event_handler(self, 
                                   event_type: str, 
                                   handler: Callable):
        """
        Register a handler for an event type.
        
        Args:
            event_type: Type of event to handle
            handler: Async callback function
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
            
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type {event_type}")

    # --- Backward-compatibility helpers ---
    async def register_handler(self, event_type: str, handler: Callable):
        """
        Backward-compatible alias for register_event_handler used by older integration code.
        """
        return await self.register_event_handler(event_type, handler)
        
    async def register_trigger(self, 
                             trigger_id: str, 
                             event_type: str,
                             condition: Dict[str, Any],
                             action: Callable,
                             cooldown_seconds: int = 3600,
                             importance: float = 0.5):
        """
        Register an event trigger.
        
        Args:
            trigger_id: Unique ID for this trigger
            event_type: Type of event that activates this trigger
            condition: Conditions that must be met
            action: Action to perform when triggered
            cooldown_seconds: Minimum seconds between triggers
            importance: Importance of this trigger (0-1)
        """
        self._validate_condition_spec(condition, context=f"trigger '{trigger_id}'")

        self.triggers[trigger_id] = {
            "event_type": event_type,
            "condition": condition,
            "action": action,
            "cooldown_seconds": cooldown_seconds,
            "importance": importance,
            "last_triggered": None
        }
        
        logger.info(f"Registered trigger {trigger_id} for event type {event_type}")
        
    async def register_pattern(self, 
                             pattern_id: str,
                             event_types: List[str],
                             pattern_config: Dict[str, Any],
                             action: Callable):
        """
        Register a pattern to monitor across multiple events.
        
        Args:
            pattern_id: Unique ID for this pattern
            event_types: Types of events to include in pattern
            pattern_config: Pattern configuration and detection rules
            action: Action to perform when pattern detected
        """
        self._validate_pattern_spec(pattern_id, event_types, pattern_config)

        self.patterns[pattern_id] = {
            "event_types": event_types,
            "config": pattern_config,
            "action": action,
            "last_triggered": None,
            "partial_matches": []
        }
        
        logger.info(f"Registered pattern {pattern_id} for event types {event_types}")
        
    async def process_event(self, 
                          event_type: str, 
                          event_data: Dict[str, Any],
                          user_id: Optional[str] = None):
        """
        Process an incoming event.
        
        Args:
            event_type: Type of event
            event_data: Event details
            user_id: Associated user ID if applicable
        """
        # Record event in history
        if event_type not in self.event_history:
            self.event_history[event_type] = []
            
        event_record = {
            "timestamp": utc_now(),
            "type": event_type,
            "data": event_data,
            "user_id": user_id
        }
        
        self.event_history[event_type].append(event_record)
        
        # Limit history size
        max_history = self.config.get("max_event_history", 1000)
        if len(self.event_history[event_type]) > max_history:
            self.event_history[event_type] = self.event_history[event_type][-max_history:]
        
        # Update user activity if applicable
        if user_id and event_type == EventType.USER_INTERACTION:
            if self.adaptive_scheduler:
                activity_level = event_data.get("activity_level", 0.5)
                await self.adaptive_scheduler.update_user_activity(user_id, activity_level)
            else:
                logger.debug("Adaptive scheduler not available, skipping user activity update")
        
        # Check triggers for this event type
        await self._check_triggers(event_type, event_data, user_id)
        
        # Check patterns involving this event type
        await self._check_patterns(event_type, event_data, user_id)
        
        # Call registered handlers
        await self._call_handlers(event_type, event_data, user_id)
        
        logger.debug(f"Processed {event_type} event for user {user_id}")

    async def publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Backward-compatible alias for process_event.
        """
        user_id = event_data.get("user_id") if isinstance(event_data, dict) else None
        return await self.process_event(event_type, event_data, user_id=user_id)

    async def schedule_event(self,
                             event_type: str,
                             event_data: Dict[str, Any],
                             delay_seconds: int = 0):
        """
        Schedule an event to be published after a delay.
        Provided for compatibility with existing code paths.
        """
        async def _delayed_publish():
            try:
                await asyncio.sleep(max(0, int(delay_seconds)))
                await self.publish_event(event_type, event_data)
            except Exception as e:
                logger.error(f"Error in scheduled event for {event_type}: {e}", exc_info=True)
        asyncio.create_task(_delayed_publish())
        
    async def _check_triggers(self, 
                            event_type: str, 
                            event_data: Dict[str, Any],
                            user_id: Optional[str]):
        """
        Check if any triggers should fire for this event.
        
        Args:
            event_type: Event type
            event_data: Event data
            user_id: User ID if applicable
        """
        now = utc_now()

        for trigger_id, trigger in self.triggers.items():
            # Check if this trigger applies to this event type
            if trigger["event_type"] != event_type:
                continue
                
            # Check cooldown
            last_triggered = trigger.get("last_triggered")
            if last_triggered:
                elapsed = (now - ensure_utc(last_triggered)).total_seconds()
                if elapsed < trigger["cooldown_seconds"]:
                    continue
                    
            # Check conditions
            if await self._evaluate_condition(trigger["condition"], event_data, user_id):
                # Trigger the action
                try:
                    logger.info(f"Triggering action for {trigger_id}")
                    trigger["last_triggered"] = now
                    asyncio.create_task(trigger["action"](event_data, user_id))
                except Exception as e:
                    logger.error(f"Error executing trigger {trigger_id}: {str(e)}", exc_info=True)
                    
    async def _check_patterns(self, 
                           event_type: str, 
                           event_data: Dict[str, Any],
                           user_id: Optional[str]):
        """
        Check for pattern matches with this event.
        
        Args:
            event_type: Event type
            event_data: Event data
            user_id: User ID if applicable
        """
        now = datetime.now(timezone.utc)
        
        for pattern_id, pattern in self.patterns.items():
            # Skip if this event type is not part of this pattern
            if event_type not in pattern["event_types"]:
                continue
                
            # Skip if on cooldown
            last_triggered = pattern.get("last_triggered")
            cooldown = pattern["config"].get("cooldown_seconds", 3600)
            if last_triggered and (now - ensure_utc(last_triggered)).total_seconds() < cooldown:
                continue
                
            # Check pattern match
            try:
                config = pattern["config"]
                pattern_type = config.get("type", "sequence")
                
                if pattern_type == "sequence":
                    await self._check_sequence_pattern(pattern_id, pattern, event_type, event_data, user_id)
                elif pattern_type == "frequency":
                    await self._check_frequency_pattern(pattern_id, pattern, event_type, event_data, user_id)
                elif pattern_type == "correlation":
                    await self._check_correlation_pattern(pattern_id, pattern, event_type, event_data, user_id)
            except Exception as e:
                logger.error(f"Error checking pattern {pattern_id}: {str(e)}", exc_info=True)
                
    async def _check_frequency_pattern(self, 
                                    pattern_id: str, 
                                    pattern: Dict[str, Any], 
                                    event_type: str,
                                    event_data: Dict[str, Any], 
                                    user_id: Optional[str]):
        """
        Check for a frequency pattern match.
        
        Args:
            pattern_id: Pattern ID
            pattern: Pattern definition
            event_type: Current event type
            event_data: Current event data
            user_id: User ID if applicable
        """
        config = pattern["config"]
        thresholds = config.get("thresholds", {})
        time_window_seconds = config.get("time_window_seconds", 3600)
        
        if not thresholds:
            return
            
        # Count events by type in the time window
        counts = {}
        now = utc_now()
        cutoff = now - timedelta(seconds=time_window_seconds)
        
        for et in pattern["event_types"]:
            if et not in self.event_history:
                counts[et] = 0
                continue
                
            counts[et] = len([
                e for e in self.event_history[et]
                if ensure_utc(e["timestamp"]) >= cutoff and (not user_id or e["user_id"] == user_id)
            ])
            
        # Check if all thresholds are met
        all_met = True
        for et, threshold in thresholds.items():
            if et not in counts or counts[et] < threshold:
                all_met = False
                break
                
        if all_met:
            try:
                logger.info(f"Pattern {pattern_id} frequency thresholds met")
                pattern["last_triggered"] = now
                
                # Collect all events in the time window
                events = []
                for et in pattern["event_types"]:
                    if et in self.event_history:
                        events.extend([
                            e for e in self.event_history[et]
                            if e["timestamp"] >= cutoff and (not user_id or e["user_id"] == user_id)
                        ])
                
                # Sort by timestamp
                events.sort(key=lambda e: e["timestamp"])
                
                asyncio.create_task(pattern["action"](events, user_id))
            except Exception as e:
                logger.error(f"Error executing pattern action {pattern_id}: {str(e)}", exc_info=True)
                
    async def _check_correlation_pattern(self, 
                                      pattern_id: str, 
                                      pattern: Dict[str, Any], 
                                      event_type: str,
                                      event_data: Dict[str, Any], 
                                      user_id: Optional[str]):
        """
        Check for correlated events pattern.
        
        Args:
            pattern_id: Pattern ID
            pattern: Pattern definition
            event_type: Current event type
            event_data: Current event data
            user_id: User ID if applicable
        """
        config = pattern["config"]
        correlations = config.get("correlations", [])
        time_window_seconds = config.get("time_window_seconds", 3600)
        
        if not correlations:
            return
            
        # Find correlated events
        now = utc_now()
        cutoff = now - timedelta(seconds=time_window_seconds)
        
        for correlation in correlations:
            source_type = correlation.get("source_type")
            target_type = correlation.get("target_type")
            source_field = correlation.get("source_field")
            target_field = correlation.get("target_field")
            
            if not all([source_type, target_type, source_field, target_field]):
                continue
                
            # This event is the source
            if event_type == source_type:
                if source_field not in event_data:
                    continue
                    
                source_value = event_data[source_field]
                
                # Find matching target events
                if target_type in self.event_history:
                    matches = []
                    for e in self.event_history[target_type]:
                        if e["timestamp"] < cutoff:
                            continue
                            
                        if user_id and e["user_id"] and e["user_id"] != user_id:
                            continue
                            
                        if target_field in e["data"] and e["data"][target_field] == source_value:
                            matches.append(e)
                            
                    if matches:
                        try:
                            logger.info(f"Pattern {pattern_id} correlation found between {source_type} and {target_type}")
                            pattern["last_triggered"] = now
                            
                            # Combine source and targets
                            correlated_events = [
                                {"type": event_type, "data": event_data, "timestamp": now}
                            ] + matches
                            
                            asyncio.create_task(pattern["action"](correlated_events, user_id))
                        except Exception as e:
                            logger.error(f"Error executing pattern action {pattern_id}: {str(e)}", exc_info=True)
                            
    async def _call_handlers(self, 
                          event_type: str, 
                          event_data: Dict[str, Any], 
                          user_id: Optional[str]):
        """
        Call all registered handlers for this event type.
        
        Args:
            event_type: Event type
            event_data: Event data
            user_id: User ID if applicable
        """
        if event_type not in self.event_handlers:
            return
            
        for handler in self.event_handlers[event_type]:
            try:
                asyncio.create_task(handler(event_data, user_id))
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {str(e)}", exc_info=True)

    # --- Condition Evaluation ---
    async def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        event_data: Dict[str, Any],
        user_id: Optional[str]
    ) -> bool:
        """
        Evaluate a condition against event data.
        
        Args:
            condition: Condition specification
            event_data: Event data to evaluate
            user_id: User ID if applicable
            
        Returns:
            True if condition is met, False otherwise
        """
        condition_type = condition.get("type", "simple")
        
        if condition_type == "simple":
            field = condition["field"]
            operator = condition["operator"]
            value = condition.get("value")
            
            # Get field value from event data
            field_value = event_data.get(field)
            
            # Handle 'any' operator (field exists)
            if operator == "any":
                return field_value is not None
                
            # Other operators require value comparison
            if field_value is None:
                return False
                
            if operator == "eq":
                return field_value == value
            elif operator == "neq":
                return field_value != value
            elif operator == "gt":
                return field_value > value
            elif operator == "gte":
                return field_value >= value
            elif operator == "lt":
                return field_value < value
            elif operator == "lte":
                return field_value <= value
            elif operator == "contains":
                return value in str(field_value)
            elif operator == "starts_with":
                return str(field_value).startswith(str(value))
                
        elif condition_type == "compound":
            subconditions = condition.get("conditions", [])
            operator = condition.get("operator", "and")
            
            if not subconditions:
                return True
                
            results = []
            for subcond in subconditions:
                result = await self._evaluate_condition(subcond, event_data, user_id)
                results.append(result)
                
            if operator == "and":
                return all(results)
            else:  # or
                return any(results)
                
        elif condition_type == "threshold":
            event_type = condition["event_type"]
            threshold = condition["threshold"]
            time_window_seconds = condition["time_window_seconds"]
            
            # Count events in time window
            now = utc_now()
            cutoff = now - timedelta(seconds=time_window_seconds)
            
            if event_type not in self.event_history:
                return False
                
            count = len([
                e for e in self.event_history[event_type]
                if ensure_utc(e["timestamp"]) >= cutoff and (not user_id or e["user_id"] == user_id)
            ])
            
            return count >= threshold
            
        elif condition_type == "custom":
            evaluator = condition.get("evaluator")
            if callable(evaluator):
                try:
                    result = evaluator(event_data, user_id)
                    if asyncio.iscoroutine(result):
                        return await result
                    return bool(result)
                except Exception as e:
                    logger.error(f"Error in custom condition evaluator: {e}", exc_info=True)
                    return False
            return False
            
        return False

    async def _check_sequence_pattern(
        self,
        pattern_id: str,
        pattern: Dict[str, Any],
        event_type: str,
        event_data: Dict[str, Any],
        user_id: Optional[str]
    ):
        """
        Check for a sequence pattern match.
        
        Tracks partial matches and triggers action when complete sequence is detected.
        
        Args:
            pattern_id: Pattern ID
            pattern: Pattern definition
            event_type: Current event type
            event_data: Current event data
            user_id: User ID if applicable
        """
        config = pattern["config"]
        sequence = config.get("sequence", [])
        timeout_seconds = config.get("timeout_seconds", 3600)
        
        if not sequence:
            return
            
        now = utc_now()
        cutoff = now - timedelta(seconds=timeout_seconds)
        
        # Get or create partial matches list for this pattern
        partial_matches = pattern.get("partial_matches", [])
        
        # Clean up expired partial matches
        partial_matches = [
            match for match in partial_matches
            if ensure_utc(match.get("started_at", now)) >= cutoff
        ]
        
        # Check if this event matches the next step in any partial sequence
        updated_matches = []
        matched_complete = False
        complete_events = None
        
        for match in partial_matches:
            next_step_idx = match.get("next_step", 0)
            
            if next_step_idx >= len(sequence):
                continue  # Already complete, shouldn't happen
                
            next_step = sequence[next_step_idx]
            step_type = next_step.get("type")
            step_condition = next_step.get("condition")
            
            # Check if current event matches this step
            if event_type == step_type:
                condition_met = await self._evaluate_condition(step_condition, event_data, user_id)
                
                if condition_met:
                    # Advance the match
                    match["events"].append({
                        "type": event_type,
                        "data": event_data,
                        "timestamp": now
                    })
                    match["next_step"] = next_step_idx + 1
                    
                    # Check if sequence is complete
                    if match["next_step"] >= len(sequence):
                        matched_complete = True
                        complete_events = match["events"]
                    else:
                        updated_matches.append(match)
                else:
                    # Keep the match for other events
                    updated_matches.append(match)
            else:
                # Event type doesn't match, keep the match
                updated_matches.append(match)
        
        # Check if this event starts a new sequence
        first_step = sequence[0]
        if event_type == first_step.get("type"):
            first_condition = first_step.get("condition")
            if await self._evaluate_condition(first_condition, event_data, user_id):
                new_match = {
                    "started_at": now,
                    "next_step": 1,
                    "events": [{
                        "type": event_type,
                        "data": event_data,
                        "timestamp": now
                    }],
                    "user_id": user_id
                }
                
                # Check if single-step sequence
                if len(sequence) == 1:
                    matched_complete = True
                    complete_events = new_match["events"]
                else:
                    updated_matches.append(new_match)
        
        # Update partial matches
        pattern["partial_matches"] = updated_matches
        
        # Trigger action if complete sequence found
        if matched_complete and complete_events:
            try:
                logger.info(f"Pattern {pattern_id} sequence completed with {len(complete_events)} events")
                pattern["last_triggered"] = now
                asyncio.create_task(pattern["action"](complete_events, user_id))
            except Exception as e:
                logger.error(f"Error executing sequence pattern action {pattern_id}: {e}", exc_info=True)

    # --- Validation helpers ---
    def _validate_condition_spec(self, condition: Dict[str, Any], context: str = "condition") -> None:
        if condition is None:
            raise ValueError(f"{context}: condition specification is required")

        condition_type = condition.get("type", "simple")

        if condition_type == "simple":
            required_keys = {"field", "operator"}
            missing = [key for key in required_keys if key not in condition]
            if missing:
                raise ValueError(f"{context}: missing keys for simple condition: {missing}")

            operator = condition["operator"]
            allowed_ops = {"eq", "neq", "gt", "gte", "lt", "lte", "contains", "starts_with", "any"}
            if operator not in allowed_ops:
                raise ValueError(f"{context}: unsupported operator '{operator}' for simple condition")

            if operator != "any" and "value" not in condition:
                raise ValueError(f"{context}: simple condition with operator '{operator}' requires 'value'")

        elif condition_type == "compound":
            subconditions = condition.get("conditions")
            if not subconditions or not isinstance(subconditions, list):
                raise ValueError(f"{context}: compound condition requires a non-empty 'conditions' list")

            operator = condition.get("operator", "and")
            if operator not in {"and", "or"}:
                raise ValueError(f"{context}: compound condition operator must be 'and' or 'or'")

            for idx, subcond in enumerate(subconditions):
                self._validate_condition_spec(subcond, context=f"{context} -> subcondition[{idx}]")

        elif condition_type == "threshold":
            if "event_type" not in condition:
                raise ValueError(f"{context}: threshold condition requires 'event_type'")
            threshold = condition.get("threshold")
            if threshold is None or not isinstance(threshold, (int, float)):
                raise ValueError(f"{context}: threshold condition requires numeric 'threshold'")
            if threshold <= 0:
                raise ValueError(f"{context}: threshold condition must have threshold > 0")
            if "time_window_seconds" not in condition or condition["time_window_seconds"] <= 0:
                raise ValueError(f"{context}: threshold condition requires positive 'time_window_seconds'")

        elif condition_type == "custom":
            evaluator = condition.get("evaluator")
            if not callable(evaluator):
                raise ValueError(f"{context}: custom condition requires callable 'evaluator'")

        else:
            raise ValueError(f"{context}: unsupported condition type '{condition_type}'")

    def _validate_pattern_spec(self, pattern_id: str, event_types: List[str], pattern_config: Dict[str, Any]) -> None:
        if not event_types:
            raise ValueError(f"pattern '{pattern_id}': must specify at least one event type")

        if pattern_config is None:
            raise ValueError(f"pattern '{pattern_id}': configuration is required")

        pattern_type = pattern_config.get("type", "sequence")

        if pattern_type == "sequence":
            sequence = pattern_config.get("sequence")
            if not sequence or not isinstance(sequence, list):
                raise ValueError(f"pattern '{pattern_id}': sequence pattern requires a non-empty 'sequence' list")

            for idx, step in enumerate(sequence):
                step_type = step.get("type")
                if not step_type:
                    raise ValueError(f"pattern '{pattern_id}': sequence step[{idx}] missing 'type'")
                if step_type not in event_types:
                    raise ValueError(f"pattern '{pattern_id}': sequence step[{idx}] type '{step_type}' not declared in event_types")
                condition = step.get("condition")
                if condition is None:
                    raise ValueError(f"pattern '{pattern_id}': sequence step[{idx}] missing 'condition'")
                self._validate_condition_spec(condition, context=f"pattern '{pattern_id}' sequence step[{idx}]")

            timeout = pattern_config.get("timeout_seconds", 0)
            if timeout <= 0:
                raise ValueError(f"pattern '{pattern_id}': sequence pattern requires positive 'timeout_seconds'")

        elif pattern_type == "frequency":
            thresholds = pattern_config.get("thresholds")
            if not thresholds or not isinstance(thresholds, dict):
                raise ValueError(f"pattern '{pattern_id}': frequency pattern requires 'thresholds' map")
            for et, threshold in thresholds.items():
                if et not in event_types:
                    raise ValueError(f"pattern '{pattern_id}': threshold defined for unknown event type '{et}'")
                if not isinstance(threshold, (int, float)) or threshold <= 0:
                    raise ValueError(f"pattern '{pattern_id}': threshold for event '{et}' must be positive number")

            window = pattern_config.get("time_window_seconds", 0)
            if window <= 0:
                raise ValueError(f"pattern '{pattern_id}': frequency pattern requires positive 'time_window_seconds'")

        elif pattern_type == "correlation":
            correlations = pattern_config.get("correlations")
            if not correlations or not isinstance(correlations, list):
                raise ValueError(f"pattern '{pattern_id}': correlation pattern requires 'correlations' list")

            for idx, correlation in enumerate(correlations):
                missing = [key for key in ("source_type", "target_type", "source_field", "target_field") if key not in correlation]
                if missing:
                    raise ValueError(f"pattern '{pattern_id}': correlation[{idx}] missing keys {missing}")
                if correlation["source_type"] not in event_types:
                    raise ValueError(f"pattern '{pattern_id}': correlation[{idx}] source_type '{correlation['source_type']}' not declared in event_types")
                if correlation["target_type"] not in event_types:
                    raise ValueError(f"pattern '{pattern_id}': correlation[{idx}] target_type '{correlation['target_type']}' not declared in event_types")

            window = pattern_config.get("time_window_seconds", 0)
            if window <= 0:
                raise ValueError(f"pattern '{pattern_id}': correlation pattern requires positive 'time_window_seconds'")

        else:
            raise ValueError(f"pattern '{pattern_id}': unsupported pattern type '{pattern_type}'")
                
    def get_event_history(self, 
                         event_type: Optional[str] = None,
                         user_id: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            limit: Maximum events to return
            
        Returns:
            List of event records
        """
        results = []
        
        if event_type:
            # Get events of a specific type
            if event_type in self.event_history:
                for event in self.event_history[event_type]:
                    if user_id and event["user_id"] != user_id:
                        continue
                    results.append(event)
        else:
            # Get all events
            for events in self.event_history.values():
                for event in events:
                    if user_id and event["user_id"] != user_id:
                        continue
                    results.append(event)
                    
        # Sort by timestamp descending
        results.sort(key=lambda e: e["timestamp"], reverse=True)
        
        # Apply limit
        return results[:limit]

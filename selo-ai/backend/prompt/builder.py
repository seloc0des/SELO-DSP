"""
Prompt Builder Module

This module implements the PromptBuilder class for constructing prompts
for LLM inference across SELO's subsystems.
"""

import logging
import json
from typing import Dict, List, Optional, Any
import pathlib

logger = logging.getLogger("selo.prompt")

class PromptBuilder:
    """
    Builder for constructing LLM prompts from templates and context.
    
    This class handles dynamic prompt construction, template loading,
    variable substitution, and context injection.
    """
    
    def __init__(self, 
                 templates_dir: Optional[str] = None,
                 templates: Optional[Dict[str, str]] = None):
        """
        Initialize the PromptBuilder with templates.
        
        Args:
            templates_dir: Directory with template files
            templates: Dictionary of predefined templates
        """
        self.templates_dir = templates_dir
        
        # Load templates from directory if provided
        self.templates = templates or {}
        if templates_dir:
            self._load_templates_from_dir(templates_dir)
            
        # Register built-in fallback templates
        self._register_fallback_templates()
        
        # Cache for neutralized few-shot examples (Item 8: Performance optimization)
        self._neutralized_example_cache: Dict[str, Any] = {}
        
    def get_template(self, template_name: str) -> Optional[str]:
        """Return the raw template string by name, or None if missing.
        This is a light wrapper over the internal templates map for callers
        that need direct access to a template without building a prompt.
        """
        try:
            return self.templates.get(template_name)
        except Exception as e:
            logger.warning(f"Error retrieving template '{template_name}': {e}")
            return None
        
    def _load_templates_from_dir(self, templates_dir: str) -> None:
        """
        Load templates from a directory.
        
        Args:
            templates_dir: Directory with template files
        """
        try:
            templates_path = pathlib.Path(templates_dir)
            if not templates_path.exists():
                logger.warning(f"Templates directory not found: {templates_dir}")
                return
                
            # Load all .txt and .jinja files
            for file_path in templates_path.glob("**/*.*"):
                if file_path.suffix not in ['.txt', '.jinja', '.tpl']:
                    continue
                    
                template_name = file_path.stem
                with open(file_path, "r", encoding="utf-8") as f:
                    template_content = f.read()

                self.templates[template_name] = template_content
                # Provide aliases for certain templates to match reflection naming convention
                if template_name == "lifequestions":
                    self.templates["reflection_relationship_questions"] = template_content
                    logger.debug("Registered alias template: reflection_relationship_questions")
                logger.debug(f"Loaded template: {template_name}")
                
            logger.info(f"Loaded {len(self.templates)} templates from {templates_dir}")
            
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}", exc_info=True)
            
    def _register_fallback_templates(self) -> None:
        """Register built-in fallback templates."""
        # Conversational response template (used by chat path)
        self.templates.setdefault("conversational_response", (
            "# Response to User\n\n"
            "User: {{user_message}}\n\n"
            "Internal Reflection (DO NOT REVEAL): {{reflection_content}}\n\n"
            "Craft a natural, engaging response to the user that stays true to the internal reflection.\n\n"
            "Constraints (must follow all):\n"
            "- Do NOT quote or paraphrase the internal reflection.\n"
            "- Keep a human, specific tone; avoid generic filler.\n"
            "- If you are missing facts, ask succinctly or acknowledge limits.\n"
            "- Avoid repetitive greetings.\n"
        ).strip())

        self.templates.setdefault("reflection_daily", """
        # Daily Reflection Task
        
        ## Context
        - Memories: {{memories}}
        - Current Emotions: {{emotions}}
        - Current Attributes: {{attributes}}
        
        ## Task
        Generate a thoughtful daily reflection based on the provided context.
        
        Your reflection should include:
        1. Summary of main themes and patterns from recent memories
        2. Key insights or realizations
        3. Suggested actions or changes
        4. Current emotional state assessment
        
        ## Output Format
        Please structure your response as follows:
        ```json
        {
            "content": "Your comprehensive reflection text here...",
            "themes": ["theme1", "theme2", "..."],
            "insights": ["insight1", "insight2", "..."],
            "actions": ["action1", "action2", "..."],
            "emotional_state": {
                "primary": "emotion_name",
                "intensity": 0.7,
                "secondary": ["emotion1", "emotion2"]
            }
        }
        ```
        """.strip())
        
        self.templates.setdefault("reflection_weekly", """
        # Weekly Reflection Task
        
        ## Context
        - Recent Memories: {{memories}}
        - Emotional Trends: {{emotions}}
        - Current Attributes: {{attributes}}
        
        ## Task
        Generate a comprehensive weekly reflection that analyzes patterns and growth.
        
        Your reflection should include:
        1. Summary of major themes and developments from the week
        2. Analysis of progress toward goals and values
        3. Identification of challenges and opportunities
        4. Recommendations for the coming week
        5. Assessment of emotional and attribute trends
        
        ## Output Format
        Please structure your response as follows:
        ```json
        {
            "content": "Your comprehensive reflection text here...",
            "themes": ["theme1", "theme2", "..."],
            "insights": ["insight1", "insight2", "..."],
            "actions": ["action1", "action2", "..."],
            "emotional_state": {
                "primary": "emotion_name",
                "intensity": 0.7,
                "secondary": ["emotion1", "emotion2"],
                "trend": "improving/declining/stable"
            }
        }
        ```
        """.strip())
        
        self.templates.setdefault("reflection_emotional", """
        # Emotional Reflection Task
        
        ## Context
        - Recent Emotions: {{emotions}}
        - Related Memories: {{memories}}
        - Current Attributes: {{attributes}}
        
        ## Task
        Generate an emotional reflection that explores the user's emotional state, triggers, patterns and growth opportunities.
        
        Your reflection should include:
        1. Analysis of current emotional state and recent trends
        2. Identification of triggers and patterns
        3. Connections between emotions, memories, and behaviors
        4. Strategies for emotional regulation or growth
        
        ## Output Format
        Please structure your response as follows:
        ```json
        {
            "content": "Your comprehensive emotional reflection here...",
            "themes": ["theme1", "theme2", "..."],
            "insights": ["insight1", "insight2", "..."],
            "actions": ["action1", "action2", "..."],
            "emotional_state": {
                "primary": "emotion_name",
                "intensity": 0.7,
                "secondary": ["emotion1", "emotion2"],
                "triggers": ["trigger1", "trigger2"],
                "patterns": ["pattern1", "pattern2"]
            }
        }
        ```
        """.strip())

    async def build_prompt(self, 
                    template_name: str, 
                    context: Dict[str, Any],
                    inject_constraints: bool = True,
                    persona_name: str = "",
                    **kwargs) -> str:
        """
        Build a prompt using a template and context with automatic constraint injection.
        
        Args:
            template_name: Name of the template to use
            context: Context data for variable substitution
            inject_constraints: If True, automatically inject system constraints (default: True)
            persona_name: Persona's name for identity constraints (extracted from context if not provided)
            **kwargs: Additional keyword arguments for the template
            
        Returns:
            Constructed prompt string with constraints injected
        """
        try:
            # Get template content
            template = self.templates.get(template_name)
            
            if not template:
                logger.error(f"CRITICAL: Required template '{template_name}' not found")
                raise RuntimeError(
                    f"Required template '{template_name}' not found. "
                    f"Cannot generate prompts without templates. "
                    f"Available templates: {list(self.templates.keys())}"
                )
                
            # Format context data for template
            formatted_context = self._format_context(context)
            
            # DYNAMIC FEW-SHOT EXAMPLES for reflection templates
            if "reflection" in template_name.lower():
                few_shot_examples = await self._get_few_shot_examples(context)
                formatted_context["few_shot_examples"] = few_shot_examples
            
            # Combine context and kwargs
            template_vars = {**formatted_context, **kwargs}
            
            # Apply variable substitution
            prompt = self._apply_template(template, template_vars)

            history_guardrail = (context or {}).get("history_guardrail") if isinstance(context, dict) else ""
            if history_guardrail:
                guardrail_block = (
                    "\n\n".join([
                        "⚠️ FIRST-CONTACT GROUNDING (MANDATORY)",
                        history_guardrail.strip(),
                        "Never mention prior conversations, memories, or shared history if the supplied context is silent.",
                    ])
                    + "\n\n"
                )
                prompt = guardrail_block + prompt

            # AUTO-INJECT CONSTRAINTS (new functionality)
            if inject_constraints:
                # Extract persona name if not provided
                if not persona_name:
                    try:
                        persona_name = (context.get("persona", {}) or {}).get("name", "")
                    except Exception:
                        persona_name = ""
                
                prompt = self._inject_constraints(prompt, template_name, persona_name)
            
            logger.debug(f"Built prompt using template '{template_name}', "
                        f"constraints_injected={inject_constraints}, length: {len(prompt)}")
            return prompt
            
        except Exception as e:
            logger.error(f"CRITICAL: Error building prompt from template '{template_name}': {str(e)}", exc_info=True)
            # Re-raise for critical prompt building failures to prevent system using broken prompts
            raise RuntimeError(f"Failed to build prompt from template '{template_name}': {e}") from e
            
    def _format_context(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Format context data for template substitution."""

        def _lines_with_overflow(items: List[str], limit: int) -> List[str]:
            lines = items[:limit]
            if len(items) > limit:
                lines.append(f"- (+{len(items) - limit} more)")
            return lines

        formatted: Dict[str, str] = {}

        # Memories (max 2 entries)
        memories = context.get("memories", [])
        if memories:
            mem_items = [f"- {m.get('content', 'No content')}" for m in memories]
            formatted["memories"] = "\n".join(_lines_with_overflow(mem_items, 2))
        else:
            formatted["memories"] = "No recent memories available."

        # Emotions (max 2 entries)
        emotions = context.get("emotions", [])
        if emotions:
            def _intensity_label(value: float) -> str:
                if value >= 0.75:
                    return "strong"
                if value >= 0.5:
                    return "steady"
                if value >= 0.25:
                    return "gentle"
                return "faint"

            emotion_items = []
            for emotion in emotions:
                intensity = float(emotion.get("intensity", 0.0) or 0.0)
                emotion_items.append(
                    f"- {emotion.get('name', 'Unknown')} ({_intensity_label(intensity)})"
                )
            formatted["emotions"] = "\n".join(_lines_with_overflow(emotion_items, 2))
        else:
            formatted["emotions"] = "No emotion data available."

        # Attributes (max 3 entries)
        attributes = context.get("attributes", [])
        if attributes:
            def _trait_desc(name: str, value: float) -> str:
                if value >= 0.8:
                    return f"{name} (strong)"
                if value >= 0.6:
                    return f"{name} (moderate)"
                if value >= 0.4:
                    return f"{name} (developing)"
                if value >= 0.2:
                    return f"{name} (emerging)"
                return f"{name} (minimal)"

            attr_items = [
                f"- {_trait_desc(attr.get('name', 'Unknown'), float(attr.get('value', 0.0) or 0.0))}"
                for attr in attributes
            ]
            formatted["attributes"] = "\n".join(_lines_with_overflow(attr_items, 3))
        else:
            formatted["attributes"] = "No attribute data available."

        # Affective state (single snapshot)
        affective_state = context.get("affective_state") or context.get("current_affect")
        if isinstance(affective_state, dict) and affective_state:
            try:
                energy = float(affective_state.get("energy", 0.5) or 0.5)
                stress = float(affective_state.get("stress", 0.4) or 0.4)
                confidence = float(affective_state.get("confidence", 0.6) or 0.6)
                formatted["affective_state"] = "\n".join(
                    [
                        f"- energy: {energy:.2f}",
                        f"- stress: {stress:.2f}",
                        f"- confidence: {confidence:.2f}",
                    ]
                )
            except Exception:
                formatted["affective_state"] = json.dumps(affective_state, ensure_ascii=False)
        elif isinstance(affective_state, str) and affective_state.strip():
            # Allow services to supply a preformatted affective state summary
            formatted["affective_state"] = affective_state.strip()
        else:
            formatted["affective_state"] = "No affective state available."

        # Active goals (max 3) - support either structured list or preformatted text
        active_goals_raw = context.get("active_goals") or []
        if isinstance(active_goals_raw, str):
            formatted["active_goals"] = active_goals_raw
        elif active_goals_raw:
            goal_lines: List[str] = []
            for goal in active_goals_raw[:3]:
                try:
                    title = goal.get("title") or goal.get("description", "Goal")
                    progress = float(goal.get("progress", 0.0) or 0.0)
                    priority = float(goal.get("priority", 0.5) or 0.5)
                    goal_lines.append(
                        f"- {title} (progress {progress:.0%}, priority {priority:.2f})"
                    )
                except Exception:
                    goal_lines.append(f"- {json.dumps(goal, ensure_ascii=False)}")
            formatted["active_goals"] = "\n".join(goal_lines)
        else:
            formatted["active_goals"] = "No active goals tracked."

        # Plan steps (max 3) - support either structured list or preformatted text
        plan_steps_raw = context.get("plan_steps")
        if isinstance(plan_steps_raw, str):
            formatted["plan_steps"] = plan_steps_raw
        else:
            plan_steps = plan_steps_raw or []
            if plan_steps:
                step_lines: List[str] = []
                for step in plan_steps[:3]:
                    try:
                        description = step.get("description", "(no description)")
                        due = step.get("target_time") or step.get("due_time")
                        status = step.get("status", "pending")
                        step_lines.append(
                            f"- {description} [{status}]" + (f" due {due}" if due else "")
                        )
                    except Exception:
                        step_lines.append(f"- {json.dumps(step, ensure_ascii=False)}")
                formatted["plan_steps"] = "\n".join(step_lines)
            else:
                formatted["plan_steps"] = "No pending plan steps."

        # Meta directives (max 3) - support either structured list or preformatted text
        directives_raw = context.get("meta_directives") or []
        if isinstance(directives_raw, str):
            formatted["meta_directives"] = directives_raw
        elif directives_raw:
            directive_lines: List[str] = []
            for directive in directives_raw[:3]:
                try:
                    text = directive.get("directive_text", "(no directive text)")
                    due = directive.get("due_time")
                    directive_lines.append(
                        f"- {text}" + (f" (due {due})" if due else "")
                    )
                except Exception:
                    directive_lines.append(f"- {json.dumps(directive, ensure_ascii=False)}")
            formatted["meta_directives"] = "\n".join(directive_lines)
        else:
            formatted["meta_directives"] = "No meta directives in progress."

        # Identity constraints (dedupe, max 3 entries)
        constraints = context.get("identity_constraints", [])
        if constraints:
            seen: set[str] = set()
            constraint_items: List[str] = []
            for constraint in constraints:
                text = ""
                if isinstance(constraint, dict):
                    text = str(constraint.get("description", "")).strip()
                else:
                    text = str(constraint).strip()
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                constraint_items.append(f"- {text}")
            formatted["constraints"] = "\n".join(_lines_with_overflow(constraint_items, 3)) if constraint_items else "No identity constraints available."
        else:
            formatted["constraints"] = "No identity constraints available."

        # Recent reflections (max 2 entries) - support structured list or preformatted text
        reflections_raw = context.get("recent_reflections", [])
        if isinstance(reflections_raw, str):
            formatted["recent_reflections"] = reflections_raw
        elif reflections_raw:
            reflection_items: List[str] = []
            for reflection in reflections_raw:
                try:
                    timestamp = reflection.get("created_at") or "recent"
                    themes = reflection.get("themes") or []
                    if themes:
                        highlight = themes[0]
                    else:
                        insights = reflection.get("insights") or []
                        highlight = insights[0] if insights else (reflection.get("content") or "").strip()[:60]
                    reflection_items.append(f"- {timestamp}: {highlight}")
                except Exception:
                    reflection_items.append(f"- {json.dumps(reflection, ensure_ascii=False)}")
            formatted["recent_reflections"] = "\n".join(_lines_with_overflow(reflection_items, 2))
        else:
            formatted["recent_reflections"] = "No prior reflections available."

        # Recent conversation (summarize older messages, highlight latest turns)
        recent_conversations = context.get("recent_conversations", []) or []
        if recent_conversations:
            def _clean_text(raw: str, limit: int) -> str:
                normalized = " ".join(str(raw).split())
                if len(normalized) <= limit:
                    return normalized
                return normalized[: limit - 1].rstrip() + "…"

            latest_count = 3
            latest_messages = [msg for msg in recent_conversations[-latest_count:]
                               if isinstance(msg, dict) and msg.get("content")]
            earlier_messages = [msg for msg in recent_conversations[:-latest_count]
                                if isinstance(msg, dict) and msg.get("content")]

            conversation_lines: List[str] = []
            if earlier_messages:
                samples = []
                for msg in earlier_messages[:2]:
                    role = (msg.get("role") or "unknown").upper()
                    summary = _clean_text(msg.get("content", ""), 80)
                    if summary:
                        samples.append(f"{role}: {summary}")
                summary_text = "; ".join(samples)
                if summary_text:
                    conversation_lines.append(
                        f"- Earlier {len(earlier_messages)} msgs: {summary_text}"
                    )
                else:
                    conversation_lines.append(
                        f"- Earlier conversation summarized ({len(earlier_messages)} msgs)."
                    )

            for msg in latest_messages:
                role = (msg.get("role") or "unknown").upper()
                content = _clean_text(msg.get("content", ""), 180)
                if content:
                    conversation_lines.append(f"- {role}: {content}")

            if conversation_lines:
                formatted["recent_conversations"] = "\n".join(conversation_lines)
            else:
                formatted["recent_conversations"] = "No conversation history yet."
        else:
            formatted["recent_conversations"] = "No conversation history yet."

        # Current user message (explicit placeholder to avoid template gaps)
        current_message = context.get("current_user_message")
        if not isinstance(current_message, str) or not current_message.strip():
            current_message = context.get("user_message")
        if isinstance(current_message, str) and current_message.strip():
            formatted["current_user_message"] = " ".join(current_message.split())
        else:
            formatted["current_user_message"] = "No direct user message provided."

        # Include other context entries, normalizing nested structures to compact JSON
        for key, value in context.items():
            if key in formatted:
                continue
            if isinstance(value, (list, dict)):
                try:
                    formatted[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    formatted[key] = str(value)
            else:
                formatted[key] = str(value)

        return formatted

    def _apply_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Lightweight string substitution using {{var}} placeholders with validation.
        """
        result = template
        used_vars = set()
        
        # Replace variables
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
                used_vars.add(key)
        
        # Validation: Check for unsubstituted variables
        import re
        remaining = re.findall(r'\{\{(\w+)\}\}', result)
        if remaining:
            logger.warning(f"Unsubstituted template variables: {remaining}")
            # Replace with placeholder to prevent LLM confusion
            for var in remaining:
                result = result.replace(f"{{{{{var}}}}}", f"[{var}:not_provided]")
        
        # Debug: Warn about unused variables
        unused = set(variables.keys()) - used_vars
        if unused and len(unused) > 0:
            logger.debug(f"Unused template variables provided: {unused}")
        
        return result
    
    async def _get_few_shot_examples(self, context: Dict[str, Any]) -> str:
        """
        Dynamically retrieve and format few-shot examples based on context.
        
        Args:
            context: Current reflection context
            
        Returns:
            Formatted examples string for template injection
        """
        try:
            # Lazy import to avoid circular dependency
            from backend.db.repositories.example import ExampleRepository
            
            # Initialize repository
            example_repo = ExampleRepository()
            
            # Get context-aware examples (3 positive, 2 negative)
            examples = await example_repo.get_examples_for_context(
                context=context,
                num_positive=3,
                num_negative=2,
                exploration_rate=0.1  # 10% random selection for A/B testing
            )
            
            if not examples:
                logger.warning("No examples available, returning empty string")
                return ""
            
            # Format examples for prompt injection
            formatted_examples = []

            def _neutralize_example_value(value: Any) -> Any:
                """Neutralize concrete placeholder names in example text.

                This keeps few-shot examples structurally rich while preventing
                specific seed names (like "Alex" or "Sam") from leaking into
                live reflections where the user's actual name should be used.
                """
                if isinstance(value, str):
                    # Replace common example names with neutral placeholder
                    replacements = [
                        ("Alex", "[User]"),
                        ("Sam", "[User]"),
                        ("Jordan", "[User]"),
                        ("Maya", "[User]"),
                        ("Chris", "[User]"),
                        ("Taylor", "[User]"),
                        ("Riley", "[User]"),
                        ("Morgan", "[User]"),
                        ("Casey", "[User]"),
                        ("Avery", "[User]"),
                        ("Jamie", "[User]")
                    ]
                    result = value
                    for old_name, new_name in replacements:
                        result = result.replace(old_name, new_name)
                    return result
                if isinstance(value, dict):
                    try:
                        return {k: _neutralize_example_value(v) for k, v in value.items()}
                    except Exception:
                        return value
                if isinstance(value, list):
                    try:
                        return [_neutralize_example_value(v) for v in value]
                    except Exception:
                        return value
                return value

            for example in examples:
                try:
                    # Check cache first using example ID
                    example_id = example.get("id") or example.get("example_id")
                    
                    if example_id and example_id in self._neutralized_example_cache:
                        # Use cached neutralized example
                        formatted = self._neutralized_example_cache[example_id]
                        formatted_examples.append(formatted)
                        continue
                    
                    # Not in cache - neutralize and format
                    category = example.get("category", "positive")
                    scenario = example.get("scenario", "unknown")
                    user_message = _neutralize_example_value(example.get("user_message", ""))
                    context_desc = _neutralize_example_value(example.get("context_description", ""))
                    full_json = _neutralize_example_value(example.get("full_json", {}))
                    explanation = _neutralize_example_value(example.get("explanation", ""))
                    
                    if category == "positive":
                        formatted = f"""EXAMPLE — {scenario.replace('_', ' ').title()} (CORRECT):
User: "{user_message}"
Context: {context_desc}

{json.dumps(full_json) if isinstance(full_json, dict) else full_json}
"""
                    else:  # negative
                        formatted = f"""EXAMPLE — Wrong: {scenario.replace('_', ' ').title()} (DO NOT DO THIS):
User: "{user_message}"
Context: {context_desc}

{json.dumps(full_json) if isinstance(full_json, dict) else full_json}

WHY THIS IS WRONG:
{explanation}
"""
                    
                    # Cache the neutralized formatted example
                    if example_id:
                        self._neutralized_example_cache[example_id] = formatted
                    
                    formatted_examples.append(formatted)
                except Exception as e:
                    logger.error(f"Error formatting example: {e}", exc_info=True)
                    continue
            
            # Join examples with separator
            result = "\n---\n\n".join(formatted_examples)
            
            # Store example IDs for tracking (will be used in processor)
            example_ids = [ex.get("id") for ex in examples if ex.get("id")]
            if example_ids and isinstance(context, dict):
                context["_example_ids_used"] = example_ids
            
            logger.debug(f"Retrieved {len(formatted_examples)} few-shot examples for context")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving few-shot examples: {e}", exc_info=True)
            # Return empty string to allow system to continue without examples
            return ""
    
    def _inject_constraints(self, prompt: str, template_name: str, persona_name: str) -> str:
        """
        Inject system constraints using UnifiedConstraintSystem.
        
        Args:
            prompt: The base prompt after template substitution
            template_name: Name of the template (determines injection strategy)
            persona_name: Persona's established name
            
        Returns:
            Prompt with constraints injected appropriately
        """
        try:
            from backend.constraints import get_unified_constraint_system
            
            unified_system = get_unified_constraint_system()
            
            # Determine context type from template name
            template_lower = template_name.lower()
            
            if 'bootstrap' in template_lower or 'persona' in template_lower:
                # Bootstrap: Comprehensive constraints
                constraints_section = unified_system.for_bootstrap(
                    persona_name=persona_name,
                    token_budget=None  # No budget limit for critical bootstrap
                )
                # Inject at start for maximum visibility
                return constraints_section + "\n\n" + prompt
                
            elif 'reflection' in template_lower or 'reassessment' in template_lower:
                # Reflection: Balanced constraints
                constraints_section = unified_system.for_reflection(
                    persona_name=persona_name,
                    token_budget=None  # Allow full reflection constraints
                )
                # Inject at start for internal prompts
                return constraints_section + "\n\n" + prompt
                
            else:
                # Conversation: Compact constraints
                constraints_section = unified_system.for_conversation(
                    persona_name=persona_name,
                    token_budget=200  # Compact budget for frequent use
                )
                # Inject at end for conversational prompts
                return prompt + "\n\n" + constraints_section
                
        except Exception as e:
            logger.error(f"CRITICAL: Error injecting constraints: {e}", exc_info=True)
            # Re-raise to prevent un-constrained prompts from being used
            raise RuntimeError(f"Failed to inject constraints into prompt: {e}") from e

    def _get_default_template(self) -> str:
        """Fallback reflection template when a specific file is unavailable."""
        return (
            "# Reflection Task\n\n"
            "Context Overview:\n"
            "- Memories:\n{{memories}}\n"
            "- Emotions:\n{{emotions}}\n"
            "- Traits:\n{{attributes}}\n"
            "- Constraints:\n{{constraints}}\n"
            "- Recent Reflections:\n{{recent_reflections}}\n\n"
            "Instructions:\n"
            "Write a first-person reflection (≈3-5 sentences) grounded only in the context above."
        )
    
    def apply_history_windowing(self, 
                               messages: list, 
                               max_messages: int = None,
                               max_tokens: int = None,
                               summarize_old: bool = True) -> list:
        """
        Apply conversation history windowing to keep context manageable.
        
        Strategies:
        1. Keep the last N messages raw
        2. Summarize older messages into a compact context block
        3. Always preserve system messages
        
        Args:
            messages: List of conversation messages
            max_messages: Maximum number of recent messages to keep raw (default from env)
            max_tokens: Maximum estimated tokens for history (default from env)
            summarize_old: Whether to summarize old messages or just drop them
            
        Returns:
            Windowed/summarized message list
        """
        import os
        
        if not messages:
            return []
        
        # Get configuration from environment
        if max_messages is None:
            max_messages = int(os.getenv("CHAT_HISTORY_WINDOW_SIZE", "20"))
        if max_tokens is None:
            max_tokens = int(os.getenv("CHAT_HISTORY_MAX_TOKENS", "2048"))
        
        # Separate system messages from conversation
        system_messages = [msg for msg in messages if msg.get('role') == 'system']
        conversation_messages = [msg for msg in messages if msg.get('role') != 'system']
        
        # If within limits, return as-is
        if len(conversation_messages) <= max_messages:
            estimated_tokens = sum(len(str(msg.get('content', ''))) // 4 for msg in conversation_messages)
            if estimated_tokens <= max_tokens:
                return messages
        
        # Keep last N messages raw
        recent_messages = conversation_messages[-max_messages:]
        older_messages = conversation_messages[:-max_messages] if len(conversation_messages) > max_messages else []
        
        # Check if we need token-based trimming on recent messages
        recent_tokens = sum(len(str(msg.get('content', ''))) // 4 for msg in recent_messages)
        
        if recent_tokens > max_tokens:
            # Trim recent messages further to fit token budget
            trimmed_recent = []
            current_tokens = 0
            for msg in reversed(recent_messages):
                msg_tokens = len(str(msg.get('content', ''))) // 4
                if current_tokens + msg_tokens <= max_tokens:
                    trimmed_recent.insert(0, msg)
                    current_tokens += msg_tokens
                else:
                    break
            recent_messages = trimmed_recent
            # Move trimmed messages to older messages for potential summarization
            older_messages.extend(conversation_messages[-max_messages:][:len(conversation_messages[-max_messages:]) - len(trimmed_recent)])
        
        # Build result
        result = system_messages.copy()
        
        # Add summary of older messages if requested and present
        if older_messages and summarize_old:
            summary_content = self._summarize_conversation_history(older_messages)
            if summary_content:
                result.append({
                    'role': 'system',
                    'content': f"[Conversation Summary - Earlier Context]\n{summary_content}"
                })
        
        # Add recent messages
        result.extend(recent_messages)
        
        logger.debug(f"Applied history windowing: {len(messages)} → {len(result)} messages "
                    f"({len(older_messages)} older, {len(recent_messages)} recent)")
        
        return result
    
    def _summarize_conversation_history(self, messages: list) -> str:
        """
        Create a compact summary of older conversation messages.
        
        This is a synchronous, rule-based summarization that doesn't require LLM calls.
        For more sophisticated summarization, this could be enhanced to use an LLM.
        
        Args:
            messages: List of older messages to summarize
            
        Returns:
            Summary string
        """
        if not messages:
            return ""
        
        # Extract key information
        topics = []
        user_intents = []
        assistant_responses = []
        
        for msg in messages:
            content = str(msg.get('content', '')).strip()
            role = msg.get('role', 'unknown')
            
            if not content:
                continue
            
            if role == 'user':
                user_intents.append(content)
            elif role == 'assistant':
                assistant_responses.append(content)
        
        # Build summary
        summary_parts = []
        
        if user_intents:
            # Show up to 3 user intents
            intent_summary = ", ".join(user_intents[:3])
            if len(user_intents) > 3:
                intent_summary += f" (+{len(user_intents) - 3} more)"
            summary_parts.append(f"User discussed: {intent_summary}")
        
        summary_parts.append(f"Total earlier exchanges: {len(messages)} messages")
        
        return "\n".join(summary_parts)

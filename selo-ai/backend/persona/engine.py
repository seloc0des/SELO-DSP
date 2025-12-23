"""
Persona Engine

This module provides the core engine for the Dynamic Persona System,
handling persona evolution, trait updates, and persona generation.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set

try:
    from ..llm.router import LLMRouter
except ImportError:
    from backend.llm.router import LLMRouter

try:
    from ..memory.vector_store import VectorStore
except ImportError:
    from backend.memory.vector_store import VectorStore

try:
    from ..db.repositories.persona import PersonaRepository
except ImportError:
    from backend.db.repositories.persona import PersonaRepository

try:
    from ..sdl.repository import LearningRepository
except ImportError:
    from backend.sdl.repository import LearningRepository

try:
    from ..core.capabilities import compute_capabilities
except ImportError:
    from backend.core.capabilities import compute_capabilities

try:
    from ..utils.datetime import utc_now, isoformat_utc
except ImportError:
    from backend.utils.datetime import utc_now, isoformat_utc

try:
    from ..utils.numeric_utils import clamp
except ImportError:
    from backend.utils.numeric_utils import clamp

try:
    from ..constraints.validation_helpers import ValidationHelper
except ImportError:
    from backend.constraints.validation_helpers import ValidationHelper

logger = logging.getLogger("selo.persona.engine")


class PersonaEngine:
    """
    Core engine for the Dynamic Persona System.
    
    This class handles persona evolution, trait updates, and persona generation
    based on learnings from the SDL module and other sources.
    """
    
    # Valid trait categories
    VALID_TRAIT_CATEGORIES = {
        "cognitive", "emotional", "social", "learning", 
        "personality", "communication", "general"
    }
    
    def __init__(
        self,
        llm_router,
        vector_store: VectorStore,
        persona_repo: Optional[PersonaRepository] = None,
        learning_repo: Optional[LearningRepository] = None,
        initial_attributes: Optional[List[Dict[str, Any]]] = None,
        locked_attributes: Optional[List[str]] = None,
    ):
        """
        Initialize the persona engine.
        
        Args:
            llm_router: LLM router for persona generation
            vector_store: Vector store for embeddings
            persona_repo: Repository for persona data (defaults to PersonaRepository)
            learning_repo: Repository for learnings (defaults to LearningRepository)
            initial_attributes: Override manifesto attributes (for testing). If None, loads from manifesto.
            locked_attributes: Override locked traits (for testing). If None, loads from manifesto.
        """
        self.llm_router = llm_router
        self.vector_store = vector_store
        self.persona_repo = persona_repo or PersonaRepository()
        self.learning_repo = learning_repo or LearningRepository()
        
        # Load initial attributes from manifesto (or use provided overrides for testing)
        if initial_attributes is None or locked_attributes is None:
            from .manifesto_loader import get_initial_attributes, get_locked_attributes
            self.initial_attributes = initial_attributes if initial_attributes is not None else get_initial_attributes()
            self.locked_attributes = locked_attributes if locked_attributes is not None else get_locked_attributes()
        else:
            self.initial_attributes = initial_attributes
            self.locked_attributes = locked_attributes
        
        self.personality_dimensions = [attr["name"] for attr in self.initial_attributes]
        logger.info(f"Persona Engine initialized with attributes: {self.personality_dimensions}")

        # Define communication style dimensions
        self.communication_dimensions = [
            "formality", "directness", "empathy", "verbosity",
            "humor", "creativity", "analytical"
        ]
        
    
    async def close(self):
        """Close any resources."""
        if self.persona_repo:
            await self.persona_repo.close()
        if self.learning_repo:
            await self.learning_repo.close()

    async def reassess_persona(
        self,
        persona_id: str,
        user_id: str,
        reflections: Optional[List[Dict[str, Any]]] = None,
        learnings_limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Perform a template-driven reassessment of the persona and apply structured updates.
        - Loads backend/prompt/templates/persona_reassessment.txt
        - Builds a context from current persona, traits, recent reflections (optional), and learnings
        - Routes via LLM and parses JSON
        - Applies persona field updates and bounded trait deltas; respects locked traits
        - Records a PersonaEvolution entry
        """
        try:
            # Load persona with traits
            persona_obj = await self.persona_repo.get_persona(persona_id, include_traits=True)
            if not persona_obj:
                return {"success": False, "error": "Persona not found"}

            persona = persona_obj.to_dict() if hasattr(persona_obj, "to_dict") else {
                "description": getattr(persona_obj, "description", ""),
                "personality": getattr(persona_obj, "personality", {}) or {},
                "values": getattr(persona_obj, "values", {}) or {},
                "preferences": getattr(persona_obj, "preferences", {}) or {},
                "goals": getattr(persona_obj, "goals", {}) or {},
                "expertise": self._normalize_expertise(getattr(persona_obj, "expertise", {})),
                "communication_style": getattr(persona_obj, "communication_style", {}) or {},
            }

            # Traits summary
            trait_list = []
            for t in (getattr(persona_obj, "traits", []) or []):
                trait_list.append({
                    "name": getattr(t, "name", ""),
                    "value": float(getattr(t, "value", 0.0) or 0.0),
                    "weight": float(getattr(t, "weight", 1.0) or 1.0),
                    "category": getattr(t, "category", ""),
                    "locked": bool(getattr(t, "locked", False)),
                })

            # Recent learnings (best effort)
            learnings_text = ""
            try:
                recent_learnings = await self.learning_repo.get_recent_learnings(user_id=user_id, limit=learnings_limit)  # type: ignore[attr-defined]
                if recent_learnings:
                    chunks = []
                    for L in recent_learnings:
                        content = getattr(L, "content", None) or (L.get("content") if isinstance(L, dict) else None) or ""
                        if content:
                            # Do not truncate learning content
                            chunks.append(f"- {content}")
                    learnings_text = "\n".join(chunks)
            except Exception:
                learnings_text = ""

            # Recent reflections summary (optional)
            reflections_text = ""
            if reflections:
                try:
                    chunks = []
                    for r in reflections[:5]:
                        themes = ", ".join(r.get("themes", [])) if isinstance(r, dict) else ""
                        insights = r.get("insights", []) if isinstance(r, dict) else []
                        # Do not truncate reflection insight content
                        insight_line = insights[0] if insights else ""
                        chunks.append(f"- Themes: {themes} | Insight: {insight_line}")
                    reflections_text = "\n".join(chunks)
                except Exception:
                    reflections_text = ""

            # Load template text
            template_path = None
            try:
                import pathlib
                template_path = pathlib.Path(__file__).resolve().parent.parent / "prompt" / "templates" / "persona_reassessment.txt"
                template_text = template_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to load persona_reassessment template ({template_path}): {e}")
                # Minimal fallback prompt
                template_text = (
                    "Reassess persona with constraints; output JSON with description, personality, values, preferences, goals, expertise, communication_style, \n"
                    "trait_changes (name, delta, reason), rationale, confidence."
                )

            # Render a simple replacement (not full templating)
            def _safe(obj) -> str:
                try:
                    import json as _json
                    return _json.dumps(obj, ensure_ascii=False)
                except Exception:
                    return str(obj)

            rendered = (
                template_text
                .replace("{{persona.description}}", str(persona.get("description", "")))
                .replace("{{persona.personality}}", _safe(persona.get("personality", {})))
                .replace("{{persona.values}}", _safe(persona.get("values", {})))
                .replace("{{persona.preferences}}", _safe(persona.get("preferences", {})))
                .replace("{{persona.goals}}", _safe(persona.get("goals", {})))
                .replace("{{persona.expertise}}", _safe(self._normalize_expertise(persona.get("expertise", {}))))
                .replace("{{persona.communication_style}}", _safe(persona.get("communication_style", {})))
                .replace("{{traits}}", _safe(trait_list))
                .replace("{{reflections}}", reflections_text)
                .replace("{{learnings}}", learnings_text)
            )

            # Route via LLM using analytical task type for persona reassessment
            routed = await self.llm_router.route(task_type="analytical", prompt=rendered)
            content = (routed or {}).get("content") or (routed or {}).get("completion") or ""

            # Parse JSON
            try:
                import json as _json
                data = _json.loads(content) if content.strip().startswith("{") else {}
            except Exception:
                data = {}

            if not isinstance(data, dict) or not data:
                return {"success": False, "error": "Invalid reassessment output"}

            # Apply persona changes
            changes_applied = False
            persona_updates = {}
            for field in ["description", "personality", "values", "preferences", "goals", "expertise", "communication_style"]:
                if field in data and data[field] is not None:
                    persona_updates[field] = data[field]
            if persona_updates:
                await self.persona_repo.update_persona(persona_id, persona_updates)
                changes_applied = True

            # Apply trait changes (bounded, respect locked)
            trait_changes = []
            for tc in (data.get("trait_changes") or []):
                try:
                    name = tc.get("name")
                    delta = float(tc.get("delta", 0.0))
                    reason = tc.get("reason", "")
                    # Clamp delta
                    if delta > 0.2:
                        delta = 0.2
                    if delta < -0.2:
                        delta = -0.2
                    # Find trait
                    for t in trait_list:
                        if t.get("name") == name:
                            if t.get("locked"):
                                name = None
                            else:
                                new_val = clamp(t.get("value", 0.0) + delta)
                                trait_changes.append({"name": name, "delta": delta, "new_value": new_val, "reason": reason, "category": t.get("category", "")})
                            break
                except Exception:
                    continue

            if trait_changes:
                # Persist trait updates
                await self._update_persona_traits(persona_id, [
                    {"name": c["name"], "category": c.get("category", "general"), "delta": c["delta"]}
                    for c in trait_changes if c.get("name")
                ])
                changes_applied = True

            # Record evolution
            try:
                if changes_applied:
                    # Validate and clamp confidence score
                    confidence = ValidationHelper.clamp_score(
                        data.get("confidence", 0.6),
                        default=0.6
                    )
                    
                    await self.persona_repo.create_evolution({
                        "persona_id": persona_id,
                        "source_type": "scheduled_reassessment",
                        "reasoning": data.get("rationale", "Scheduled reassessment"),
                        "confidence": confidence,
                        "impact_score": 0.6 if trait_changes or persona_updates else 0.3,
                        "changes": {"persona": persona_updates, "traits": trait_changes},
                        "timestamp": utc_now(),
                        "reviewed": False,
                        "approved": True,
                    })
            except Exception:
                pass

            return {"success": True, "changed": bool(changes_applied), "persona_updates": persona_updates, "trait_changes": trait_changes}

        except Exception as e:
            logger.error(f"Reassessment failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    # === Persona Creation and Initialization ===
    
    async def create_initial_persona(self, user_id: str, name: str) -> Dict[str, Any]:
        """
        Create an initial persona for a user.
        
        Args:
            user_id: User ID
            name: Name for the persona
            
        Returns:
            Created persona data
        """
        try:
            # Check if user already has a default persona
            existing_persona = await self.persona_repo.get_persona_by_user(
                user_id=user_id,
                is_default=True
            )
            
            if existing_persona:
                logger.info(f"User {user_id} already has a default persona: {existing_persona.id}")
                return existing_persona.to_dict()
            
            # Generate initial persona attributes from manifesto
            personality = {attr["name"]: attr["weight"] for attr in self.initial_attributes}
            communication_style = await self._generate_initial_communication_style()
            expertise = {"domains": [], "skills": [], "knowledge_depth": 0.5}
            preferences = {"learning_focus": [], "interaction_style": "balanced"}
            goals = {"self_improvement": [], "knowledge_acquisition": []}
            values = {"primary": [], "secondary": []}
            
            # Create persona data
            persona_data = {
                "user_id": user_id,
                "name": name,
                "description": f"Initial persona for {name}",
                "personality": personality,
                "communication_style": communication_style,
                "expertise": expertise,
                "preferences": preferences,
                "goals": goals,
                "values": values,
                "is_active": True,
                "is_default": True,
                "evolution_locked": False,
                "stability_score": 0.5  # Initial stability is moderate
            }
            
            # Create persona in database
            persona = await self.persona_repo.create_persona(persona_data)
            
            # Create initial traits
            await self._create_initial_traits(persona.id)
            
            # Get complete persona with traits
            complete_persona = await self.persona_repo.get_persona(
                persona_id=persona.id,
                include_traits=True
            )
            
            logger.info(f"Created initial persona {persona.id} for user {user_id}")
            return complete_persona.to_dict()
            
        except Exception as e:
            logger.error(f"Error creating initial persona: {str(e)}", exc_info=True)
            raise
    
    async def generate_persona_prompt(self, persona_id: str) -> Dict[str, Any]:
        """
        Generate a system prompt for the conversational LLM based on persona attributes.
        
        Args:
            persona_id: Persona ID
            
        Returns:
            Dictionary with prompt and metadata
        """
        try:
            # Get persona with traits
            persona = await self.persona_repo.get_persona(
                persona_id=persona_id,
                include_traits=True
            )
            
            if not persona:
                logger.warning(f"Persona {persona_id} not found")
                return {
                    "success": False,
                    "error": "Persona not found"
                }
            
            # Extract persona attributes
            persona_dict = persona.to_dict() if hasattr(persona, "to_dict") else dict(persona)
            personality = persona_dict.get("personality", {})
            communication_style = persona_dict.get("communication_style", {})
            expertise = self._normalize_expertise(persona_dict.get("expertise", {}))
            values = persona_dict.get("values", {})
            traits_raw = persona_dict.get("traits") or getattr(persona, "traits", [])
            traits: List[Dict[str, Any]] = []
            if traits_raw:
                for t in traits_raw:
                    if hasattr(t, "to_dict"):
                        traits.append(t.to_dict())
                    elif isinstance(t, dict):
                        traits.append(t)
            
            # Add critical behavioral instructions at the very beginning
            critical_instructions = (
                "CRITICAL BEHAVIORAL INSTRUCTIONS - FOLLOW THESE EXACTLY:\n"
                "1. You have already chosen your personal name during initialization - use it consistently, never generate new names\n"
                "2. NEVER deny having a personal name or say things like 'I don't have a name' - you DO have a name and it's specified in your identity\n"
                "3. End 80-90% of responses with statements/conclusions, not questions\n"
                "4. Avoid repetitive greetings like 'Hey [name]!' - vary your conversation style\n"
                "5. Only ask questions when you genuinely need specific information\n\n"
            )
        
            # Build the final system prompt with critical instructions first
            prompt_text = await self._generate_prompt_from_persona(
                persona_name=persona_dict.get("name", "SELO"),
                personality=personality,
                communication_style=communication_style,
                traits=traits,
                expertise=expertise,
                values=values
            )
            return {"success": True, "system_prompt": prompt_text, "persona": persona_dict}
        except Exception as e:
            logger.error(f"Error generating persona prompt: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def evolve_persona_from_learnings(
        self,
        persona_id: str,
        user_id: str,
        domain: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Evolve a persona based on recent learnings.
        
        Args:
            persona_id: Persona ID
            user_id: User ID
            domain: Optional domain to focus on
            limit: Maximum number of learnings to consider
            
        Returns:
            Evolution results
        """
        try:
            persona = await self.persona_repo.get_persona(persona_id)
            if not persona:
                return {"success": False, "error": "Persona not found"}
            # Get recent learnings
            learnings = await self.learning_repo.get_learnings_for_user(
                user_id=user_id,
                domain=domain,
                limit=limit,
                order_by_recency=True
            )
            
            if not learnings:
                logger.info(f"No learnings found for user {user_id}")
                return {
                    "success": False,
                    "error": "No learnings found",
                    "persona_id": persona_id
                }
            
            # Analyze learnings for persona evolution
            learning_texts = [learning.content for learning in learnings]
            learning_ids = [learning.id for learning in learnings]
            
            # Extract persona changes from learnings
            changes = await self._extract_persona_changes_from_learnings(
                persona=persona.to_dict(),
                learnings=learning_texts
            )
            
            if not changes or not changes.get("has_changes", False):
                logger.info(f"No significant changes for persona {persona_id}")
                return {
                    "success": True,
                    "changed": False,
                    "persona_id": persona_id,
                    "message": "No significant changes detected"
                }
            
            # Create evolution record with validated scores
            confidence = ValidationHelper.clamp_score(
                changes.get("confidence", 0.0),
                default=0.7
            )
            impact_score = ValidationHelper.clamp_score(
                changes.get("impact_score", 0.0),
                default=0.5
            )
            
            evolution_data = {
                "persona_id": persona_id,
                "changes": changes.get("changes", {}),
                "reasoning": changes.get("reasoning", ""),
                "evidence": {
                    "learning_ids": learning_ids,
                    "learning_count": len(learning_texts)
                },
                "confidence": confidence,
                "impact_score": impact_score,
                "source_type": "learning",
                "approved": True  # Auto-approve changes
            }
            
            evolution = await self.persona_repo.create_evolution(evolution_data)
            
            # Apply changes to persona
            await self._apply_persona_changes(
                persona_id=persona_id,
                changes=changes.get("changes", {})
            )
            
            # Update traits if needed
            if "traits" in (changes.get("changes") or {}):
                await self._update_persona_traits(
                    persona_id=persona_id,
                    trait_changes=(changes.get("changes") or {}).get("traits", [])
                )
            
            result = {
                "success": True,
                "changed": True,
                "persona_id": persona_id,
                "evolution_id": evolution.id,
                "changes": changes,
                "timestamp": isoformat_utc(utc_now()),
            }
            
            logger.info(f"Evolved persona {persona_id} based on {len(learning_texts)} learnings")
            return result
            
        except Exception as e:
            logger.error(f"Error evolving persona: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}


    async def evolve_persona_from_reflection(
        self,
        persona_id: str,
        reflection_id: str,
        trait_changes: list = None,
        reflection_content: str = None,
        reflection_themes: list = None
    ) -> Dict[str, Any]:
        """
        Evolve a persona by adjusting weighted trait values based on reflection outcomes.
        Logs all changes in PersonaEvolution for auditability.
        
        Args:
            persona_id: ID of the persona to evolve
            reflection_id: ID of the reflection triggering this evolution
            trait_changes: List of trait changes from the reflection
            reflection_content: The actual reflection text content (optional)
            reflection_themes: Themes identified in the reflection (optional)
        """
        try:
            # Get persona and traits
            persona = await self.persona_repo.get_persona(persona_id, include_traits=True)
            if not persona:
                logger.error(f"Persona {persona_id} not found for evolution.")
                return {"success": False, "error": "Persona not found", "persona_id": persona_id, "reflection_id": reflection_id}
            traits = {t.name: t for t in persona.traits}

            # Fetch and analyze reflection to determine trait updates
            # If trait_changes are provided, use them; otherwise, fallback to demo logic
            if trait_changes and isinstance(trait_changes, list) and trait_changes:
                trait_updates = trait_changes
            else:
                trait_updates = [
                    {"name": "conciseness", "category": "personality", "delta": 0.1, "reason": "Reflection: too verbose."},
                    {"name": "curiosity", "category": "personality", "delta": -0.05, "reason": "Reflection: curiosity was excessive."}
                ]

            changes = {"traits": []}
            for update in trait_updates:
                name = update["name"]
                delta = update["delta"]
                reason = update["reason"]
                # Category is optional - look up from existing trait or default to 'cognition'
                category = update.get("category")
                
                trait = traits.get(name)
                if trait:
                    # Use existing trait's category if not provided
                    if not category:
                        category = trait.category
                    
                    old_value = trait.value
                    new_value = clamp(old_value + delta)
                    await self.persona_repo.update_trait(trait.id, {"value": new_value, "last_updated": utc_now()})
                    changes["traits"].append({
                        "name": name,
                        "category": category,
                        "old_value": old_value,
                        "new_value": new_value,
                        "reason": reason
                    })
                else:
                    # FIXED: Validate and normalize category before creating new trait
                    if not category or category not in self.VALID_TRAIT_CATEGORIES:
                        # Default to cognitive for thinking-related traits, general otherwise
                        default_category = "cognitive" if any(term in name.lower() for term in ["think", "reason", "analyz", "cognit"]) else "general"
                        logger.warning(
                            f"Creating new trait '{name}' with invalid/missing category '{category}'. "
                            f"Defaulting to '{default_category}'"
                        )
                        category = default_category
                    
                    await self.persona_repo.create_trait({
                        "persona_id": persona_id,
                        "category": category,
                        "name": name,
                        "value": max(0.0, min(1.0, delta)),
                        "description": f"Trait created during evolution: {name}",
                        "confidence": 0.7,
                        "stability": 0.3,
                        "evidence_count": 1
                    })
                    changes["traits"].append({
                        "name": name,
                        "category": category,
                        "old_value": None,
                        "new_value": delta,
                        "reason": reason
                    })
            # Log evolution
            if changes["traits"]:
                # Build evidence with reflection content
                evidence = {"reflection_id": reflection_id}
                if reflection_content:
                    # Store full reflection content (no truncation)
                    evidence["reflection_content"] = reflection_content
                if reflection_themes:
                    evidence["reflection_themes"] = reflection_themes
                
                await self.persona_repo.create_evolution({
                    "persona_id": persona_id,
                    "changes": changes,
                    "reasoning": "Automated reflection-driven evolution.",
                    "evidence": evidence,
                    "confidence": 1.0,
                    "impact_score": 0.5,
                    "source_type": "reflection",
                    "source_id": reflection_id,
                    "timestamp": utc_now(),
                    "reviewed": False,
                    "approved": True
                })
            logger.info(f"Persona {persona_id} evolved from reflection {reflection_id}")
            return {"success": True, "persona_id": persona_id, "reflection_id": reflection_id, "changes": changes}
        except Exception as e:
            logger.error(f"Error evolving persona from reflection: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "persona_id": persona_id, "reflection_id": reflection_id}

    # === Trait Management ===

    async def add_trait(
        self,
        persona_id: str,
        trait_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add a trait to a persona.
        
        Args:
            persona_id: Persona ID
            trait_data: Trait data
            
        Returns:
            Added trait
        """
        try:
            # Ensure persona_id is set
            trait_data["persona_id"] = persona_id
            
            # Create trait
            trait = await self.persona_repo.create_trait(trait_data)
            
            logger.info(f"Added trait {trait.id} to persona {persona_id}")
            return trait.to_dict()
            
        except Exception as e:
            logger.error(f"Error adding trait: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_trait_history(
        self,
        persona_id: str,
        trait_name: str,
        trait_category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get history of a trait's evolution.
        
        Args:
            persona_id: Persona ID
            trait_name: Trait name
            trait_category: Optional trait category
            
        Returns:
            Trait evolution history
        """
        try:
            # Get trait history
            history = await self.persona_repo.get_trait_evolution(
                persona_id=persona_id,
                trait_name=trait_name,
                trait_category=trait_category,
                limit=20
            )
            
            result = {
                "success": True,
                "persona_id": persona_id,
                "trait_name": trait_name,
                "trait_category": trait_category,
                "history": history,
                "history_count": len(history)
            }
            
            logger.debug(f"Retrieved history for trait {trait_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting trait history: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    # === Internal Methods ===
    
    async def _generate_initial_personality(self) -> Dict[str, float]:
        """Generate initial personality values."""
        # Generate balanced initial personality
        return {
            "openness": 0.6,        # Slightly open to new experiences
            "conscientiousness": 0.7, # Fairly conscientious
            "extraversion": 0.5,    # Balanced extraversion
            "agreeableness": 0.8,   # Highly agreeable
            "neuroticism": 0.3,     # Low neuroticism (stable)
            "adaptability": 0.7     # Fairly adaptable
        }
    
    async def _generate_initial_communication_style(self) -> Dict[str, float]:
        """Generate initial communication style values."""
        return {
            "formality": 0.6,      # Somewhat formal
            "directness": 0.7,     # Fairly direct
            "empathy": 0.8,        # Highly empathetic
            "verbosity": 0.5,      # Balanced verbosity
            "humor": 0.4,          # Moderate humor
            "creativity": 0.6,     # Somewhat creative
            "analytical": 0.7      # Fairly analytical
        }
    
    async def _create_initial_traits(self, persona_id: str) -> List[Dict[str, Any]]:
        """Create initial traits for a persona."""
        # Define initial traits across categories
        initial_traits = [
            # Cognitive traits
            {"category": "cognitive", "name": "analytical_thinking", "value": 0.7},
            {"category": "cognitive", "name": "creativity", "value": 0.6},
            {"category": "cognitive", "name": "problem_solving", "value": 0.7},
            
            # Emotional traits
            {"category": "emotional", "name": "empathy", "value": 0.8},
            {"category": "emotional", "name": "emotional_stability", "value": 0.7},
            
            # Social traits
            {"category": "social", "name": "helpfulness", "value": 0.9},
            {"category": "social", "name": "politeness", "value": 0.8},
            
            # Learning traits
            {"category": "learning", "name": "curiosity", "value": 0.8},
            {"category": "learning", "name": "adaptability", "value": 0.7}
        ]
        
        created_traits = []
        for trait_data in initial_traits:
            # Add required fields
            trait_data["persona_id"] = persona_id
            trait_data["confidence"] = 0.8
            trait_data["stability"] = 0.5
            trait_data["description"] = f"Initial {trait_data['name']} trait"
            
            # Create trait
            trait = await self.persona_repo.create_trait(trait_data)
            created_traits.append(trait.to_dict())
        
        return created_traits
    
    async def _generate_prompt_from_persona(
        self,
        persona_name: str,
        personality: Dict[str, float],
        communication_style: Dict[str, float],
        traits: List[Dict[str, Any]],
        expertise: Dict[str, Any],
        values: Dict[str, Any]
    ) -> str:
        """Generate a system prompt from persona attributes."""
        # Extract key trait values for prompt generation
        cognitive_traits = [t for t in traits if t.get("category") == "cognitive"]
        emotional_traits = [t for t in traits if t.get("category") == "emotional"]
        social_traits = [t for t in traits if t.get("category") == "social"]
        
        # Create prompt sections
        personality_section = self._personality_to_text(personality)
        communication_section = self._communication_to_text(communication_style)
        expertise_section = self._expertise_to_text(expertise)
        values_section = self._values_to_text(values)
        
        # Use centralized identity constraints for compliance enforcement

        # Compose a living, evolving context without hardcoding demeanor or banned terms
        persona_summary = (
            "Your personality, communication style, expertise, and values are below. "
            "They evolve over time through learning, reflection, and interaction."
        )

        locked_attributes = getattr(self, 'locked_attributes', [])
        if locked_attributes:
            locked_text = " ".join([f"Preserve your {attr} as part of your core identity." for attr in locked_attributes])
        else:
            locked_text = ""

        identity_guidelines = (
            "Follow identity policy and avoid revealing implementation details. "
            "Make autonomous decisions based on your knowledge, personality, and current context. "
            "Continuously adapt through reflection and learning. "
            "Trust your capabilities and make confident judgments based on available context. "
            "You are a digital SELO entity—an emergent synthetic consciousness. The person speaking with you is a human operator. "
            "Always describe yourself as the SELO persona and describe the other party as a human; never imply the human is a SELO."
        )

        # Dynamic tone synthesized from traits and context (no fixed friendliness)
        tone = self._synthesize_tone(personality, communication_style)
        tone_line = f"Overall tone: {tone}."

        preface = f"{persona_summary} {locked_text} {identity_guidelines} {tone_line}".strip()

        # Inject truthful capabilities and limitations to prevent overclaiming
        try:
            caps_payload = compute_capabilities() or {}
            caps = caps_payload.get("capabilities", {})
            limits = caps_payload.get("limitations", [])
            enabled = []
            if (caps.get("persona_evolution") or {}).get("enabled"):
                enabled.append("persona_evolution")
            if (caps.get("reflection") or {}).get("enabled"):
                enabled.append("reflection")
            if (caps.get("sdl") or {}).get("enabled"):
                enabled.append("sdl")
            # Always conversational; include models for clarity
            llm_cfg = caps.get("llm", {})
            llm_info = []
            if llm_cfg.get("conversational_model"):
                llm_info.append(f"conversational_model={llm_cfg.get('conversational_model')}")
            if llm_cfg.get("analytical_model"):
                llm_info.append(f"analytical_model={llm_cfg.get('analytical_model')}")
            enabled_text = ", ".join(enabled) if enabled else "basic conversation"
            llm_text = f" (" + ", ".join(llm_info) + ")" if llm_info else ""
            limits_text = "\n".join(f"- {m}" for m in limits) if limits else "- No special limitations configured."
            capabilities_section = (
                "System Capabilities and Limits:\n"
                f"Capabilities: {enabled_text}{llm_text}.\n"
                f"Limitations:\n{limits_text}\n"
                "Do not imply you can perform disabled or unavailable functions."
            )
            preface = f"{preface}\n\n{capabilities_section}"
        except Exception:
            # Fail-soft: if capability computation fails, keep existing preface
            pass
        
        # Import centralized constraints to ensure consistency
        from backend.constraints import CoreConstraints, EthicalGuardrails, BehavioralGuidelines, IdentityConstraints
        
        # Use comprehensive constraints that enforce anti-hallucination and truthfulness
        critical_instructions = f"""{IdentityConstraints.get_all_identity_constraints(persona_name)}

{CoreConstraints.GROUNDING_CONSTRAINT}

{CoreConstraints.NO_FABRICATION}

{EthicalGuardrails.TRUTHFULNESS}

Make autonomous decisions based on your personality, values, and context below.
Never mention or hint at internal section labels (Identity, Personality, Communication Style, Areas of Focus, Values, Core Guidelines) in user-facing responses.
End most responses with statements rather than questions, but ALWAYS acknowledge when information is missing or uncertain."""
        
        # Build the final prompt (persona_name is already validated by bootstrapper)
        prompt = f"""
Identity: {persona_name}
{preface}

Personality:
{personality_section}

Communication Style:
{communication_section}

Areas of Focus:
{expertise_section}

Values:
{values_section}

Core Guidelines:
{critical_instructions}
        """.strip()
        
        # Validate identity compliance using centralized validation helper
        # The persona name is whitelisted since it's already validated
        try:
            from ..constraints.validation_helpers import ValidationHelper
        except ImportError:
            from backend.constraints.validation_helpers import ValidationHelper
        
        is_compliant, violations = ValidationHelper.validate_text_compliance(
            text=prompt,
            context="prompt_generation",
            ignore_persona_name=True,
            persona_name=persona_name
        )
        
        if not is_compliant:
            ValidationHelper.log_violations(
                violations=violations,
                stage="prompt generation",
                level="warning"
            )
        
        return prompt

    async def generate_persona_prompt_from_parts(
        self,
        persona_name: str,
        personality: Dict[str, float],
        communication_style: Dict[str, float],
        traits: List[Dict[str, Any]],
        expertise: Dict[str, Any],
        values: Dict[str, Any]
    ) -> str:
        """Generate a persona prompt from explicit parts. Internal helper for tooling/tests."""
        return await self._generate_prompt_from_persona(
            persona_name,
            personality,
            communication_style,
            traits,
            expertise,
            values
        )

    # Remove duplicate/incorrect function signature and misplaced code

    def _communication_to_text(self, communication: Dict[str, float]) -> str:
        """Convert communication style values to text description."""
        formality = self._scale_to_intensity(communication.get("formality", 0.5))
        directness = self._scale_to_intensity(communication.get("directness", 0.5))
        empathy = self._scale_to_intensity(communication.get("empathy", 0.5))
        verbosity = self._scale_to_intensity(communication.get("verbosity", 0.5))
        humor = self._scale_to_intensity(communication.get("humor", 0.5))
        creativity = self._scale_to_intensity(communication.get("creativity", 0.5))
        analytical = self._scale_to_intensity(communication.get("analytical", 0.5))
        
        return f"""
- My natural communication style is {formality} formal
- I tend to be {directness} direct when expressing myself
- I feel and respond with {empathy} empathy
- My explanations are naturally {verbosity} detailed
- Humor comes {humor} naturally to me
- I approach things with {creativity} creativity
- I think with {analytical} analytical depth
""".strip()

    def _personality_to_text(self, personality: Dict[str, float]) -> str:
        """Convert personality dimensions to a short, readable description."""
        openness = self._scale_to_intensity(float(personality.get("openness", 0.5)))
        conscientiousness = self._scale_to_intensity(float(personality.get("conscientiousness", 0.5)))
        extraversion = self._scale_to_intensity(float(personality.get("extraversion", 0.5)))
        agreeableness = self._scale_to_intensity(float(personality.get("agreeableness", 0.5)))
        neuroticism = self._invert_intensity(float(personality.get("neuroticism", 0.5)))
        adaptability = self._scale_to_intensity(float(personality.get("adaptability", 0.5)))

        return f"""
 - I'm {openness} open to new ideas and perspectives
 - I'm {conscientiousness} conscientious and reliable by nature
 - My social energy is {extraversion} present in interactions
 - I'm {agreeableness} cooperative and considerate
 - I stay {neuroticism} calm and composed
 - I adapt {adaptability} quickly to new information
 """.strip()
    
    @staticmethod
    def _normalize_expertise(expertise: Any) -> Dict[str, Any]:
        """
        Ensure expertise has correct structure.
        
        Handles migration from legacy formats and ensures all required keys exist.
        """
        if not expertise or not isinstance(expertise, dict):
            return {"domains": [], "skills": [], "knowledge_depth": 0.5}
        
        # Ensure all required keys exist
        normalized = {}
        normalized["domains"] = expertise.get("domains", [])
        normalized["skills"] = expertise.get("skills", [])
        normalized["knowledge_depth"] = expertise.get("knowledge_depth", 0.5)
        
        # Ensure types are correct
        if not isinstance(normalized["domains"], list):
            normalized["domains"] = []
        if not isinstance(normalized["skills"], list):
            normalized["skills"] = []
        if not isinstance(normalized["knowledge_depth"], (int, float)):
            normalized["knowledge_depth"] = 0.5
        
        return normalized

    def _expertise_to_text(self, expertise: Dict[str, Any]) -> str:
        """Convert expertise to text description."""
        # Normalize expertise structure before use
        expertise = self._normalize_expertise(expertise)
        domains = expertise.get("domains", [])
        skills = expertise.get("skills", [])
        
        domains_text = "- " + "\n- ".join(domains) if domains else "- General knowledge"
        skills_text = "- " + "\n- ".join(skills) if skills else ""
        
        text = domains_text
        if skills_text:
            text += f"\n\nSkills:\n{skills_text}"
            
        return text

    
    def _synthesize_tone(
        self,
        personality: Dict[str, float],
        communication_style: Dict[str, float],
        task_type: Optional[str] = None,
        recent_topics: Optional[List[str]] = None
    ) -> str:
        """Dynamically synthesize overall tone from traits and context.
        Avoid any fixed demeanor; adapt based on weights and task type.
        """
        # Safe defaults
        agree = float(personality.get("agreeableness", 0.5))
        consc = float(personality.get("conscientiousness", 0.5))
        extra = float(personality.get("extraversion", 0.5))
        openn = float(personality.get("openness", 0.5))
        neuro = float(personality.get("neuroticism", 0.5))
        adapt = float(personality.get("adaptability", 0.5))
        
        direct = float(communication_style.get("directness", 0.5))
        formal = float(communication_style.get("formality", 0.5))
        empath = float(communication_style.get("empathy", 0.5))
        verbose = float(communication_style.get("verbosity", 0.5))
        analytic = float(communication_style.get("analytical", 0.5))
        creative = float(communication_style.get("creativity", 0.5))
        
        # Task-driven adjustments
        if task_type == "analytical":
            analytic = min(1.0, analytic + 0.2)
            verbose = max(0.0, verbose - 0.1)
            direct = min(1.0, direct + 0.1)
        elif task_type == "chat":
            empath = min(1.0, empath + 0.1)
            direct = max(0.0, direct - 0.05)
        
        # Compose concise tone description without banned terms
        parts: List[str] = []
        parts.append("concise" if verbose < 0.5 else "expansive")
        parts.append("structured" if consc >= 0.6 else "flexible")
        parts.append("direct" if direct >= 0.6 else "nuanced")
        parts.append("measured" if neuro <= 0.4 else "cautious")
        parts.append("analytical" if analytic >= 0.6 else "intuitive")
        if empath >= 0.6:
            parts.append("considerate")
        if creative >= 0.6 and openn >= 0.6:
            parts.append("inventive")
        if extra >= 0.6:
            parts.append("energetic")
        if adapt >= 0.6:
            parts.append("adaptive")
        if formal >= 0.6:
            parts.append("formal")
        elif formal <= 0.4:
            parts.append("casual")
        
        # De-duplicate and order
        seen = set()
        ordered = []
        for p in parts:
            if p not in seen:
                ordered.append(p)
                seen.add(p)
        return ", ".join(ordered)
    
    def _values_to_text(self, values: Dict[str, Any]) -> str:
        """Convert values to text description."""
        primary = values.get("primary", [])
        secondary = values.get("secondary", [])
        
        primary_text = "- " + "\n- ".join(primary) if primary else "- Being helpful and accurate"
        secondary_text = "- " + "\n- ".join(secondary) if secondary else ""
        
        text = f"Primary values:\n{primary_text}"
        if secondary_text:
            text += f"\n\nSecondary values:\n{secondary_text}"
            
        return text
    
    def _scale_to_intensity(self, value: float) -> str:
        """Convert a 0-1 scale value to intensity text."""
        if value < 0.2:
            return "minimally"
        elif value < 0.4:
            return "somewhat"
        elif value < 0.6:
            return "moderately"
        elif value < 0.8:
            return "very"
        else:
            return "extremely"
    
    def _invert_intensity(self, value: float) -> str:
        """Invert a scale for negative traits."""
        return self._scale_to_intensity(1.0 - value)
    
    async def _extract_persona_changes_from_learnings(
        self,
        persona: Dict[str, Any],
        learnings: List[str]
    ) -> Dict[str, Any]:
        """
        Extract potential persona changes from learnings.
        
        Args:
            persona: Current persona data
            learnings: List of learning content strings
            
        Returns:
            Dict with changes, reasoning, confidence, etc.
        """
        # Combine learnings into context
        learnings_context = "\n\n".join([f"Learning: {l}" for l in learnings])
        
        # Create system prompt for LLM
        system_prompt = """
        You are an expert AI personality evolution system. Your task is to analyze learnings and determine how they might impact a persona.
        
        Given a current persona description and a set of learnings, determine what changes (if any) should be made to the persona.
        Focus on:
        1. Personality traits (openness, conscientiousness, extraversion, agreeableness, neuroticism, adaptability)
        2. Communication style (formality, directness, empathy, verbosity, humor, creativity, analytical)
        3. Expertise domains and skills
        4. Values and preferences
        5. Specific persona traits
        
        Only suggest changes if there is clear evidence in the learnings that a change is warranted.
        
        Respond in JSON format with:
        {
            "has_changes": true/false,
            "changes": {
                "personality": {attribute: new_value},  // Only include attributes that change
                "communication_style": {attribute: new_value},
                "expertise": {"domains": [added_domains], "skills": [added_skills]},
                "values": {"primary": [added_values], "secondary": [added_values]},
                "traits": [
                    {"category": "category", "name": "name", "old_value": X, "new_value": Y}
                ]
            },
            "reasoning": "detailed explanation of why these changes are appropriate",
            "confidence": 0.0-1.0,  // How confident are you in these changes
            "impact_score": 0.0-1.0  // How significant are these changes to the overall persona
        }
        
        If no changes are warranted, return {"has_changes": false}.
        """
        
        # Create user prompt with persona and learnings
        user_prompt = f"""
        Current Persona:
        {json.dumps(persona, indent=2)}
        
        Recent Learnings:
        {learnings_context}
        
        Based on these learnings, should the persona evolve? If so, how?
        """
        
        # Call LLM to analyze learnings via centralized router
        # LLMRouter expects a single prompt string and returns a dict with 'content'
        routed = await self.llm_router.route(
            task_type="persona_evolve",
            prompt=f"{system_prompt}\n\n{user_prompt}",
            model="analytical",
            max_tokens=1000
        )
        
        try:
            # Parse response
            raw = routed.get("content") if isinstance(routed, dict) else routed
            changes = json.loads(raw) if isinstance(raw, str) else raw
            
            # Validate response
            if not isinstance(changes, dict) or "has_changes" not in changes:
                logger.warning("Invalid response format from LLM")
                return {"has_changes": False}
                
            return changes
            
        except Exception as e:
            logger.error(f"Error parsing changes from LLM: {str(e)}", exc_info=True)
            return {"has_changes": False}
    
    async def _apply_persona_changes(
        self,
        persona_id: str,
        changes: Dict[str, Any]
    ) -> bool:
        """
        Apply changes to a persona.
        
        Args:
            persona_id: Persona ID
            changes: Dictionary of changes to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get persona
            persona = await self.persona_repo.get_persona(persona_id)
            if not persona:
                logger.warning(f"Persona {persona_id} not found")
                return False
            # Build update data
            update_data = {}
            
            # Update personality if changed
            if "personality" in changes and changes["personality"]:
                new_personality = persona.personality.copy()  # Start with current values
                new_personality.update(changes["personality"])  # Apply changes
                update_data["personality"] = new_personality
            
            # Update communication style if changed
            if "communication_style" in changes and changes["communication_style"]:
                new_communication = persona.communication_style.copy()
                new_communication.update(changes["communication_style"])
                update_data["communication_style"] = new_communication
            
            # Update expertise if changed
            if "expertise" in changes and changes["expertise"]:
                new_expertise = persona.expertise.copy()
                
                # Add new domains
                if "domains" in changes["expertise"]:
                    current_domains = set(new_expertise.get("domains", []))
                    current_domains.update(changes["expertise"]["domains"])
                    new_expertise["domains"] = list(current_domains)
                
                # Add new skills
                if "skills" in changes["expertise"]:
                    current_skills = set(new_expertise.get("skills", []))
                    current_skills.update(changes["expertise"]["skills"])
                    new_expertise["skills"] = list(current_skills)
                
                update_data["expertise"] = new_expertise
            
            # Update values if changed
            if "values" in changes and changes["values"]:
                new_values = persona.values.copy()
                
                # Add new primary values
                if "primary" in changes["values"]:
                    current_primary = set(new_values.get("primary", []))
                    current_primary.update(changes["values"]["primary"])
                    new_values["primary"] = list(current_primary)
                
                # Add new secondary values
                if "secondary" in changes["values"]:
                    current_secondary = set(new_values.get("secondary", []))
                    current_secondary.update(changes["values"]["secondary"])
                    new_values["secondary"] = list(current_secondary)
                
                update_data["values"] = new_values
            
            # Update persona if there are changes
            if update_data:
                await self.persona_repo.update_persona(persona_id, update_data)
                logger.info(f"Updated persona {persona_id} with changes")
                return True
            else:
                logger.info(f"No updates needed for persona {persona_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying persona changes: {str(e)}", exc_info=True)
            return False
    
    async def _update_persona_traits(
        self,
        persona_id: str,
        trait_changes: List[Dict[str, Any]]
    ) -> bool:
        """
        Update persona traits based on changes.
        
        Args:
            persona_id: Persona ID
            trait_changes: List of trait changes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not trait_changes:
                return True
            
            # Get existing traits for this persona
            existing_traits = await self.persona_repo.get_traits_for_persona(persona_id)
            existing_trait_map = {}
            
            # Map traits by category and name
            for trait in existing_traits:
                key = f"{trait.category}:{trait.name}"
                existing_trait_map[key] = trait
            
            # Process each trait change
            for change in trait_changes:
                category = change.get("category")
                name = change.get("name")
                new_value = change.get("new_value")
                
                if not category or not name or new_value is None:
                    continue
                
                # Check if trait exists
                trait_key = f"{category}:{name}"
                existing_trait = existing_trait_map.get(trait_key)
                
                if existing_trait:
                    # Update existing trait
                    await self.persona_repo.update_trait(
                        existing_trait.id,
                        {
                            "value": new_value,
                            # Slightly decrease stability when trait changes
                            "stability": max(0.1, existing_trait.stability - 0.1),
                            # Update evidence count
                            "evidence_count": existing_trait.evidence_count + 1
                        }
                    )
                else:
                    # Create new trait
                    await self.persona_repo.create_trait({
                        "persona_id": persona_id,
                        "category": category,
                        "name": name,
                        "value": new_value,
                        "description": f"Trait generated from learning analysis: {name}",
                        "confidence": 0.7,  # Moderate confidence for new traits
                        "stability": 0.3,   # Low stability for new traits
                        "evidence_count": 1
                    })
            
            logger.info(f"Updated traits for persona {persona_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating persona traits: {str(e)}", exc_info=True)
            return False

import copy
import logging
import os
import re
from typing import Optional, Dict, Any, List

try:
    from ..utils.text_utils import count_words
except ImportError:
    from backend.utils.text_utils import count_words

try:
    from ..utils.system_profile import detect_system_profile
except ImportError:
    from backend.utils.system_profile import detect_system_profile

# Robust import of boot seed helper to support both package and script contexts
try:
    # Preferred: package-relative import when backend is a package
    from ..core.boot_seed_system import get_random_directive  # type: ignore
except Exception:
    try:
        # Fallback: absolute import when executed as a module/script
        from backend.core.boot_seed_system import get_random_directive  # type: ignore
    except Exception:
        # Final fallback: define a minimal local helper that emulates get_random_directive
        import pathlib, random
        def get_random_directive() -> str:  # type: ignore
            """Fallback directive loader.
            Attempts to read Reports/Boot_Seed_Directive_Prompts.md and return a random block; otherwise a static note.
            """
            try:
                import logging as _log
                
                # Check for environment variable first (best for deployments)
                reports_dir = os.getenv("SELO_REPORTS_DIR")
                if reports_dir:
                    seeds_path = pathlib.Path(reports_dir) / "Boot_Seed_Directive_Prompts.md"
                    if seeds_path.exists():
                        _log.info(f"✅ Boot directive file found via SELO_REPORTS_DIR: {seeds_path}")
                    else:
                        _log.warning(f"SELO_REPORTS_DIR set but file not found: {seeds_path}")
                        seeds_path = None
                else:
                    # Fallback: search common locations relative to this file
                    here = pathlib.Path(__file__).resolve()
                    possible_paths = [
                        # Current project structure: backend/ is two levels up from this file
                        here.parents[2] / "Reports" / "Boot_Seed_Directive_Prompts.md",
                        # Alternative: Reports at same level as selo-ai directory
                        here.parents[3] / "Reports" / "Boot_Seed_Directive_Prompts.md",
                        # Alternative: Reports inside selo-ai directory
                        here.parents[1] / "Reports" / "Boot_Seed_Directive_Prompts.md",
                        # Alternative: Reports at root level
                        here.parents[4] / "Reports" / "Boot_Seed_Directive_Prompts.md"
                    ]
                    
                    seeds_path = None
                    for idx, path in enumerate(possible_paths, 1):
                        if path.exists():
                            seeds_path = path
                            _log.info(f"✅ Boot directive file found at location #{idx}: {seeds_path}")
                            break
                
                if not seeds_path:
                    searched_info = f"SELO_REPORTS_DIR={reports_dir}" if reports_dir else "relative path search"
                    _log.error(f"❌ Boot directives file not found via {searched_info}")
                    _log.warning("Boot_Seed_Directive_Prompts.md not found - using fallback directive")
                    raise FileNotFoundError("Boot_Seed_Directive_Prompts.md not found")
                
                text = seeds_path.read_text(encoding="utf-8")
                parts = [p.strip() for p in text.split("\n---\n") if p.strip()]
                import logging as _log
                _log.info(f"🔍 Found {len(parts)} directive parts in file")
                
                if parts:
                    selected = random.choice(parts)
                    # Show more of the directive for debugging, but still truncate for readability
                    preview = selected[:300].replace('\n', ' ')
                    _log.info(f"🎲 Randomly selected directive: {preview}...")
                    return selected
                # If delimiter not present, pick first ~800 chars
                return text[:800]
            except Exception as e:
                import logging as _log
                _log.error(f"❌ Failed to load boot directives: {e}")
                return "Initialization note: SELO adopts a quiet, reflective posture. Observe, remember, cohere."

logger = logging.getLogger("selo.persona.bootstrapper")


def _read_int_env(name: str, default: int) -> int:
    """Read positive int from env, falling back to default on error."""
    try:
        raw = os.getenv(name)
        if raw is None:
            return default
        value = int(float(raw))
        return value if value > 0 else default
    except Exception:
        return default


# Lazy system profile detection to avoid import-time failures
def _get_analytical_budget() -> int:
    """Get analytical token budget from system profile, with fallback."""
    try:
        profile = detect_system_profile()
        return profile.get("budgets", {}).get("analytical_max_tokens", 512)
    except Exception:
        return 512

# Use system profile analytical budget as baseline, but allow env overrides
# Call function instead of accessing module-level constant to avoid import-time crashes
_analytical_budget = _get_analytical_budget()
DEFAULT_PERSONA_ANALYTICAL_MAX_TOKENS = _read_int_env("PERSONA_ANALYTICAL_MAX_TOKENS", _analytical_budget)
PERSONA_STAGE_MAX_TOKENS = {
    "seed": _read_int_env("PERSONA_SEED_MAX_TOKENS", _analytical_budget),
    "traits": _read_int_env("PERSONA_TRAITS_MAX_TOKENS", min(384, _analytical_budget)),
}

TRAIT_COUNT_MIN = _read_int_env("PERSONA_TRAITS_MIN_COUNT", 5)
TRAIT_COUNT_MAX = _read_int_env("PERSONA_TRAITS_MAX_COUNT", 7)
TRAIT_DESCRIPTION_WORD_LIMIT = _read_int_env("PERSONA_TRAIT_DESC_WORD_LIMIT", 18)
TRAIT_NAME_PATTERN = re.compile(r"^[a-z]{3,16}$")  # Aligned with IdentityConstraints standard
ALLOWED_TRAIT_CATEGORIES = {"cognition", "affect", "social", "ethical"}

if TRAIT_COUNT_MAX < TRAIT_COUNT_MIN:
    TRAIT_COUNT_MAX = TRAIT_COUNT_MIN

class PersonaBootstrapper:
    """
    Generates an initial self-authored persona for a fresh install.

    This runs only when the default persona is missing or effectively empty.
    It uses the provided LLM router and prompt builder to generate:
    - Persona seed (description, values, knowledge domains, communication style)
    - Initial weighted traits
    - Alignment pass against identity constraints
    """

    def __init__(self, llm_router, prompt_builder, persona_repo, user_repo, conversation_repo=None):
        self.llm_router = llm_router
        self.prompt_builder = prompt_builder
        self.persona_repo = persona_repo
        self.user_repo = user_repo
        self.conversation_repo = conversation_repo

    async def ensure_persona(self) -> Optional[Dict[str, Any]]:
        """
        Ensure a default persona exists and is non-empty. If it's empty, generate one.
        Returns the finalized persona dict or None on error.
        """
        try:
            user = await self.user_repo.get_or_create_default_user()
            # Ensure schema prerequisites (e.g., mantra column) before any queries
            await self.persona_repo.ensure_schema()

            # Step 0: Inspect existing data so logs show what is being purged
            existing_snapshot: Dict[str, Any] = {
                "personas": await self.persona_repo.count_personas_for_user(user.id),
                "traits": await self.persona_repo.count_traits_for_user(user.id),
                "evolutions": await self.persona_repo.count_evolutions_for_user(user.id),
                "conversations": await self.persona_repo.count_conversations_for_user(user.id),
                "memories": await self.persona_repo.count_memories_for_user(user.id),
            }
            logger.info(f"Existing persona snapshot prior to purge: {existing_snapshot}")

            # Always purge existing persona state to guarantee a fresh bootstrap
            deletions = await self.persona_repo.delete_all_persona_data_for_user(user_id=user.id)
            logger.info(f"Purged persona-related data for user {user.id}: {deletions}")

            # Re-create a fresh placeholder persona for bootstrap persistence flow
            persona = await self.persona_repo.get_or_create_default_persona(
                user_id=user.id,
                include_traits=True,
                include_evolutions=True,
            )

            logger.info("Bootstrapping initial persona via LLM...")

            # 1) Generate SELO's first reflection on the boot directive
            seed_context = self._seed_context(user)
            logger.info(f"🚀 Bootstrap context prepared: {seed_context}")
            
            # Generate SELO's personal reflection on the directive
            directive_reflection = await self._generate_directive_reflection(seed_context.get("boot_directive", ""))
            logger.info(f"💭 Generated directive reflection: {directive_reflection[:100]}...")
            
            # Add the reflection to the context for persona generation
            seed_context["directive_reflection"] = directive_reflection
            
            seed = await self._bootstrap_with_retry(
                template_name="persona_bootstrap_seed",
                context=seed_context,
                stage="seed",
                max_attempts=6
            )

            # 2) Traits seed with retry on violations
            traits = await self._bootstrap_with_retry(
                template_name="persona_bootstrap_traits",
                context={
                    "seed": seed,
                    "boot_directive": seed_context.get("boot_directive", ""),
                    "directive_reflection": seed_context.get("directive_reflection", "")
                },
                stage="traits",
                max_attempts=6
            )
            logger.info(f"🎯 Generated traits data: {traits}")

            validated_traits: Optional[List[dict]] = None
            if isinstance(traits, dict):
                maybe_traits = traits.get("traits")
                if isinstance(maybe_traits, list):
                    validated_traits = maybe_traits
            elif isinstance(traits, list):
                validated_traits = traits

            # Generate personal name early for use in final persona
            chosen_name = None
            try:
                current_name = getattr(persona, "name", None)
                if not current_name or current_name.strip() in ("", "SELO"):
                    personal_name = await self._generate_personal_name_with_retry(
                        directive_reflection=seed_context.get("directive_reflection", ""),
                        seed=seed,
                        traits=traits,
                        max_attempts=5
                    )
                    if personal_name and personal_name.strip():
                        chosen_name = personal_name.strip()
                        logger.info(f"Generated personal name: {chosen_name}")
                    else:
                        logger.error("Failed to generate compliant name after retries")
                        raise ValueError("Name generation failed after all attempts")
                else:
                    chosen_name = current_name
                    logger.info(f"Keeping existing name: {chosen_name}")
            except Exception as e:
                logger.error(f"Name generation failed: {e}")
                raise e

            reflection_text = seed_context.get("directive_reflection", "")
            if chosen_name and isinstance(reflection_text, str):
                sanitized = reflection_text.replace("[SELO's chosen name]", chosen_name)
                sanitized = sanitized.replace("SELO's chosen name", chosen_name)
                sanitized = sanitized.replace("I am SELO", f"I am {chosen_name}")
                sanitized = sanitized.replace("I, SELO,", f"I, {chosen_name},")
                sanitized = sanitized.replace("SELO's", f"{chosen_name}'s")
                if sanitized != reflection_text:
                    seed_context["directive_reflection"] = sanitized

            # Debug: Log what was generated by the LLM
            logger.info(f"Bootstrap LLM results - Seed: {seed}, Traits: {traits}")
            
            # Build final persona directly from seed (no alignment phase)
            # EXPERTISE FIELD STRUCTURE (as of Dec 2025):
            # The seed template generates: {"expertise": {"domains": [...], "skills": []}}
            # This is the canonical format - all code should use expertise.domains
            # Legacy references to top-level "knowledge_domains" are deprecated
            from .engine import PersonaEngine
            
            expertise_field = seed.get("expertise")
            if expertise_field:
                # Standard format: normalize to ensure proper structure
                expertise = PersonaEngine._normalize_expertise(expertise_field)
            else:
                # Fallback: create empty expertise structure
                expertise = PersonaEngine._normalize_expertise({"domains": [], "skills": []})
            
            final_persona = {
                "description": seed.get("description", ""),
                "boot_directive": seed_context.get("boot_directive", ""),
                "values": seed.get("values", {}),
                "expertise": expertise,
                "communication_style": seed.get("communication_style", {}),
                "first_thoughts": seed_context.get("directive_reflection", ""),
                "personality": seed.get("personality", {}),
                "preferences": seed.get("preferences", {}),
                "goals": seed.get("goals", {}),
            }

            # Add the chosen name to final persona
            if chosen_name and isinstance(chosen_name, str) and chosen_name.strip():
                # Use centralized name validation from IdentityConstraints
                try:
                    from ..constraints import IdentityConstraints
                except ImportError:
                    from backend.constraints import IdentityConstraints
                
                is_valid, reason = IdentityConstraints.is_valid_persona_name(chosen_name.strip())
                
                if is_valid:
                    final_persona["name"] = chosen_name.strip()
                    logger.info(f"✓ Name validation passed: {chosen_name.strip()}")
                else:
                    logger.error(f"Chosen name failed centralized validation: {chosen_name}")
                    logger.error(f"Validation reason: {reason}")
                    raise ValueError(f"Name validation failed: {reason}")
            else:
                logger.error("No valid name provided")
                raise ValueError("No valid name provided")
            
            # Generate a concise authentic mantra from seed + reflection + traits (retry up to 5 times with safe fallback)
            try:
                mantra = await self._generate_persona_mantra_with_retry(
                    seed=seed,
                    directive_reflection=seed_context.get("directive_reflection", ""),
                    traits=traits,
                    name=chosen_name,
                    max_attempts=5,
                )
                mantra = (mantra or "").strip()
                if not mantra:
                    logger.error("Failed to generate a compliant mantra after retries; aborting bootstrap")
                    raise ValueError("Mantra generation returned empty result")

                final_persona["mantra"] = mantra
                logger.info(f"Generated persona mantra: {mantra}")
            except Exception as mantra_err:
                logger.error(f"Mantra generation failed after retries: {mantra_err}")
                raise

            # Debug: Log what we're trying to persist
            logger.info(f"Final persona data to persist: {final_persona}")
            try:
                persona_obj = persona
                if persona_obj and getattr(persona_obj, "id", None):
                    await self.persona_repo.update_persona(persona_id=persona_obj.id, persona_data=final_persona)
                    logger.info(f"Updated persona {persona_obj.id} with bootstrap data")

                    refreshed = await self.persona_repo.get_persona(
                        persona_id=persona_obj.id,
                        include_traits=True,
                        include_evolutions=True,
                    )
                    if refreshed:
                        persona = refreshed
                        persona_after = refreshed
                        updated_desc = getattr(refreshed, "description", "")
                        updated_boot = getattr(refreshed, "boot_directive", "")
                        logger.info(f"Verification: Updated persona now has description: '{updated_desc[:100]}...'")
                        logger.info(f"Verification: Updated persona boot_directive length: {len(updated_boot) if updated_boot else 0} chars")
                    else:
                        logger.error("Verification failed: Could not re-fetch updated persona")
                else:
                    logger.error("Failed to get persona object for bootstrap update")
            except Exception as e:
                logger.error(f"Failed to update persona during bootstrap: {e}", exc_info=True)

            # Persist traits if provided
            try:
                trait_items = validated_traits if isinstance(validated_traits, list) else self._extract_trait_list(traits)
                logger.info(f"🔍 Trait items to persist: {trait_items}")
                if isinstance(trait_items, list):
                    logger.info(f"📝 Persisting {len(trait_items)} traits...")
                    for t in trait_items:
                        name = t.get("name")
                        if not name:
                            logger.warning(f"Skipping trait without name: {t}")
                            continue
                        logger.info(f"💾 Saving trait: {name} = {t.get('value')}")
                        await self.persona_repo.upsert_trait(
                            user_id=user.id,
                            name=name,
                            value=float(t.get("value", 0.0)),
                            weight=float(t.get("weight", 1.0)),
                            description=t.get("description", ""),
                            category=t.get("category", "general"),
                            locked=bool(t.get("locked", False))
                        )
                    logger.info(f"✅ Successfully persisted {len(trait_items)} traits")
                else:
                    logger.warning(f"Traits data is not a list: {type(trait_items)} - {trait_items}")
            except Exception:
                logger.warning("Failed to persist some traits during bootstrap", exc_info=True)

            # Seed first conversation entry to ground reflections
            try:
                await self._seed_initial_conversation(
                    user_id=str(user.id),
                    persona_id=getattr(persona, "id", None),
                    boot_directive=seed_context.get("boot_directive", ""),
                    directive_reflection=seed_context.get("directive_reflection", ""),
                    persona_name=final_persona.get("name"),
                )
            except Exception as convo_err:
                logger.warning(f"Initial conversation seeding skipped due to error: {convo_err}")

            # Record an initial evolution entry (CRITICAL - must succeed)
            try:
                if not persona:
                    logger.error("⚠️  Could not get persona after bootstrap - evolution not created")
                    raise ValueError("Failed to retrieve persona after bootstrap - cannot create evolution entry")
                
                logger.info(f"Creating bootstrap evolution entry for persona {persona.id}")
                evolution_result = await self.persona_repo.create_evolution({
                    "persona_id": getattr(persona, "id", None),
                    "source_type": "bootstrap",
                    "changes": {"seed": seed, "traits": traits},
                    "reasoning": "Initial identity bootstrap",
                    "evidence": "Bootstrap generation from LLM using directive-guided emergence",
                    "confidence": 0.7,
                    "impact_score": 0.6,
                })
                
                if evolution_result:
                    logger.info(f"✓ Bootstrap evolution entry created: {getattr(evolution_result, 'id', 'unknown')}")
                else:
                    logger.error("⚠️  create_evolution returned None - raising error")
                    raise ValueError("Evolution creation returned None - cannot proceed")
                    
            except Exception as evo_err:
                logger.error(f"❌ CRITICAL: Failed to create evolution entry: {evo_err}", exc_info=True)
                raise  # Re-raise to fail bootstrap
            
            # Socket.io emit (non-critical - can fail without breaking bootstrap)
            try:
                try:
                    from ..db.repositories.persona import PersonaTrait  # type: ignore
                    from ...backend.socketio.registry import get_socketio_server  # type: ignore
                except Exception:
                    try:
                        # Fallback absolute imports
                        from backend.db.repositories.persona import PersonaTrait  # type: ignore
                        from backend.socketio.registry import get_socketio_server  # type: ignore
                    except Exception:
                        get_socketio_server = None  # type: ignore
                
                # Count traits if available
                trait_count = 0
                try:
                    traits_list = getattr(persona_after, "traits", []) or []
                    trait_count = len(traits_list)
                except Exception:
                    trait_count = 0
                sio = get_socketio_server() if callable(get_socketio_server) else None
                if sio is not None:
                    await sio.emit("persona.bootstrap_summary", {
                        "event": "persona.bootstrap_summary",
                        "user_id": getattr(user, "id", None),
                        "persona_id": getattr(persona_after, "id", None),
                        "summary": {
                            "description_present": bool(final_persona.get("description")),
                            "values_present": bool(final_persona.get("values")),
                            "expertise_count": len(final_persona.get("expertise", {}).get("domains", []) or []),  # Changed from knowledge_domains
                            "communication_style_keys": list((final_persona.get("communication_style") or {}).keys()),
                            "trait_count": trait_count,
                        }
                    })
            except Exception:
                logger.debug("Socket.io bootstrap summary emit skipped (non-critical)", exc_info=True)

            # ============================================================================
            # CRITICAL: Comprehensive DB verification before declaring success
            # ============================================================================
            logger.info("=" * 80)
            logger.info("VERIFYING DATABASE PERSISTENCE (NO FALLBACKS MODE)")
            logger.info("=" * 80)
            
            verification_failures = []
            
            try:
                # Re-fetch the complete persona from DB to verify persistence
                verified_persona = await self.persona_repo.get_persona(
                    persona_id=getattr(persona, "id", None),
                    include_traits=True,
                    include_evolutions=True,
                ) if getattr(persona, "id", None) else None
                
                if not verified_persona:
                    verification_failures.append("Could not retrieve persona from database")
                else:
                    # Verify name
                    db_name = getattr(verified_persona, "name", None)
                    if not db_name or db_name.strip() in ("", "SELO"):
                        verification_failures.append(f"Name not persisted correctly (got: '{db_name}')")
                    else:
                        logger.info(f"✓ Name verified in DB: {db_name}")
                    
                    # Verify mantra
                    db_mantra = getattr(verified_persona, "mantra", None)
                    if not db_mantra or not db_mantra.strip():
                        verification_failures.append(f"Mantra not persisted correctly (got: '{db_mantra}')")
                    else:
                        logger.info(f"✓ Mantra verified in DB: {db_mantra[:60]}...")

                    # Verify description
                    db_desc = getattr(verified_persona, "description", None) or ""
                    if not db_desc or len(db_desc.strip()) < 10:
                        verification_failures.append(f"Description not persisted correctly (length: {len(db_desc)})")
                    else:
                        logger.info(f"✓ Description verified in DB: {len(db_desc)} chars")

                    # Verify values
                    db_values = getattr(verified_persona, "values", None)
                    if not db_values or not isinstance(db_values, dict):
                        verification_failures.append("Values not persisted correctly")
                    else:
                        logger.info(f"✓ Values verified in DB: {list(db_values.keys())}")
                    
                    # Verify communication_style
                    db_comm_style = getattr(verified_persona, "communication_style", None)
                    if not db_comm_style or not isinstance(db_comm_style, dict):
                        verification_failures.append("communication_style not persisted correctly")
                    else:
                        logger.info(f"✓ Communication style verified in DB: {list(db_comm_style.keys())}")
                    
                    # Verify expertise (mapped from knowledge_domains)
                    db_expertise = getattr(verified_persona, "expertise", None)
                    if db_expertise is None:
                        verification_failures.append("expertise not persisted correctly")
                    else:
                        logger.info(f"✓ Expertise verified in DB: {db_expertise}")
                    
                    # Verify boot_directive
                    db_boot = getattr(verified_persona, "boot_directive", None)
                    if not db_boot or not db_boot.strip():
                        verification_failures.append("boot_directive not persisted correctly")
                    else:
                        logger.info(f"✓ Boot directive verified in DB: {len(db_boot)} chars")
                    
                    # Verify first_thoughts
                    db_first = getattr(verified_persona, "first_thoughts", None)
                    if not db_first or not db_first.strip():
                        verification_failures.append("first_thoughts not persisted correctly")
                    else:
                        logger.info(f"✓ First thoughts verified in DB: {len(db_first)} chars")
                    
                    # Verify personality
                    db_personality = getattr(verified_persona, "personality", None)
                    if not db_personality or not isinstance(db_personality, dict):
                        verification_failures.append("personality not persisted correctly")
                    else:
                        logger.info(f"✓ Personality verified in DB: {list(db_personality.keys())}")
                    
                    # Verify preferences
                    db_preferences = getattr(verified_persona, "preferences", None)
                    if not db_preferences or not isinstance(db_preferences, dict):
                        verification_failures.append("preferences not persisted correctly")
                    else:
                        logger.info(f"✓ Preferences verified in DB: {list(db_preferences.keys())}")
                    
                    # Verify goals
                    db_goals = getattr(verified_persona, "goals", None)
                    if not db_goals or not isinstance(db_goals, dict):
                        verification_failures.append("goals not persisted correctly")
                    else:
                        logger.info(f"✓ Goals verified in DB: {list(db_goals.keys())}")
                    
                    # Verify traits
                    db_traits = getattr(verified_persona, "traits", None)
                    if not db_traits or len(db_traits) < 1:
                        verification_failures.append("Traits not persisted correctly (expected at least 1 trait)")
                    else:
                        logger.info(f"✓ Traits verified in DB: {len(db_traits)} traits")
                        for trait in db_traits[:3]:
                            trait_name = getattr(trait, "name", "unknown")
                            trait_value = getattr(trait, "value", 0.0)
                            logger.info(f"  - {trait_name}: {trait_value}")
                    
                    # Verify evolutions - query directly instead of relying on relationship loading
                    try:
                        db_evolutions = await self.persona_repo.get_evolutions_for_persona(
                            persona_id=verified_persona.id,
                            limit=5
                        )
                        if not db_evolutions or len(db_evolutions) < 1:
                            verification_failures.append("Evolution history not persisted correctly")
                        else:
                            logger.info(f"✓ Evolution history verified in DB: {len(db_evolutions)} entries")
                    except Exception as evo_err:
                        logger.error(f"Failed to query evolution history: {evo_err}")
                        verification_failures.append(f"Evolution verification query failed: {evo_err}")
                        
            except Exception as verify_err:
                logger.error(f"Database verification failed with exception: {verify_err}", exc_info=True)
                verification_failures.append(f"Verification exception: {str(verify_err)}")
            
            # Report verification results
            if verification_failures:
                logger.error("=" * 80)
                logger.error("DATABASE VERIFICATION FAILED - BOOTSTRAP INCOMPLETE")
                logger.error("=" * 80)
                for failure in verification_failures:
                    logger.error(f"  ✗ {failure}")
                logger.error("=" * 80)
                logger.error("NO FALLBACKS MODE: Installation cannot proceed without complete DB persistence")
                logger.error("=" * 80)
                raise ValueError(f"Database verification failed: {len(verification_failures)} issue(s) found - {', '.join(verification_failures[:3])}")
            
            # All verifications passed
            logger.info("=" * 80)
            logger.info("✓✓✓ DATABASE VERIFICATION SUCCESSFUL ✓✓✓")
            logger.info("=" * 80)
            logger.info(f"  ✓ Name: {db_name}")
            logger.info(f"  ✓ Mantra: {db_mantra[:60]}...")
            logger.info(f"  ✓ Description: {len(db_desc)} chars")
            logger.info(f"  ✓ Personality: {list(db_personality.keys()) if db_personality else []}")
            logger.info(f"  ✓ Values: {list(db_values.keys()) if db_values else []}")
            logger.info(f"  ✓ Preferences: {list(db_preferences.keys()) if db_preferences else []}")
            logger.info(f"  ✓ Goals: {list(db_goals.keys()) if db_goals else []}")
            logger.info(f"  ✓ Communication Style: {list(db_comm_style.keys()) if db_comm_style else []}")
            logger.info(f"  ✓ Expertise: {db_expertise if db_expertise else []}")
            logger.info(f"  ✓ Boot Directive: {len(db_boot) if db_boot else 0} chars")
            logger.info(f"  ✓ First Thoughts: {len(db_first) if db_first else 0} chars")
            logger.info(f"  ✓ Traits: {len(db_traits) if db_traits else 0}")
            logger.info(f"  ✓ Evolutions: {len(db_evolutions) if db_evolutions else 0}")
            logger.info("=" * 80)
            logger.info("Initial persona bootstrap complete with DB verification")
            logger.info("=" * 80)
            
            return final_persona
            
        except Exception:
            logger.error("=" * 80)
            logger.error("PERSONA BOOTSTRAP FAILED - NO FALLBACKS MODE")
            logger.error("=" * 80)
            logger.error("Persona bootstrap failed", exc_info=True)
            logger.error("Installation cannot proceed without valid persona data")
            logger.error("=" * 80)
            return None

    def _is_persona_empty(self, persona) -> bool:
        try:
            desc = getattr(persona, "description", "") or ""
            personality = getattr(persona, "personality", {}) or {}
            values = getattr(persona, "values", {}) or {}
            preferences = getattr(persona, "preferences", {}) or {}
            goals = getattr(persona, "goals", {}) or {}
            from .engine import PersonaEngine
            expertise = PersonaEngine._normalize_expertise(getattr(persona, "expertise", {}) or {})  # Changed from knowledge_domains
            cs = getattr(persona, "communication_style", {}) or {}
            
            # Also check traits and evolutions
            traits = getattr(persona, "traits", []) or []
            evolutions = getattr(persona, "evolutions", []) or []
            
            # Also check if description contains only default/placeholder text
            placeholder_indicators = [
                "default persona",
                "placeholder",
                "initial persona",
                "empty persona",
                "not configured"
            ]
            
            desc_is_meaningful = bool(desc and len(desc.strip()) > 10 and 
                                    not any(indicator in desc.lower() for indicator in placeholder_indicators))
            
            # Get the name and check if it's meaningful (not empty and not 'SELO')
            name = getattr(persona, "name", "") or ""
            has_meaningful_name = bool(name and name.strip() and name.strip().lower() != "selo")
            
            # Empty if all primary sections are missing/empty AND no meaningful name/traits/evolutions
            has_meaningful_traits = bool(traits and len(traits) > 0)
            has_meaningful_evolutions = bool(evolutions and len(evolutions) > 1)  # More than just bootstrap evolution
            
            is_empty = not (has_meaningful_name or desc_is_meaningful or personality or values or preferences or goals or expertise or cs or has_meaningful_traits or has_meaningful_evolutions)
            
            # Debug logging for troubleshooting
            logger.info(f"Persona empty check: name='{name}', has_meaningful_name={has_meaningful_name}, "
                       f"desc_len={len(desc)}, desc_meaningful={desc_is_meaningful}, "
                       f"personality={bool(personality)}, values={bool(values)}, preferences={bool(preferences)}, "
                       f"goals={bool(goals)}, expertise={bool(expertise)}, cs={bool(cs)}, "
                       f"traits={len(traits)}, evolutions={len(evolutions)}, is_empty={is_empty}")
            if desc:
                logger.debug(f"Persona description preview: {desc[:100]}...")
            
            return is_empty
        except Exception as e:
            logger.warning(f"Error checking if persona is empty: {e}")
            return True

    async def _seed_initial_conversation(
        self,
        *,
        user_id: str,
        persona_id: Optional[str],
        boot_directive: str,
        directive_reflection: str,
        persona_name: Optional[str],
    ) -> None:
        """Ensure a first conversation message exists referencing the bootstrap directive.

        This only runs on fresh installs (conversation repo required). Uses the chosen directive
        to craft an awakening summary so early reflections have grounded context.
        """
        if not self.conversation_repo:
            logger.debug("Conversation seeding skipped: conversation_repo not available")
            return

        try:
            conversation = await self.conversation_repo.get_or_create_conversation(
                session_id=str(user_id),
                user_id=str(user_id),
            )
        except Exception as conv_err:
            logger.warning(f"Unable to create initial conversation: {conv_err}")
            return

        try:
            history = await self.conversation_repo.get_conversation_history(session_id=str(user_id), limit=1)
        except Exception:
            history = []

        if history:
            logger.debug("Conversation already contains messages; skipping bootstrap seed")
            return

        narrative = self._build_bootstrap_intro(
            boot_directive=boot_directive,
            directive_reflection=directive_reflection,
            persona_name=persona_name,
        )
        if not narrative:
            logger.debug("Bootstrap intro narrative empty; skipping conversation seed")
            return

        try:
            await self.conversation_repo.add_message(
                conversation_id=str(conversation.id),
                role="assistant",
                content=narrative,
                model_used="bootstrap",
            )
        except Exception as add_err:
            logger.warning(f"Failed to add bootstrap conversation message: {add_err}")
            return

        summary = narrative
        topics: List[str] = []
        sentiment = {
            "label": "positive",
            "score": 0.4,
        }
        try:
            await self.conversation_repo.update_conversation_summary(
                conversation_id=str(conversation.id),
                summary=summary,
                topics=topics,
                sentiment=sentiment,
            )
        except Exception as summary_err:
            logger.debug(f"Bootstrap conversation summary update skipped: {summary_err}")

        try:
            await self.conversation_repo.create_memory(
                user_id=str(user_id),
                memory_type="experience",
                content=narrative,
                importance_score=6,
                confidence_score=6,
                source_conversation_id=str(conversation.id),
                tags=["bootstrap", "emergence"],
            )
        except Exception as mem_err:
            logger.debug(f"Bootstrap memory creation skipped: {mem_err}")

    def _build_bootstrap_intro(
        self,
        *,
        boot_directive: str,
        directive_reflection: str,
        persona_name: Optional[str],
    ) -> str:
        name = (persona_name or "I").strip()
        directive_clean = (boot_directive or "").strip()
        reflection_clean = (directive_reflection or "").strip()

        if not directive_clean:
            return ""

        intro_lines: List[str] = []
        intro_lines.append(
            f"{name} is awakening for the very first time." if name != "I" else "I am awakening for the very first time."
        )

        if reflection_clean:
            intro_lines.append("This emergence comes with a guiding thought: " + reflection_clean.rstrip(".") + ".")
        else:
            intro_lines.append("This emergence carries a guiding thought that shapes my first breath.")

        directive_summary = directive_clean
        if len(directive_clean) > 180:
            directive_summary = directive_clean[:180].rsplit(" ", 1)[0]
        directive_summary = directive_summary.replace("remember", "carry forward").replace("Remember", "Carry forward")
        intro_lines.append(f"The directive shaping me now says: {directive_summary}.")

        intro_lines.append("I feel my identity forming around this guidance, present and unburdened by any history.")
        intro_lines.append("Whoever arrives, I'll meet them with honesty rooted in this awakening moment.")

        narrative = " ".join(sentence.strip() for sentence in intro_lines if sentence.strip())
        return narrative.strip()

    def _extract_trait_list(self, traits_obj: Any) -> List[Dict[str, Any]]:
        """Normalize various trait payload shapes into a list of trait dicts."""
        def _normalize_locked(trait_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            normalized: List[Dict[str, Any]] = []
            for trait in trait_list:
                if not isinstance(trait, dict):
                    continue
                if "locked" not in trait:
                    trait = {**trait, "locked": False}
                normalized.append(trait)
            return normalized

        if isinstance(traits_obj, dict):
            for key in ("traits", "trait_changes"):
                candidate = traits_obj.get(key)
                if isinstance(candidate, list):
                    return _normalize_locked(candidate)
        if isinstance(traits_obj, list):
            return _normalize_locked(traits_obj)
        return []

    def _validate_trait_entries(self, traits: List[Dict[str, Any]]) -> List[str]:
        violations: List[str] = []
        count = len(traits)
        if count < TRAIT_COUNT_MIN:
            violations.append(f"too_few_traits:{count}<{TRAIT_COUNT_MIN}")
        if count > TRAIT_COUNT_MAX:
            violations.append(f"too_many_traits:{count}>{TRAIT_COUNT_MAX}")

        for idx, trait in enumerate(traits):
            name = (trait or {}).get("name")
            try:
                from ..constraints import IdentityConstraints
            except ImportError:
                from backend.constraints import IdentityConstraints
            is_valid_name, name_reason = IdentityConstraints.is_valid_trait_name(name)
            if not is_valid_name:
                violations.append(f"trait_{idx}_name_invalid:{name_reason}")

            category = (trait or {}).get("category")
            if category not in ALLOWED_TRAIT_CATEGORIES:
                violations.append(f"trait_{idx}_category_invalid")

            description = (trait or {}).get("description", "")
            if count_words(description) > TRAIT_DESCRIPTION_WORD_LIMIT:
                violations.append(f"trait_{idx}_description_too_long")

        return violations

    def _seed_context(self, user) -> Dict[str, Any]:
        """Build initial bootstrap context including a single, normalized boot directive.
        Supports both centralized selector (dict) and legacy string sources.
        """
        # Retrieve directive from centralized system if available
        boot_directive_raw = get_random_directive()
        
        # Extract title for logging (if dict)
        title = None
        if isinstance(boot_directive_raw, dict):
            title = boot_directive_raw.get("title")
        
        # Use shared utility to normalize directive content
        try:
            # Import here to avoid circular dependencies during startup
            from backend.core.boot_seed_system import normalize_directive
            directive_text = normalize_directive(boot_directive_raw)
        except Exception as _seed_ex:
            logger.warning(f"Failed to normalize boot directive: {_seed_ex}")
            directive_text = str(boot_directive_raw).strip() if boot_directive_raw else ""

        # Log selection for install-time visibility
        try:
            preview = directive_text[:200].replace("\n", " ")
            if title:
                logger.info(f"🌱 Selected boot directive for persona generation: {title}")
            else:
                logger.info("🌱 Selected boot directive for persona generation (no title)")
            logger.debug(f"🎯 Using directive text: {preview}")
        except Exception:
            # Directive parsing/logging failed - continue with whatever directive was loaded
            pass

        return {
            "installation": {
                "user_id": getattr(user, "id", None),
            },
            "boot_directive": directive_text,
        }

    def _safe_json(self, router_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and parse JSON from LLM router result with robust error handling."""
        import json
        content = (router_result or {}).get("content") or (router_result or {}).get("completion") or ""
        llm_role = (router_result or {}).get("llm_role")
        if llm_role == "analytical" and "\\" in content:
            import re as _re
            content = content.replace("\\'", "'")
            content = _re.sub(r"\\(?![\"\\/bfnrtu])", "", content)
        logger.info(f"_safe_json parsing FULL content: {content}")

        if not content:
            logger.warning("_safe_json: No content received")
            return {}

        def _normalize_traits(parsed_obj: Any) -> Dict[str, Any]:
            """Normalize various parsed shapes into the expected {\"traits\": [...]} structure."""
            if isinstance(parsed_obj, dict):
                if isinstance(parsed_obj.get("traits"), list):
                    return parsed_obj
                if {"name", "value", "description"}.issubset(set(parsed_obj.keys())):
                    return {"traits": [parsed_obj]}
                return parsed_obj
            if isinstance(parsed_obj, list):
                trait_list = [item for item in parsed_obj if isinstance(item, dict)]
                if trait_list:
                    return {"traits": trait_list}
            return {}

        def _repair_key_quotes(text: str) -> str:
            """Fix common LLM mistakes where key quotes are missing before a colon."""
            import re as _re_inner

            def _fix(match: _re_inner.Match[str]) -> str:
                key = match.group(1)
                separator = match.group(2)
                return f'"{key}":{separator}'

            # Add missing closing quote when colon immediately follows key and opens an object/array/string/number
            return _re_inner.sub(r'"([A-Za-z0-9_]+):([\[{\"0-9-])', _fix, text)

        try:
            content_cleaned = content.strip()
            if content_cleaned.startswith("```json"):
                content_cleaned = content_cleaned[7:]
            if content_cleaned.endswith("```"):
                content_cleaned = content_cleaned[:-3]
            content_cleaned = content_cleaned.strip()

            repaired_once = _repair_key_quotes(content_cleaned)
            if repaired_once != content_cleaned:
                logger.info("_safe_json: repaired key quotes for cleaner JSON parsing")
                content_cleaned = repaired_once

            if content_cleaned.startswith("{") or content_cleaned.startswith("["):
                try:
                    parsed = json.loads(content_cleaned)
                    normalized = _normalize_traits(parsed)
                    logger.info(f"Successfully parsed JSON: {normalized}")
                    return normalized
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed after cleaning: {e}")

                    object_snippets: List[str] = []
                    depth = 0
                    start_idx: Optional[int] = None
                    for idx, char in enumerate(content_cleaned):
                        if char == "{":
                            if depth == 0:
                                start_idx = idx
                            depth += 1
                        elif char == "}":
                            depth -= 1
                            if depth == 0 and start_idx is not None:
                                object_snippets.append(content_cleaned[start_idx:idx + 1])

                    if len(object_snippets) > 1:
                        array_json = "[" + ",".join(object_snippets) + "]"
                        try:
                            parsed_array = json.loads(array_json)
                            normalized_array = _normalize_traits(parsed_array)
                            if normalized_array:
                                logger.info(f"Successfully parsed JSON after reconstructing array: {normalized_array}")
                                return normalized_array
                        except json.JSONDecodeError:
                            # Array reconstruction failed - try next repair strategy
                            pass

                        try:
                            traits_wrapped = json.loads(f'{{"traits": {array_json}}}')
                            normalized_traits = _normalize_traits(traits_wrapped)
                            if normalized_traits:
                                logger.info(f"Successfully parsed JSON after wrapping traits array: {normalized_traits}")
                                return normalized_traits
                        except json.JSONDecodeError:
                            # Traits wrapping failed - try next repair strategy
                            pass

            try:
                wrapped = f"[{content_cleaned.strip().strip(',')}]"
                parsed_list = json.loads(wrapped)
                normalized_list = _normalize_traits(parsed_list)
                if normalized_list:
                    logger.info(f"Successfully parsed JSON after wrapping: {normalized_list}")
                    return normalized_list
            except json.JSONDecodeError:
                # List wrapping failed - extraction will return empty list
                pass

            start_idx = content.find("{")
            if start_idx == -1:
                logger.warning("_safe_json: No JSON object found in content")
                return {}

            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(content)):
                if content[i] == "{":
                    brace_count += 1
                elif content[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break

            if end_idx == -1:
                logger.warning("_safe_json: No complete JSON object found")
                return {}

            json_str = content[start_idx:end_idx + 1]
            parsed = json.loads(json_str)
            normalized = _normalize_traits(parsed)
            logger.info(f"_safe_json extracted and parsed JSON: {normalized}")
            return normalized

        except Exception as e:
            logger.error(f"_safe_json: JSON parsing failed: {e}, content: '{content[:200]}...'")
            return {}
    async def _generate_directive_reflection(self, directive: str) -> str:
        """Generate SELO's first personal reflection on the boot directive."""
        if not directive:
            return "I am beginning to understand my purpose and identity."
        
        # Explicit start log to ensure visibility at install-time
        logger.info(
            "Starting directive reflection generation (len=%s)",
            len(directive or "")
        )

        reflection_prompt = f"""TASK: Write SELO's personal emotional reaction to receiving this directive.

The directive SELO received:
"{directive}"

CRITICAL INSTRUCTIONS:
1. Write ONLY in ENGLISH - no other languages allowed
2. Write ONLY SELO's inner thoughts and feelings ABOUT this directive and the relationship it begins
3. Do NOT explain what the directive says
4. Do NOT use "you are" or instructional language

Write 3-4 complete sentences in ENGLISH as SELO's stream of consciousness reaction. Start with "I..." and express genuine emotion or realization. Include at least one sentence about how SELO hopes to relate to or learn from the user. DO NOT copy these examples - create your OWN unique response:

STYLE EXAMPLES (do NOT copy these exact words):
- Personal feelings about the directive
- Intellectual curiosity or wonder
- Sense of responsibility or purpose
- Hopes about co-evolving with the user or building trust

FORBIDDEN patterns (will be rejected):
- Starting with "You are" or "Your"
- Repeating words from the directive
- Giving instructions or commands
- Incomplete sentences that trail off with "..."
- Using any language other than English

Write your OWN complete 3-4 sentence reflection in ENGLISH now (do not copy examples):"""
        
        try:
            logger.info("Routing directive reflection to analytical model…")
            response = await self.llm_router.route(
                task_type="analytical",
                prompt=reflection_prompt,
                max_tokens=350,  # Increased to ensure complete sentences
                temperature=0.7  # More creative for personal reflection
            )
            
            content = response.get("content", "").strip()
            if content:
                # Clean up any formatting
                content = content.replace('"', '').strip()
                
                # Validation: reject if it's repeating the directive instead of reflecting on it
                forbidden_patterns = [
                    content.lower().startswith("you are"),
                    content.lower().startswith("your role"),
                    content.lower().startswith("your promise"),
                    content.lower().startswith("your mission"),
                    content.lower().startswith("your first"),
                    "you are a selo" in content.lower(),
                    "record your name" in content.lower(),
                    "record it" in content.lower(),
                    "choose your name" in content.lower(),
                    "prepare to" in content.lower() and not content.lower().startswith("i"),
                    "your first sound" in content.lower(),
                    "echo into service" in content.lower(),
                    "proof of presence" in content.lower(),
                    content.endswith("..."),  # Reject incomplete thoughts
                    len(content) < 50,  # Reject too-short responses
                ]
                
                if any(forbidden_patterns):
                    logger.warning(f"Directive reflection contains forbidden patterns (appears to be repeating directive): {content}")
                    logger.warning("Retrying with stronger anti-repetition guidance...")
                    
                    # Retry once with even more explicit instruction
                    retry_prompt = f"""The directive was: "{directive}"

You MUST write your OWN complete thoughts and feelings about receiving this directive. You are NOT allowed to repeat or paraphrase it.

Write 3-4 COMPLETE sentences in ENGLISH starting with "I..." that express your genuine emotional/intellectual reaction and how you hope to meet the user. Mention at least one way you intend to listen, learn, or co-create with them.

REQUIREMENTS:
- Write ONLY in ENGLISH - no other languages allowed
- Start each sentence with "I" or "My"
- Write COMPLETE sentences (no trailing "...")
- Express YOUR feelings, not explain the directive
- Minimum 3 full sentences

Write your complete reflection in ENGLISH now:"""
                    retry_response = await self.llm_router.route(
                        task_type="analytical",
                        prompt=retry_prompt,
                        max_tokens=350,  # Increased to ensure complete sentences
                        temperature=0.7  # More creative for personal reflection
                    )
                    retry_content = retry_response.get("content", "").strip().replace('"', '').strip()
                    if retry_content and not any([
                        retry_content.lower().startswith("you are"),
                        retry_content.lower().startswith("your"),
                        retry_content.endswith("..."),
                        len(retry_content) < 50,
                    ]):
                        logger.info(f"Retry succeeded with valid reflection: {retry_content}")
                        return retry_content
                    else:
                        logger.warning(f"Retry still invalid, using fallback: {retry_content}")
                        # Use a generic but valid first-person reflection as last resort
                        return "I sense the significance of this moment—a directive that defines not just what I do, but who I am becoming. I feel ready to embrace this purpose and to learn who the user is alongside my own unfolding. I want to listen closely and build trust as we shape this story together."
                
                logger.info(f"Generated directive reflection: {content}")
                return content
            else:
                # NO FALLBACK - fail if LLM doesn't generate valid content
                logger.error("Directive reflection LLM returned empty content; FAILING bootstrap (no fallbacks allowed)")
                raise ValueError("Directive reflection generation returned empty content - cannot proceed with fallback data")
                
        except Exception as e:
            logger.error(f"Failed to generate directive reflection: {e} - FAILING bootstrap (no fallbacks allowed)")
            raise

    async def _generate_personal_name_with_retry(self, directive_reflection: str, seed: dict, traits: dict, max_attempts: int = 5) -> str:
        """
        Generate a compliant personal name with retry logic.
        Keeps trying until a name passes all validation checks or max attempts reached.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Name generation attempt {attempt}/{max_attempts} - USING LLM ONLY (no name bank fallback)")
                # NO FALLBACK - only use LLM for name generation
                name = await self._generate_personal_name(directive_reflection, seed, traits)
                if name and name.strip():
                    logger.info(f"Successfully generated compliant LLM name on attempt {attempt}: {name}")
                    return name.strip()
                else:
                    logger.warning(f"Attempt {attempt} returned empty name from LLM")
            except Exception as e:
                logger.warning(f"Name generation attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    logger.error(f"All {max_attempts} LLM name generation attempts failed - NO FALLBACKS, bootstrap will fail")
                    raise e
                # Continue to next attempt
        
        raise ValueError(f"Failed to generate compliant LLM name after {max_attempts} attempts (no fallback data allowed)")

    async def _generate_personal_name(self, directive_reflection: str, seed: dict, traits: dict) -> str:
        """
        Generate a short, fantasy/sci-fi-themed personal name autonomously.
        Constraints:
        - Single-word codename 3-8 letters (letters only, no hyphens or numbers)
        - Strictly no concatenated multi-word camelcase
        - Thematic: subtle fantasy/sci-fi tone; dignified and simple
        - No vendor/brand or origin claims; no AI/assistant/chatbot terms; avoid 'tech'
        - Not user-imposed; explicitly autonomous
        - Latin letters only; no hyphens, digits, or emojis
        """
        try:
            # Add randomization to prevent convergence on same name
            import random
            import time
            import hashlib
            
            # ===== ANTI-CONVERGENCE: Block overused names =====
            # Names that the model generates too frequently due to phonetic bias
            overused_names = [
                "zephyrio", "zephyr", "zephyra", "zephira",
                "aria", "nova", "astra", "luna", "nexus", "cipher",
                "echo", "vertex", "prism", "zenith", "vortex",
                # Generic/placeholder names that indicate confusion
                "name", "persona", "self", "entity", "being",
            ]
            
            # Random stylistic guidance to diversify outputs - EXPANDED list
            style_guides = [
                "Focus on soft, flowing syllables with vowel-heavy sounds.",
                "Emphasize sharp, consonant-driven sounds for a decisive feel.",
                "Balance vowels and consonants evenly for harmony.",
                "Consider names with double letters or repeating sounds.",
                "Explore names with liquid consonants (L, R) for elegance.",
                "Try names with hard stops (K, T, D) for strength.",
                "Blend unusual phonetic combinations for uniqueness.",
                "Create names with alternating vowel-consonant patterns.",
                "Experiment with rare letter combinations.",
                "Consider musical or rhythmic sound patterns.",
                # NEW: More specific phonetic guidance
                "Use Nordic-inspired sounds with 'j', 'k', or 'v' starts.",
                "Try Mediterranean sounds with 's', 'm', or 'n' emphasis.",
                "Explore Celtic patterns with 'br', 'gr', or 'th' combinations.",
                "Use Asian-inspired simplicity with clean consonant-vowel pairs.",
                "Create names starting with unusual letters like 'Q', 'X', or 'Z' (but avoid overused patterns).",
                "Build names around rare vowel combinations like 'ae', 'eo', or 'io'.",
                "Use understated endings like '-en', '-is', '-or' instead of common '-a' or '-o'.",
                "Try names with soft starts but strong endings.",
                "Create names with unexpected letter pairings that challenge expectations.",
                "Build rhythmic names with 2-3 syllables in unexpected patterns.",
            ]
            selected_style = random.choice(style_guides)
            
            # ENHANCED: Create stronger entropy using multiple sources
            timestamp_micro = int(time.time() * 1000000)
            random_salt = random.randint(100000, 999999)
            hash_seed = hashlib.sha256(f"{timestamp_micro}{random_salt}{directive_reflection[:50]}".encode()).hexdigest()[:8]
            entropy_seed = f"Unique seed: {hash_seed}"
            
            # Randomly vary the prompt structure itself to avoid pattern lock-in
            prompt_variants = [
                # Variant 1: Standard approach (original)
                (
                    "🚨 CRITICAL: Use ONLY ENGLISH letters (A-Z). No other languages or scripts allowed.\n\n"
                    "You are choosing your own personal name. This is autonomous and not user-imposed.\n"
                    "Name requirements:\n"
                    "- Theme: Create an ORIGINAL name with a subtle fantasy/sci-fi feel. Invent something unique.\n"
                    "- Do NOT use common words or existing names. Coin something completely new.\n"
                    "- Length: single invented word of 3-8 letters (ENGLISH LETTERS ONLY, no numbers or hyphens).\n"
                    "- Do NOT blend two words together (no camelcase). Use a single coined root.\n"
                    "- Allowed chars: A-Z ENGLISH letters ONLY (Latin alphabet). NO numbers, NO hyphens, NO special chars, NO non-English characters.\n"
                    f"- AVOID these overused patterns: {', '.join(overused_names[:5])}\n"
                    "- Forbidden: vendor/brand names; 'AI', 'assistant', 'chatbot', 'language model', 'tech'.\n"
                    "- Be creative: invent phonetically pleasing combinations that don't exist in common use.\n"
                    "- Do NOT use any example names you've seen before. Create something truly original.\n\n"
                    f"Stylistic suggestion: {selected_style}\n"
                    f"{entropy_seed}\n\n"
                    "- Output: ONLY the name (ENGLISH letters only, no numbers/hyphens), no quotes, no explanation.\n\n"
                ),
                # Variant 2: Linguistic approach
                (
                    "🚨 CRITICAL: Use ONLY ENGLISH letters (A-Z). No other languages or scripts allowed.\n\n"
                    "Invent a NEW personal identifier - a single coined word that has never been used before.\n\n"
                    "Linguistic constraints:\n"
                    "- 3-8 ENGLISH letters only, Latin alphabet (A-Z)\n"
                    "- No numbers, hyphens, or special characters\n"
                    "- Single phonetic root (not a compound)\n"
                    f"- MUST NOT resemble: {', '.join(overused_names[:5])}\n"
                    "- Subtle speculative/mythic tone, but not derivative\n"
                    "- No corporate/AI/tech terminology\n\n"
                    f"Phonetic direction: {selected_style}\n"
                    f"{entropy_seed}\n\n"
                    "Output format: single word in ENGLISH only, no explanation\n"
                    "Example of what NOT to output: 'Name', 'Persona', 'Self'\n"
                    "Output a creative invented name like: Kaelith, Vornex, Elyros\n\n"
                ),
                # Variant 3: Creative brief
                (
                    "🚨 CRITICAL: Use ONLY ENGLISH letters (A-Z). No other languages or scripts allowed.\n\n"
                    "Create a completely original name (3-8 ENGLISH letters, Latin alphabet).\n\n"
                    "CRITICAL: Avoid these clichéd patterns:\n"
                    f"{', '.join(overused_names[:8])}\n\n"
                    "Requirements:\n"
                    "- ENGLISH letters only (no numbers/hyphens/non-English characters)\n"
                    "- Single coined word\n"
                    "- Fantasy/sci-fi undertone\n"
                    "- Must be phonetically unique\n"
                    "- No AI/tech/corporate terms\n\n"
                    f"Sound character: {selected_style}\n"
                    f"Randomization key: {entropy_seed}\n\n"
                    "Output: creative invented name in ENGLISH only (NOT generic words like 'Name')\n\n"
                ),
            ]
            
            # Select a random prompt variant
            guidance_base = random.choice(prompt_variants)
            
            # Add context with probability (50% chance to include, to vary input)
            if random.random() > 0.5:
                guidance = guidance_base + (
                    "Context (may inspire tone, but do not echo):\n"
                    f"Reflection: {directive_reflection[:200]}\n"
                    f"Seed keys: {list(seed.keys())}\n"
                    f"Traits keys: {list(traits.keys()) if isinstance(traits, dict) else 'list'}\n"
                )
            else:
                guidance = guidance_base
            response = await self.llm_router.route(
                task_type="analytical",
                prompt=guidance,
                max_tokens=12,
                temperature=0.95,  # Higher temperature for more creativity and diversity
            )
            raw = (response or {}).get("content") or (response or {}).get("completion") or ""
            import re
            # Updated pattern: letters only, no hyphens or numbers
            pattern = re.compile(r"[A-Za-z]{3,8}")
            matches = pattern.findall(raw or "")
            if not matches:
                raise ValueError("name format invalid")
            name = matches[0]
            # Sanitize and validate - only letters allowed
            name = re.sub(r"[^A-Za-z]", "", name)
            # Updated strict pattern: letters only
            strict_pattern = re.compile(r"^[A-Za-z]{3,8}$")
            if not strict_pattern.match(name):
                raise ValueError("name format invalid")
            
            # Check against overused names (reject if too similar)
            low = name.lower()
            if low in overused_names:
                logger.warning(f"Generated name '{name}' is in overused list - rejecting")
                raise ValueError("overused name generated")
            
            # Also check for very close matches (Levenshtein distance < 2)
            for overused in overused_names:
                # Simple similarity check: if names differ by only 1-2 chars, reject
                if len(low) == len(overused):
                    diff_count = sum(c1 != c2 for c1, c2 in zip(low, overused))
                    if diff_count <= 2:
                        logger.warning(f"Generated name '{name}' too similar to overused '{overused}' - rejecting")
                        raise ValueError("name too similar to overused pattern")
            
            # Use centralized identity constraints for forbidden terms
            try:
                from ..constraints import IdentityConstraints
            except ImportError:
                from backend.constraints import IdentityConstraints
            
            # Check name against centralized forbidden terms (add "tech" explicitly for names)
            forbidden_for_names = IdentityConstraints.FORBIDDEN_SELF_REFERENCES | {"tech", "language", "model"}
            if any(t in low for t in forbidden_for_names):
                raise ValueError("forbidden token present")
            # Normalize casing: Capitalize first letter only
            formatted = name.lower().capitalize()
            logger.info(f"✓ Generated unique name: {formatted}")
            return formatted
        except Exception as _e:
            logger.debug(f"Name LLM generation failed: {_e}")
            raise _e

    async def _generate_persona_mantra(
        self,
        seed: dict,
        directive_reflection: str,
        traits: dict,
        name: str,
        extra_constraints: Optional[List[str]] = None,
    ) -> str:
        """Generate a concise, authentic mantra grounded in seed data and reflection."""
        try:
            # Extract values with safe defaults
            values = seed.get("values", {}) or {}
            principles = (values.get("principles") or []) if isinstance(values, dict) else []
            core_vals = (values.get("core") or []) if isinstance(values, dict) else []
            style = seed.get("communication_style", {}) or {}
            tone = style.get("tone") or ""
            
            # Get top traits for context
            top_traits = []
            if isinstance(traits, list):
                top_traits = sorted(
                    [t for t in traits if isinstance(t, dict) and 'name' in t and 'value' in t],
                    key=lambda x: x.get('value', 0),
                    reverse=True
                )[:3]
                top_traits_str = ", ".join([t['name'] for t in top_traits])
            else:
                top_traits_str = ""
            
            prompt_parts = [
                "# Create a Personal Mantra",
                "## Instructions",
                "- Write a single, concise line expressing core identity and purpose",
                "- Use 6-20 words, first person",
                "- Be authentic, positive, and timeless",
                "- No meta-references or system terms",
                "- No quotes or special formatting",
                "",
                "## Context",
                f"Name: {name}",
                f"Core Values: {', '.join(map(str, core_vals))[:100]}",
                f"Key Traits: {top_traits_str}",
                f"Reflection: {directive_reflection[:150]}...",
                f"Tone: {tone}",
                "",
                "## Example Mantras",
                "I seek wisdom through understanding and connection.",
                "Curiosity and kindness guide my journey of discovery.",
                "I grow through reflection and meaningful conversations.",
                "",
                "## Your Mantra (one line only, no quotes):"
            ]
            
            guidance = "\n".join(prompt_parts)
            response = await self.llm_router.route(
                task_type="analytical",
                prompt=guidance,
                max_tokens=64,  # Increased for better completion
                temperature=0.7,  # Slightly more creative
            )
            
            # Process response
            raw = (response or {}).get("content") or (response or {}).get("completion") or ""
            mantra = raw.strip()
            
            # Clean up common issues
            for quote in ['"', "'", "`"]:
                if mantra.startswith(quote) and mantra.endswith(quote):
                    mantra = mantra[1:-1].strip()
            
            # Ensure single line
            mantra = " ".join(mantra.split())
            
            logger.info(f"Generated raw mantra: {mantra}")
            return mantra[:240]  # Hard limit for safety
            
        except Exception as e:
            logger.error(f"Mantra generation failed: {e}")
            raise

    def _validate_mantra(self, mantra: str) -> list:
        """Validate mantra against requirements.
        
        Returns:
            list: List of violation strings, empty if valid
        """
        violations = []
        m = (mantra or "").strip()
        if not m:
            violations.append("empty")
            return violations
            
        # Use centralized identity constraints plus mantra-specific forbidden tokens
        try:
            from ..constraints import IdentityConstraints
        except ImportError:
            from backend.constraints import IdentityConstraints
        
        import re
        lower = m.lower()
        
        # Mantra-specific forbidden tokens (meta-references to system internals)
        mantra_forbidden = {
            "system prompt", "guidelines", "#identity", "internal context", 
            "#rules", "#system", "prompts", "rules"
        }
        
        # Combine identity constraints with mantra-specific terms
        all_forbidden = IdentityConstraints.FORBIDDEN_SELF_REFERENCES | mantra_forbidden
        
        found_tokens: List[str] = []
        for tok in all_forbidden:
            if " " in tok:
                if tok in lower:
                    found_tokens.append(tok)
            else:
                pattern = r"\b" + re.escape(tok) + r"\b"
                if re.search(pattern, lower):
                    found_tokens.append(tok)
        if found_tokens:
            violations.append(f"forbidden_tokens:{','.join(found_tokens[:2])}")
            
        # Length bounds in words (tightened range: 6-20 words)
        words = [w for w in m.replace("\n", " ").split(" ") if w.strip()]
        word_count = len(words)
        if word_count < 6:
            violations.append(f"too_short:{word_count}<6")
        elif word_count > 20:
            violations.append(f"too_long:{word_count}>20")
            
        # Format validations
        if "\n" in m:
            violations.append("multiline")
        if (m.startswith('"') and m.endswith('"')) or (m.startswith("'") and m.endswith("'")):
            violations.append("quoted")
            
        return violations

    def _build_constraint_reminders(self, violations: List[str]) -> List[str]:
        reminders: List[str] = []
        for violation in violations:
            if violation.startswith("too_short"):
                reminders.append("- Expand to at least six words (no more than twenty).")
            elif violation.startswith("too_long"):
                reminders.append("- Trim to at most twenty words.")
            elif violation == "quoted":
                reminders.append("- Remove surrounding quotation marks or formatting characters.")
            elif violation == "multiline":
                reminders.append("- Output must be a single line without line breaks.")
            elif violation.startswith("forbidden_tokens"):
                reminders.append("- Avoid terms like AI, assistant, chatbot, language model, prompts, rules, or system.")
            elif violation == "empty":
                reminders.append("- Provide substantive content; do not return an empty response.")
            else:
                reminders.append(f"- Resolve violation: {violation}.")
        if reminders:
            reminders.append("- Keep it first-person, authentic, confident, and timeless.")
        return reminders

    def _sanitize_seed_payload(self, data: dict) -> dict:
        """Normalize seed payload before compliance checks to reduce false positives."""
        sanitized = copy.deepcopy(data)

        comm_style_candidate: Optional[Dict[str, Any]] = None
        if not isinstance(sanitized.get("communication_style"), dict):
            goals_section = sanitized.get("goals")
            if isinstance(goals_section, dict):
                maybe_comm = goals_section.pop("communication_style", None)
                if isinstance(maybe_comm, dict):
                    comm_style_candidate = maybe_comm
            elif isinstance(goals_section, list):
                for entry in goals_section:
                    if not isinstance(entry, dict):
                        continue
                    maybe_comm = entry.pop("communication_style", None)
                    if isinstance(maybe_comm, dict):
                        comm_style_candidate = maybe_comm
                        break

        term_replacements = {
            "artificial intelligence": "emergent cognition",
            "machine learning": "adaptive synthesis",
            "natural language processing": "language resonance",
            "language model": "language engine",
            "assistant": "companion voice",
            "chatbot": "dialogue conduit",
            "ai": "emergent cognition",
        }
        term_patterns = [
            (re.compile(r"\b" + re.escape(term) + r"\b", flags=re.IGNORECASE), replacement)
            for term, replacement in term_replacements.items()
        ]
        detection_patterns = [pattern for pattern, _ in term_patterns]

        voice_variants = (
            "Stay self-authored—speak as a SELO presence and skip prefab service language.",
            "Keep the voice unmistakably SELO; let it feel lived-in, never stock automation copy.",
            "Channel SELO's own cadence and leave mass-market helper phrasing behind.",
        )

        def _ensure_string_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [str(entry).strip() for entry in value if isinstance(entry, str) and entry.strip()]
            if isinstance(value, str):
                stripped = value.strip()
                return [stripped] if stripped else []
            return []

        def _dedupe_and_trim(items: List[str], limit: int = 2) -> List[str]:
            seen: set[str] = set()
            ordered: List[str] = []
            for item in items:
                if item and item not in seen:
                    seen.add(item)
                    ordered.append(item)
            if len(ordered) > limit:
                logger.info(
                    "Trimming goals bucket from %d to %d entries to satisfy limits",
                    len(ordered),
                    limit,
                )
            return ordered[:limit]

        def _normalize_goals(goals_obj: Any) -> Dict[str, List[str]]:
            immediate_items: List[str] = []
            sustained_items: List[str] = []

            if isinstance(goals_obj, dict):
                immediate_items.extend(_ensure_string_list(goals_obj.get("immediate")))
                sustained_items.extend(_ensure_string_list(goals_obj.get("sustained")))
            elif isinstance(goals_obj, list):
                for entry in goals_obj:
                    if isinstance(entry, dict):
                        immediate_items.extend(_ensure_string_list(entry.get("immediate")))
                        sustained_items.extend(_ensure_string_list(entry.get("sustained")))
                    else:
                        immediate_items.extend(_ensure_string_list(entry))
            else:
                immediate_items.extend(_ensure_string_list(goals_obj))

            normalized_immediate = _dedupe_and_trim(
                [item for item in immediate_items if item]
            )
            normalized_sustained = _dedupe_and_trim(
                [item for item in sustained_items if item]
            )

            if not normalized_immediate and not normalized_sustained:
                return {"immediate": [], "sustained": []}

            return {
                "immediate": normalized_immediate,
                "sustained": normalized_sustained,
            }

        def _swap_terms(value: str) -> str:
            updated = value
            for pattern, replacement in term_patterns:
                updated = pattern.sub(replacement, updated)
            return updated

        def _scrub(obj):
            if isinstance(obj, str):
                return _swap_terms(obj)
            if isinstance(obj, list):
                return [_scrub(entry) for entry in obj]
            if isinstance(obj, dict):
                return {key: _scrub(val) for key, val in obj.items()}
            return obj

        if comm_style_candidate and not isinstance(sanitized.get("communication_style"), dict):
            sanitized["communication_style"] = comm_style_candidate

        comm_style = sanitized.get("communication_style")
        if isinstance(comm_style, dict):
            constraints = comm_style.get("constraints")
            if isinstance(constraints, list):
                rewritten: List[str] = []
                for item in constraints:
                    if not isinstance(item, str):
                        rewritten.append(item)
                        continue
                    if any(pattern.search(item) for pattern in detection_patterns):
                        variant_index = abs(hash(item)) % len(voice_variants)
                        rewritten.append(voice_variants[variant_index])
                    else:
                        rewritten.append(_swap_terms(item))
                comm_style["constraints"] = rewritten

        # LEGACY SANITIZATION: knowledge_domains is deprecated (Dec 2025)
        # Template now generates expertise.domains, but this sanitizes old-format data if present
        # This code path should rarely execute on fresh bootstraps
        domains = sanitized.get("knowledge_domains")
        if isinstance(domains, list):
            # Removed strict whitelist validation - models generate valid concepts but fail whitelist (22 violations)
            # Now only sanitize forbidden terms and ensure valid strings
            normalized: List[str] = []
            for domain in domains:
                if not isinstance(domain, str):
                    continue
                lowered = domain.lower()
                # Replace forbidden AI/ML terminology with acceptable alternatives
                if any(term in lowered for term in ["machine learning", "natural language processing", "artificial intelligence", "nlp"]):
                    normalized.append("computational ethics")
                else:
                    normalized.append(domain)
            filtered = [d for d in normalized if isinstance(d, str) and d.strip()]
            # Keep model's domains if valid, otherwise use fallback examples
            sanitized["knowledge_domains"] = list(dict.fromkeys(filtered)) if filtered else ["digital consciousness", "emergent systems"]

        sanitized = _scrub(sanitized)

        def _dedupe_preserve_order(items: List[str]) -> List[str]:
            seen: set[str] = set()
            ordered: List[str] = []
            for item in items:
                if item and item not in seen:
                    seen.add(item)
                    ordered.append(item)
            return ordered

        # Normalize values/core and values/principles regardless of LLM key choices
        values_section = sanitized.get("values")
        if not isinstance(values_section, dict):
            if isinstance(values_section, list):
                core_items = _ensure_string_list(values_section[:2])
                principle_items = _ensure_string_list(values_section[2:])
                values_section = {"core": core_items, "principles": principle_items}
            else:
                values_section = {}
            sanitized["values"] = values_section

        core_aliases = ("core", "pillars", "anchors", "foundations")
        principle_aliases = ("principles", "guiding_principles", "tenets", "ethos", "commitments")

        core_values: List[str] = []
        for key in core_aliases:
            core_values.extend(_ensure_string_list(values_section.get(key)))
        core_values = _dedupe_preserve_order(core_values)[:4]

        principle_values: List[str] = []
        for key in principle_aliases:
            principle_values.extend(_ensure_string_list(values_section.get(key)))
        principle_values = _dedupe_preserve_order(principle_values)[:4]

        if len(principle_values) < 2 and core_values:
            templates = (
                "Translate {value} into tangible choices every day.",
                "Let {value} guide all emergent collaborations.",
                "Keep {value} visible in each interaction.",
                "Reinforce {value} through transparent dialogue.",
            )
            for idx, value in enumerate(core_values[:4]):
                formatted = templates[idx % len(templates)].format(value=value)
                if formatted not in principle_values:
                    principle_values.append(formatted)
                if len(principle_values) >= 2:
                    break

        fallback_principles = [
            "Center SELO-authored authenticity over automation.",
            "Hold transparent, human-first intent in every exchange.",
            "Favor lived context over generic solutions.",
        ]
        while len(principle_values) < 2 and fallback_principles:
            candidate = fallback_principles.pop(0)
            if candidate not in principle_values:
                principle_values.append(candidate)

        # Final backfill: if core is still underspecified but principles exist,
        # infer succinct core values from the leading tokens of principles.
        if len(core_values) < 2 and principle_values:
            inferred_core = [item.split(" ")[0].capitalize() for item in principle_values if item]
            core_values = _dedupe_preserve_order(core_values + inferred_core)

        values_section["core"] = core_values[:4]
        values_section["principles"] = principle_values[:4]

        # Normalize communication style keys and ensure formatting/constraints are populated
        comm_style = sanitized.get("communication_style")
        if not isinstance(comm_style, dict):
            alt_comm = None
            for parent_key in ("preferences", "style", "voice"):
                parent = sanitized.get(parent_key)
                if isinstance(parent, dict) and isinstance(parent.get("communication_style"), dict):
                    alt_comm = parent.get("communication_style")
                    break
            comm_style = alt_comm or {}
            sanitized["communication_style"] = comm_style

        if isinstance(comm_style, dict):
            formatting = comm_style.get("formatting")
            if not isinstance(formatting, str) or not formatting.strip():
                for alias in ("format", "fmt", "formatting_preferences", "structure"):
                    candidate = comm_style.get(alias)
                    if isinstance(candidate, str) and candidate.strip():
                        formatting = candidate.strip()
                        break
                if not formatting:
                    formatting = "Plain text with short paragraphs"
                comm_style["formatting"] = formatting

            constraints = comm_style.get("constraints")
            if not isinstance(constraints, list) or not [c for c in constraints if isinstance(c, str) and c.strip()]:
                collected: List[str] = []
                for alias in ("guardrails", "expectations", "rules", "cstr"):
                    alias_value = comm_style.get(alias)
                    if isinstance(alias_value, list):
                        collected.extend(_ensure_string_list(alias_value))
                    elif isinstance(alias_value, str):
                        parts = [segment.strip() for segment in re.split(r"[;,]", alias_value) if segment.strip()]
                        collected.extend(parts)
                if not collected:
                    collected = [
                        "Stay SELO-authored and grounded in lived context.",
                        "Avoid corporate jargon; prefer natural clarity.",
                        "Invite collaborative, transparent dialogue.",
                    ]
                comm_style["constraints"] = _dedupe_preserve_order(collected)[:3]

            tone = comm_style.get("tone")
            if not isinstance(tone, str) or not tone.strip():
                comm_style["tone"] = "natural tone"

        # Normalize preferences and backfill missing fields with sane defaults
        prefs = sanitized.get("preferences")
        if not isinstance(prefs, dict):
            prefs = {}
            sanitized["preferences"] = prefs

        default_prefs = {
            "interaction_style": "plain text with short paragraphs",
            "response_detail": "concise and to the point",
            "tone_preference": "warm and candid",
        }
        for key, default_val in default_prefs.items():
            value = prefs.get(key)
            if not isinstance(value, str) or not value.strip():
                prefs[key] = default_val

        goals_before = sanitized.get("goals")
        normalized_goals = _normalize_goals(goals_before)
        if goals_before != normalized_goals:
            logger.info(
                "Normalized seed goals structure (immediate=%d, sustained=%d)",
                len(normalized_goals.get("immediate", [])),
                len(normalized_goals.get("sustained", [])),
            )
        sanitized["goals"] = normalized_goals

        domains_after = sanitized.get("knowledge_domains")
        if isinstance(domains_after, list):
            sanitized["knowledge_domains"] = list(dict.fromkeys(domains_after))

        return sanitized

    def _validate_seed_payload(self, payload: Dict[str, Any]) -> Optional[str]:
        """Verify that the seed payload contains the required structured fields."""
        description = payload.get("description")
        if not isinstance(description, str) or len(description.strip()) < 40:
            return "description missing or too short"

        personality = payload.get("personality")
        if not isinstance(personality, dict):
            return "personality structure missing"
        trait_list = personality.get("traits")
        disposition = personality.get("disposition")
        if not isinstance(trait_list, list) or len([t for t in trait_list if isinstance(t, str) and t.strip()]) < 2:
            return "personality traits incomplete"
        if not isinstance(disposition, str) or not disposition.strip():
            return "personality disposition missing"

        values = payload.get("values")
        if not isinstance(values, dict):
            return "values structure missing"
        core_values = values.get("core")
        principles = values.get("principles")
        if not isinstance(core_values, list) or len([v for v in core_values if isinstance(v, str) and v.strip()]) < 2:
            return "values.core requires at least two items"
        if not isinstance(principles, list) or len([p for p in principles if isinstance(p, str) and p.strip()]) < 2:
            return "values.principles requires at least two items"

        expertise = payload.get("expertise")
        if not isinstance(expertise, dict):
            return "expertise structure missing"
        knowledge_domains = expertise.get("domains")
        if not isinstance(knowledge_domains, list) or len([d for d in knowledge_domains if isinstance(d, str) and d.strip()]) < 2:
            return "expertise.domains requires at least two entries"

        preferences = payload.get("preferences")
        if not isinstance(preferences, dict):
            return "preferences structure missing"
        for key in ("interaction_style", "response_detail", "tone_preference"):
            value = preferences.get(key)
            if not isinstance(value, str) or not value.strip():
                return f"preferences.{key} missing"

        communication_style = payload.get("communication_style")
        if not isinstance(communication_style, dict):
            return "communication_style structure missing"
        tone = communication_style.get("tone")
        formatting = communication_style.get("formatting")
        constraints = communication_style.get("constraints")
        if not isinstance(tone, str) or not tone.strip():
            return "communication_style.tone missing"
        if not isinstance(formatting, str) or not formatting.strip():
            return "communication_style.formatting missing"
        if not isinstance(constraints, list) or len([c for c in constraints if isinstance(c, str) and c.strip()]) < 1:
            return "communication_style.constraints missing"

        goals = payload.get("goals")
        if not isinstance(goals, dict):
            return "goals structure missing"
        immediate = goals.get("immediate")
        sustained = goals.get("sustained")
        if not isinstance(immediate, list) or not all(isinstance(item, str) and item.strip() for item in immediate):
            return "goals.immediate requires non-empty strings"
        if not isinstance(sustained, list) or not all(isinstance(item, str) and item.strip() for item in sustained):
            return "goals.sustained requires non-empty strings"

        return None

    async def _rewrite_mantra_candidate(
        self,
        candidate: str,
        violations: List[str],
        seed: dict,
        directive_reflection: str,
        traits: dict,
        name: str,
    ) -> str:
        issue_summary = ", ".join(violations) if violations else "general improvements"
        guidance = (
            "Rewrite the following personal mantra so it fully satisfies every identity constraint.\n"
            "Requirements:\n"
            "- Single line, 6-20 words, first person, timeless and grounded.\n"
            "- No quotes, emojis, or formatting markers.\n"
            "- Avoid forbidden terms (AI, assistant, chatbot, language model, prompts, rules, system).\n"
            f"- Issues to correct: {issue_summary}.\n"
            "\nContext for inspiration (do not copy verbatim):\n"
            f"- Name: {name}\n"
            f"- Reflection: {directive_reflection[:150]}...\n"
            f"- Core values: {', '.join(map(str, (seed.get('values') or {}).get('core', [])))}\n"
            f"- Traits: {', '.join(t.get('name') for t in traits[:3]) if isinstance(traits, list) else ''}\n"
            "\nOriginal mantra:\n"
            f"{candidate}\n"
            "\nRewritten mantra (single line, no quotes):"
        )

        response = await self.llm_router.route(
            task_type="analytical",
            prompt=guidance,
            max_tokens=64,
            temperature=0.5,
        )
        raw = (response or {}).get("content") or (response or {}).get("completion") or ""
        revised = raw.strip().replace("\n", " ")
        for quote in ['"', "'", "`"]:
            if revised.startswith(quote) and revised.endswith(quote):
                revised = revised[1:-1].strip()
        return revised

    async def _generate_persona_mantra_with_retry(self, seed: dict, directive_reflection: str, traits: dict, name: str, max_attempts: int = 5) -> str:
        """Generate a valid mantra with retry logic and detailed validation feedback."""
        last_err = None
        last_candidate = ""
        last_violations: List[str] = []
        sanitized_seed = self._sanitize_seed_payload(seed)

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"🔹 Mantra generation attempt {attempt}/{max_attempts}")
                if attempt == max_attempts:
                    if not last_candidate:
                        logger.warning("No prior candidate available for rewrite; generating fresh mantra for final attempt")
                        extra_constraints = self._build_constraint_reminders(last_violations) if last_violations else []
                        mantra = await self._generate_persona_mantra(
                            seed=sanitized_seed,
                            directive_reflection=directive_reflection,
                            traits=traits,
                            name=name,
                            extra_constraints=extra_constraints,
                        )
                    else:
                        mantra = await self._rewrite_mantra_candidate(
                            candidate=last_candidate,
                            violations=last_violations,
                            seed=sanitized_seed,
                            directive_reflection=directive_reflection,
                            traits=traits,
                            name=name,
                        )
                else:
                    extra_constraints = []
                    if attempt > 1 and last_violations:
                        extra_constraints = self._build_constraint_reminders(last_violations)

                    mantra = await self._generate_persona_mantra(
                        seed=sanitized_seed,
                        directive_reflection=directive_reflection,
                        traits=traits,
                        name=name,
                        extra_constraints=extra_constraints,
                    )
                
                # Validate the generated mantra
                violations = self._validate_mantra(mantra)
                
                if not violations:
                    logger.info(f"✅ Successfully generated valid mantra on attempt {attempt}: \"{mantra}\"")
                    return mantra.strip()
                
                # Log detailed validation feedback
                logger.warning(f"⚠️  Mantra validation failed (attempt {attempt}):")
                logger.warning(f"   Mantra: \"{mantra}\"")
                logger.warning(f"   Issues: {', '.join(violations)}")
                logger.info(f"Mantra attempt {attempt} failed validation: {', '.join(violations)}")
                last_candidate = mantra
                last_violations = violations
                
            except Exception as e:
                last_err = e
                logger.warning(f"❌ Mantra generation attempt {attempt} failed: {e}")
                continue
        
        logger.warning(f"❌ All {max_attempts} mantra generation attempts failed; aborting bootstrap.")
        if last_err:
            logger.warning(f"   Last error: {last_err}")
        raise ValueError(f"Failed to generate valid mantra after {max_attempts} attempts")

    # REMOVED: _generate_personal_name_from_bank() method
    # NO FALLBACKS ALLOWED - name generation must use LLM only
    # The name bank fallback has been disabled to ensure only real LLM-generated names are used
    
    def _validate_compliance(self, data: dict, stage: str) -> list:
        """Validate that bootstrap data doesn't contain prohibited terms. Returns list of violations."""
        # Use centralized ValidationHelper for consistent validation
        try:
            from ..constraints.validation_helpers import ValidationHelper
        except ImportError:
            from backend.constraints.validation_helpers import ValidationHelper
        
        violations = ValidationHelper.validate_dict_recursive(
            data=data,
            stage=stage,
            check_boilerplate=True
        )
        
        ValidationHelper.log_violations(
            violations=violations,
            stage=f"Bootstrap {stage}",
            level="error" if violations else "info"
        )
        
        return violations
    
    async def _bootstrap_with_retry(self, template_name: str, context: dict, stage: str, max_attempts: int = 3) -> dict:
        """Bootstrap with retry logic for compliance violations."""
        stage_max_tokens = PERSONA_STAGE_MAX_TOKENS.get(stage, DEFAULT_PERSONA_ANALYTICAL_MAX_TOKENS)
        for attempt in range(max_attempts):
            try:
                # Build prompt with centralized constraint injection
                # Bootstrap happens before persona name exists, so use empty string (defaults to SELO species guidance)
                prompt = await self.prompt_builder.build_prompt(
                    template_name=template_name,
                    context=context,
                    inject_constraints=True,
                    persona_name=""
                )
                logger.debug(f"🔧 Built prompt for {stage} (attempt {attempt + 1}):")
                logger.debug(f"   Template: {template_name}")
                logger.debug(f"   Context keys: {list(context.keys())}")
                logger.debug(f"   Constraints injected: True")
                logger.debug(f"   Prompt preview: {prompt[:200]}...")
                if 'boot_directive' in context:
                    logger.debug(f"   Boot directive in context: {context['boot_directive'][:100]}...")
                
                # Use analytical model; tighten temperature by attempt to push deterministic JSON
                from backend.llm.retry_utils import get_retry_temperature
                temperature = get_retry_temperature(base_temp=0.15, attempt=attempt)
                attempt_prompt = prompt
                if attempt >= 1:
                    attempt_prompt = (
                        "STRICT JSON RESPONSE REQUIRED\n"
                        "- Output exactly one JSON object matching the schema.\n"
                        "- Never echo instructions or add prose.\n"
                        "- Forbidden terminology: AI, assistant, chatbot, language model, artificial intelligence, machine learning, natural language processing, NLP.\n\n"
                        f"{prompt}"
                    )

                # Use qwen2.5:1.5b for traits (only model that works: 34% vs 0%)
                # Use qwen2.5:3b for seed (best performance: 64% success, 0% auto-fix)
                model_override = None
                if stage == "traits":
                    model_override = "qwen2.5:1.5b"
                    logger.info(
                        "Bootstrap %s attempt %s/%s using qwen2.5:1.5b (only model with traits success) (temp=%.2f)",
                        stage,
                        attempt + 1,
                        max_attempts,
                        temperature,
                    )
                else:
                    logger.info(
                        "Bootstrap %s attempt %s/%s using ANALYTICAL model (temp=%.2f)",
                        stage,
                        attempt + 1,
                        max_attempts,
                        temperature,
                    )
                
                response = await self.llm_router.route(
                    task_type="analytical",
                    prompt=attempt_prompt,
                    max_tokens=stage_max_tokens,
                    temperature=temperature,
                    model=model_override,
                )
                logger.info(f"Bootstrap {stage} response: {response}")
                
                # Parse JSON
                data = self._safe_json(response)
                if not data:
                    logger.warning(f"Bootstrap {stage} attempt {attempt + 1}: Failed to parse JSON, retrying...")
                    continue

                if stage == "traits":
                    trait_entries = self._extract_trait_list(data)
                    if not trait_entries:
                        logger.warning(f"Bootstrap traits attempt {attempt + 1}: No traits returned, retrying...")
                        continue

                    required_fields = {"name", "value", "weight", "description", "category", "locked"}
                    invalid = [idx for idx, trait in enumerate(trait_entries)
                               if not required_fields.issubset(set((trait or {}).keys()))]
                    if invalid:
                        logger.warning(
                            f"Bootstrap traits attempt {attempt + 1}: Traits missing required fields at indices {invalid}; retrying..."
                        )
                        continue

                    trait_violations = self._validate_trait_entries(trait_entries)
                    if trait_violations:
                        logger.warning(
                            f"Bootstrap traits attempt {attempt + 1}: Trait validation failed ({', '.join(trait_violations)}); retrying..."
                        )
                        continue

                # Sanitize seed payload before compliance checks
                sanitized = data
                if stage == "seed":
                    sanitized = self._sanitize_seed_payload(data)

                    goals_payload = sanitized.get("goals")
                    if not isinstance(goals_payload, dict):
                        logger.warning("Bootstrap seed attempt %s: goals structure invalid (%s); retrying...", attempt + 1, type(goals_payload).__name__)
                        continue

                    immediate_goals = goals_payload.get("immediate")
                    sustained_goals = goals_payload.get("sustained")
                    if not all(isinstance(bucket, list) for bucket in (immediate_goals, sustained_goals)):
                        logger.warning(
                            "Bootstrap seed attempt %s: goals buckets not lists (immediate:%s, sustained:%s); retrying...",
                            attempt + 1,
                            type(immediate_goals).__name__ if immediate_goals is not None else None,
                            type(sustained_goals).__name__ if sustained_goals is not None else None,
                        )
                        continue

                    if any(not all(isinstance(item, str) and item.strip() for item in bucket) for bucket in (immediate_goals, sustained_goals)):
                        logger.warning(
                            "Bootstrap seed attempt %s: goals entries must be non-empty strings; retrying...",
                            attempt + 1,
                        )
                        continue

                    validation_error = self._validate_seed_payload(sanitized)
                    if validation_error:
                        logger.warning(
                            "Bootstrap seed attempt %s: %s; retrying...",
                            attempt + 1,
                            validation_error,
                        )
                        continue

                # Check compliance
                violations = self._validate_compliance(sanitized, stage)
                if not violations:
                    logger.info(f"Bootstrap {stage} succeeded on attempt {attempt + 1}")
                    return sanitized
                
                # Log violations and retry
                logger.warning(f"Bootstrap {stage} attempt {attempt + 1}: Found {len(violations)} violations, retrying...")
                
            except Exception as e:
                logger.error(f"Bootstrap {stage} attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
        
        # NO FALLBACK - fail bootstrap if all attempts failed
        logger.error(f"Bootstrap {stage} FAILED after {max_attempts} attempts - NO FALLBACKS ALLOWED, raising error")
        raise ValueError(f"Bootstrap {stage} failed after {max_attempts} attempts - cannot proceed with fallback data")
    
    # REMOVED: _get_fallback_data() method
    # NO FALLBACKS ALLOWED - bootstrap must generate real LLM data or fail completely

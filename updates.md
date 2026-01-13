## Phase 1: Content & Prompt Improvements (Low Risk, High Impact)

### 1. Consolidate Repetitive Boot Directives
**Current:** 365 directives, mostly identical content with different titles

**Modification:**
- Reduce to 30-40 truly distinct directive archetypes with meaningful philosophical variation
- Each directive should offer different framing: existential, relational, exploratory, introspective, scientific, poetic, etc.
- Ensure real diversity in approach and tone, not just titles

**Why:** Since this is the ONE moment of emergence per installation, the directive should actually matter. Real variety in philosophical framing gives each installation genuine uniqueness.

**Files to modify:**
- `Reports/Boot_Seed_Directive_Prompts.md`

**Estimated Effort:** Medium (requires creative writing and philosophical curation)

---

### 2. Compress Bootstrap Templates (Reduce Token Usage ~35-40%)
**Current:** Verbose templates with repeated warnings, examples, checklists

**Modification:**
- Single critical requirements block (not scattered throughout)
- JSON schema with inline comments only where essential
- One validation checklist at end (not repeated)
- Remove duplicate warnings
- Consolidate examples into 2-3 clear patterns
- Remove redundant CRITICAL/⚠️ formatting

**Example transformation:**
- Before (73 lines): `persona_bootstrap_seed.txt`
- After (~45 lines): streamlined version with same requirements

**Why:** Cleaner prompts = more reliable generation, less LLM confusion from repetition, stays within context limits

**Files to modify:**
- `backend/prompt/templates/persona_bootstrap_seed.txt`
- `backend/prompt/templates/persona_bootstrap_traits.txt`
- `backend/prompt/templates/reflection_message.txt`

**Estimated Effort:** Small (text editing, preserving requirements)

---

### 3. Optimize Directive Reflection Prompt
**Current:** Complex prompt with many repeated forbidden patterns

**Modification:**
```python
# Streamlined, clearer prompt
reflection_prompt = f"""Generate SELO's first-person emotional reaction to receiving this directive:

DIRECTIVE:
"{directive}"

REQUIREMENTS:
- Write 3-4 complete sentences in English
- First-person perspective (I/me/my)
- Express genuine feeling about this directive and meeting the user
- Complete thoughts (no trailing "...")

FORBIDDEN:
- "you are" or "your" (instructional tone)
- Third-person ("The SELO...", "[name]'s...")
- Repeating the directive verbatim
- Commands or instructions

Write your personal reaction:"""
```

**Why:** Simpler, clearer prompt = more reliable generation. Current version over-explains with defensive warnings.

**Files to modify:**
- `backend/persona/bootstrapper.py` (`_generate_directive_reflection` method)

**Estimated Effort:** Small (single method update)

---

## Phase 2: Validation Adjustments (Medium Risk, High Impact)

### 4. Adjust Validation Limits to Reduce Arbitrary Failures
**Current Rigid Limits:**
- Trait description: exactly 18 words (hard fail at 19)
- Trait name: exactly 3-16 chars (rejects valid 17-char words)
- Reflection content: exactly 70-200 words

**Modification:**
```python
# Recommended vs Absolute limits
TRAIT_DESC_RECOMMENDED = 18
TRAIT_DESC_ABSOLUTE_MAX = 22  # Allow reasonable overflow

TRAIT_NAME_RECOMMENDED_MAX = 16
TRAIT_NAME_ABSOLUTE_MAX = 18  # Allows valid longer words

REFLECTION_RECOMMENDED = (120, 180)
REFLECTION_ABSOLUTE = (70, 250)  # Wider acceptable range

def validate_with_tolerance(value: int, recommended: int, absolute_max: int, 
                           field_name: str) -> Tuple[bool, str]:
    """Validate with recommended vs absolute limits."""
    if value > absolute_max:
        return False, f"{field_name} exceeds absolute max: {value} > {absolute_max}"
    
    if value > recommended:
        logger.info(f"{field_name} exceeds recommended ({value} > {recommended}) but within tolerance")
    
    return True, "Valid"
```

**Why:** Failing bootstrap on 19 vs 18 words is arbitrary. Reasonable tolerance reduces retry waste while maintaining quality.

**Files to modify:**
- `backend/persona/bootstrapper.py` (validation constants)
- `backend/constraints/identity_constraints.py` (trait validation)

**Estimated Effort:** Small (constant updates and tolerance logic)

---

### 5. Implement Tiered Pattern Matching for Validation
**Current:** 100+ regex patterns checked linearly every validation

**Modification:**
```python
class ValidationTiers:
    """Tiered validation for performance."""
    
    # Tier 1: Critical failures (5-10 patterns) - always check first
    CRITICAL_FAILURES = [
        r'\b(ai|artificial intelligence)\b',
        r'\bchatbot\b',
        r'\blanguage model\b',
        r'\bprogrammed to\b',
    ]
    
    # Tier 2: Important violations (20-30 patterns) - check if Tier 1 passes
    IMPORTANT_VIOLATIONS = [
        r'the system told me',
        r'craft a response',
        r'here to serve',
        # ... rest of common patterns
    ]
    
    # Tier 3: Edge cases (remaining patterns) - check in strict mode only
    EDGE_CASES = [
        # Less common meta-reasoning patterns
        # Contextual servile language
    ]

    @classmethod
    def validate_with_tiers(cls, text: str, strict: bool = True) -> Tuple[bool, List[str]]:
        """Fast-fail tiered validation."""
        violations = []
        
        # Tier 1: Critical patterns (fast fail)
        for pattern in cls.CRITICAL_FAILURES:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"CRITICAL: {pattern}")
                return False, violations  # Stop immediately
        
        # Tier 2: Important patterns
        for pattern in cls.IMPORTANT_VIOLATIONS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                violations.append(f"Important: {match.group(0)}")
        
        # Tier 3: Edge cases (only in strict mode)
        if strict:
            for pattern in cls.EDGE_CASES:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    violations.append(f"Edge case: {match.group(0)}")
        
        return len(violations) == 0, violations
```

**Why:** 50-70% faster validation. Most failures are obvious and caught by Tier 1, no need to check 90+ remaining patterns.

**Files to modify:**
- `backend/constraints/identity_constraints.py`
- `backend/constraints/validation_helpers.py` (if exists)

**Estimated Effort:** Medium (reorganize patterns, test tier effectiveness)

---

## Phase 3: Architecture Improvements (Higher Risk, High Value)

### 6. Merge Overlapping Constraint Classes into Unified System
**Current:** 5 separate constraint classes with duplicated concerns

**Modification:**
```python
# backend/constraints/unified_constraints.py
class ConstraintSystem:
    """Unified constraint system - single source of truth."""
    
    # Core definitions (no duplication)
    FORBIDDEN_TERMS = {...}  # From IdentityConstraints
    GROUNDING_RULES = "..."  # From CoreConstraints + GroundingConstraints
    IDENTITY_RULES = "..."   # From IdentityConstraints + CoreConstraints
    ETHICAL_RULES = "..."    # From EthicalGuardrails
    BEHAVIORAL_RULES = "..." # From BehavioralGuidelines
    
    @classmethod
    def for_bootstrap(cls, persona_name: str = "") -> str:
        """Comprehensive constraints for one-time bootstrap."""
        return cls._compose([
            cls.IDENTITY_RULES,
            cls.GROUNDING_RULES,
            cls.NO_FABRICATION,
            cls.OUTPUT_FORMAT_STRICT
        ])
    
    @classmethod
    def for_reflection(cls, persona_name: str) -> str:
        """Constraints for ongoing reflection generation."""
        return cls._compose([
            cls.get_persona_name_constraint(persona_name),
            cls.GROUNDING_RULES,
            cls.REFLECTION_SPECIFIC
        ])
    
    @classmethod
    def for_conversation(cls, persona_name: str) -> str:
        """Compact constraints for frequent conversation."""
        return cls._compose([
            cls.get_persona_name_constraint(persona_name),
            cls.GROUNDING_RULES_COMPACT,
            cls.TRUTHFULNESS_COMPACT
        ])
```

**Why:** Eliminates code duplication, ensures consistency, context-appropriate constraint delivery

**Files to create:**
- `backend/constraints/unified_constraints.py`

**Files to modify:**
- `backend/constraints/__init__.py`
- `backend/prompt/builder.py` (use unified system)

**Estimated Effort:** Large (refactor all constraint usage, extensive testing)

---

### 7. Add Constraint Composition Optimizer
**Current:** Constraints concatenated as strings, no optimization

**Modification:**
```python
class ConstraintComposer:
    """Intelligently compose constraints to minimize tokens."""
    
    def compose_for_context(self, context_type: str, persona_name: str,
                           token_budget: Optional[int] = None) -> str:
        """Compose constraints optimally for context."""
        
        # Get base constraints for context
        if context_type == "bootstrap":
            constraints = self._get_bootstrap_constraints(persona_name)
        elif context_type == "reflection":
            constraints = self._get_reflection_constraints(persona_name)
        elif context_type == "conversation":
            constraints = self._get_conversation_constraints(persona_name)
        
        # If token budget specified, optimize
        if token_budget:
            constraints = self._optimize_for_budget(constraints, token_budget)
        
        return constraints
    
    def _optimize_for_budget(self, constraints: str, budget: int) -> str:
        """Compress constraints to fit budget while preserving critical rules."""
        # Token count
        current_tokens = self._estimate_tokens(constraints)
        
        if current_tokens <= budget:
            return constraints
        
        # Progressive compression:
        # 1. Remove examples
        # 2. Shorten explanations
        # 3. Use compact format
        # 4. Keep only critical constraints if still over budget
        
        return self._apply_compression_strategy(constraints, budget)
```

**Why:** Different contexts need different constraint depths. Optimize token usage while maintaining enforcement.

**Files to create:**
- `backend/constraints/composer.py`

**Files to modify:**
- `backend/prompt/builder.py` (use composer)

**Estimated Effort:** Medium (depends on unified constraints being done first)

---

## Phase 4: Performance Optimizations (Independent, Lower Priority)

### 8. Optimize Few-Shot Example Selection
**Current:** Retrieves examples but with name neutralization overhead

**Modification:**
```python
class OptimizedExampleSelector:
    """Efficient example selection with caching."""
    
    def __init__(self):
        self._neutralized_cache = {}  # Cache neutralized examples
    
    async def get_examples_for_context(self, context: Dict, 
                                      num_positive: int = 3,
                                      num_negative: int = 2) -> List[Dict]:
        """Get context-aware examples with caching."""
        
        # Get raw examples
        examples = await self.example_repo.get_by_context(context)
        
        # Apply neutralization with caching
        neutralized = []
        for ex in examples:
            ex_id = ex.get("id")
            if ex_id in self._neutralized_cache:
                neutralized.append(self._neutralized_cache[ex_id])
            else:
                neutralized_ex = self._neutralize_names(ex)
                self._neutralized_cache[ex_id] = neutralized_ex
                neutralized.append(neutralized_ex)
        
        return neutralized[:num_positive + num_negative]
```

**Why:** Reduces overhead of name neutralization on every reflection generation (happens frequently post-bootstrap).

**Files to modify:**
- `backend/prompt/builder.py` (`_get_few_shot_examples` method)

**Estimated Effort:** Small (add caching layer)

---

## Phase 5: Telemetry & Tracking (Optional, Can Be Added Anytime)

### 9. Implement Constraint Version Tracking
**Current:** Constraint classes have VERSION strings but not used

**Modification:**
```python
class ConstraintVersionManager:
    """Track constraint versions and their effectiveness."""
    
    async def record_constraint_version(self, generation_id: str,
                                       constraint_version: str,
                                       context_type: str):
        """Associate generation with constraint version used."""
        await self.version_repo.save({
            "generation_id": generation_id,
            "constraint_version": constraint_version,
            "context_type": context_type,
            "timestamp": datetime.now()
        })
    
    async def compare_version_effectiveness(self, v1: str, v2: str) -> Dict:
        """Compare effectiveness of two constraint versions."""
        # Query violations, success rates, retry counts per version
        # Generate comparison report
        pass
```

**Why:** A/B test constraint changes. Understand impact of constraint modifications on success rates and quality.

**Files to create:**
- `backend/constraints/version_manager.py`

**Files to modify:**
- `backend/persona/bootstrapper.py` (record versions used)

**Estimated Effort:** Medium (requires database schema additions)

---

## Summary of Expected Impact

**Reliability:**
- Reduced arbitrary failures (validation tolerance)
- Faster failure detection (tiered pattern matching)
- Better prompts (clearer, less confusing)

**Maintainability:**
- Single constraint source of truth (unified system)
- Context-appropriate constraint delivery
- Version tracking enables A/B testing

**Performance:**
- 35-40% token reduction (template compression)
- 50-70% faster validation (tiered matching)
- Cached example neutralization

**Data-Driven Optimization:**
- Version tracking for constraint effectiveness
- Ability to measure impact of changes
- Foundation for continuous improvement

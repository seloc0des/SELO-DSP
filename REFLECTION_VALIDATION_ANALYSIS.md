# Reflection Validation System Analysis

**Date**: December 26, 2025  
**Issue**: Persistent reflection validation failures causing HTTP 503 errors  
**Error Pattern**: `content_length_out_of_bounds` - LLM consistently generates 56-78 word reflections vs required 80-180 words

---

## Executive Summary

The reflection system has **multiple conflicting word count configurations** across different layers, creating a mismatch between what the LLM is instructed to generate and what the validation layer accepts. The installer sets different values than the code defaults, and the prompt templates contain yet another set of ranges.

### Critical Finding
**The LLM is being told to generate 80-250 words in prompts, but validation expects 80-180 words, and the installer writes 90-180 to .env files.**

---

## 1. Configuration Inconsistencies

### 1.1 Environment Variable Conflicts

**Installer Script** (`install-complete.sh:227-228`):
```bash
export TIER_REFLECTION_WORD_MAX=180
export TIER_REFLECTION_WORD_MIN=90
```

**Written to .env** (`install-complete.sh:927, 931`):
```bash
REFLECTION_WORD_MIN=${TIER_REFLECTION_WORD_MIN:-100}  # Fallback: 100
REFLECTION_WORD_MAX=${TIER_REFLECTION_WORD_MAX:-150}  # Fallback: 150
```

**Code Default** (`config/reflection_config.py:84-85`):
```python
self.word_count_min = int(os.getenv("REFLECTION_WORD_MIN", "90"))
self.word_count_max = int(os.getenv("REFLECTION_WORD_MAX", "180"))  # Tier-aware fallback
```

**Processor Fallback** (`reflection/processor.py:138-144`):
```python
cls._word_count_config = {
    'min': getattr(cfg, 'word_count_min', 80),  # Fallback: 80
    'max': getattr(cfg, 'word_count_max', 180)  # Fallback: 180
}
```

**Documentation** (`.env.example:107-108`):
```bash
REFLECTION_WORD_MIN=80
REFLECTION_WORD_MAX=180
```

### 1.2 Prompt Template Conflicts

**Template File** (`prompt/templates/reflection_message.txt:13, 30, 46, 100, 105, 117`):
- Line 13: `// 80-250 words, 2-3 paragraphs`
- Line 30: `"Your reflection here (80-250 words)"`
- Line 46: `Content must be 80-250 words total`
- Line 100: `Write 2-3 paragraphs of genuine reflection (80-250 words)`
- Line 105: `✓ Content is 80-250 words`
- Line 117: `✓ Content is 80-250 words`

**Dynamic Prompt Injection** (`reflection/processor.py:1711-1715`):
```python
f"\n⚠️ WORD COUNT: Your 'content' field MUST be {config_word_min}-{config_word_max} words (NOT negotiable)\n"
f"   - Write 2-3 FULL paragraphs (each paragraph = 3-5 complete sentences)\n"
f"   - Each sentence should have emotional depth and specific observations\n"
f"   - Vary your sentence structures: mix short and long, use dashes for pauses, blend statements with questions\n"
f"   - Aim for ~15 words per sentence on average (6-12 sentences total to reach {config_word_min}-{config_word_max} words)\n"
```

**Problem**: The template says 80-250, but the dynamic injection uses config values (80-180 or 90-180 depending on .env).

### 1.3 Message-Type Override Range

**Environment Variable** (`.env.example:113-114`):
```bash
REFLECTION_MESSAGE_WORD_MIN=80
REFLECTION_MESSAGE_WORD_MAX=320
```

**Code Implementation** (`reflection/processor.py:186-200`):
```python
if rtype in {"message", "memory_triggered"}:
    msg_min = int(os.getenv("REFLECTION_MESSAGE_WORD_MIN", "80"))
    msg_max = int(os.getenv("REFLECTION_MESSAGE_WORD_MAX", "320"))
    return {"min": msg_min, "max": msg_max}
```

**Problem**: Message reflections can be 80-320 words, but the LLM is still being instructed with the base template (80-250).

---

## 2. Validation Logic Issues

### 2.1 Schema Validation Function

**Location**: `reflection/processor.py:535-560`

```python
def _reflection_meets_schema(self, reflection_obj: dict, reflection_type: Optional[str] = None) -> bool:
    # ... validation checks ...
    word_count = count_words(content_val)
    word_config = self.get_type_word_bounds(reflection_type)
    min_words = word_config['min']
    max_words = word_config['max']
    if word_count < min_words or word_count > max_words:
        logger.debug(
            "Reflection content length outside %s-%s range during schema check: %s words",
            min_words, max_words, word_count,
        )
        return False  # HARD FAILURE
```

**Issue**: This function returns `False` for ANY word count violation, triggering the retry loop.

### 2.2 Post-Check Validation

**Location**: `reflection/processor.py:2234-2247`

```python
word_count = count_words(text)
if word_count < config_word_min or word_count > config_word_max:
    logger.warning(
        f"Length: {word_count} words (expected {config_word_min}-{config_word_max}). "
        "Treating as soft length guard (content_length_out_of_bounds); reflection will still be used."
    )
    return {
        "compliant": False,
        "violations": ["content_length_out_of_bounds"],
        "soft": True,  # ← Marked as "soft" but still triggers retry
        "word_count": word_count,
        "expected_min": config_word_min,
        "expected_max": config_word_max,
    }
```

**Issue**: Despite being marked as `"soft": True`, this violation still triggers the retry loop at line 2365.

### 2.3 Retry Exhaustion Logic

**Location**: `reflection/processor.py:2544-2584`

```python
# If all retries failed, propagate the failure - DO NOT store non-compliant reflection
if not (post_check or {}).get("compliant", True):
    violations = set((post_check or {}).get("violations", []) or [])
    # REMOVED: No longer accept length violations as "soft" errors
    # All violations are now hard failures that reject the reflection
    word_count = (post_check or {}).get('word_count', 0)
    expected_min = (post_check or {}).get('expected_min', config_word_min)
    expected_max = (post_check or {}).get('expected_max', config_word_max)
    
    # Log detailed error with word count if available
    if word_count > 0:
        logger.error(
            f"🚫 CRITICAL: Reflection validation failed after {retry_count} attempts. "
            f"Violations: {list(violations)}. "
            f"Word count: {word_count}, expected: {expected_min}-{expected_max}. "
            "REJECTING reflection."
        )
```

**Issue**: Comment says "REMOVED: No longer accept length violations as 'soft' errors" but the code at line 2243 still marks them as soft. This is a logic inconsistency.

### 2.4 Retry Configuration

**Max Retries**: `REFLECTION_IDENTITY_MAX_RETRIES=4` (default)  
**Retry Budget**: `REFLECTION_IDENTITY_TIMEOUT_S=120` (default)

**Location**: `reflection/processor.py:2126-2129`

---

## 3. Few-Shot Example Issues

### 3.1 Example Word Counts

**Positive Example** (`db/repositories/example.py:127`):
- **Actual**: 175 words
- **Expected**: 80-180 words ✅ (within range)

**Negative Example** (`db/repositories/example.py:141`):
- **Actual**: 58 words
- **Expected**: Should demonstrate violation ✅ (correctly shows what NOT to do)

**Problem**: The positive example is at the HIGH end (175 words), which may bias the LLM toward longer outputs. However, the LLM is actually generating SHORT outputs (56-78 words), suggesting the example isn't being followed.

### 3.2 Example Selection

**Location**: `db/repositories/example.py:156-216`

The system uses **1 positive + 1 negative** example for all reflections. This minimal approach may not provide enough guidance for the LLM.

---

## 4. Root Cause Analysis

### 4.1 Why LLM Generates Short Reflections

1. **Model Capability**: `qwen2.5:3b` is a small model that may struggle with consistent long-form generation
2. **Temperature**: `0.35` is relatively low, reducing creativity/verbosity
3. **Token Budget**: `REFLECTION_NUM_PREDICT=0` (unbounded) and `REFLECTION_MAX_TOKENS=0` (unbounded) means no hard limit, but the model may naturally stop early
4. **Prompt Confusion**: Multiple conflicting word count instructions (80-250 in template, 80-180 in dynamic injection)
5. **Example Bias**: Only 1 positive example, and it's at 175 words (high end), but LLM still generates short

### 4.2 Why Validation Fails

1. **Strict Enforcement**: Word count violations are treated as HARD failures despite "soft" flag
2. **No Tolerance Band**: Even 78 words (2 words short of 80) triggers full rejection
3. **Retry Loop**: System retries up to 4 times, but each retry uses the same flawed prompt
4. **Fallback Disabled**: After retries fail, system raises RuntimeError instead of using fallback

---

## 5. Installer vs Runtime Mismatch

### 5.1 Installer Sets Wrong Values

**Tier Detection** (`detect-tier.sh:57-58`):
```bash
export TIER_REFLECTION_WORD_MAX=180
export TIER_REFLECTION_WORD_MIN=90
```

**But Installer Writes** (`install-complete.sh:927, 931`):
```bash
REFLECTION_WORD_MIN=${TIER_REFLECTION_WORD_MIN:-100}  # Uses 90 if set, else 100
REFLECTION_WORD_MAX=${TIER_REFLECTION_WORD_MAX:-150}  # Uses 180 if set, else 150
```

**Problem**: The fallback values (100-150) don't match the tier values (90-180) OR the code defaults (80-180).

### 5.2 No .env File Present

**Finding**: Running `cat .env` in backend directory shows **"No .env file found"**

This means:
- System is using **code defaults** (80-180) from `reflection_config.py`
- NOT using installer values
- NOT using .env.example values

---

## 6. Specific Error Pattern from Logs

### 6.1 Observed Failures

```
Word count: 78, expected: 80-320  # 2 words short
Word count: 77, expected: 90-180  # Different range!
Word count: 56, expected: 80-320  # Significantly short
```

**Analysis**: The "expected" ranges vary between logs, indicating:
- Sometimes using message-type override (80-320)
- Sometimes using base config (90-180)
- Inconsistent application of word bounds

### 6.2 Retry Behavior

From logs:
```
"🔄 Reflection violated constraints (attempt 1/4). Violations: ['content_length_out_of_bounds']"
"Retry 1 failed with error: Reflection failed schema validation: content length 77 words outside 90-180 range."
"🚫 CRITICAL: Reflection validation failed after 1 attempts"
```

**Problem**: Only 1 retry attempted despite `max_retries=4`. This suggests the retry loop is exiting early.

---

## 7. Errors and Inconsistencies Summary

### Critical Errors

1. **❌ No .env file**: System running on code defaults, not installer configuration
2. **❌ Installer fallback mismatch**: Writes 100-150 when tier sets 90-180
3. **❌ Template vs validation mismatch**: Template says 80-250, validation checks 80-180
4. **❌ "Soft" violations treated as hard**: Code marks length as `"soft": True` but still rejects
5. **❌ Retry loop exits early**: Only 1 retry instead of 4 configured retries
6. **❌ Conflicting word count sources**: At least 5 different ranges across codebase

### Inconsistencies

1. **⚠️ Multiple word count ranges**:
   - Template: 80-250
   - Code default: 80-180
   - Installer tier: 90-180
   - Installer fallback: 100-150
   - Message override: 80-320
   - .env.example: 80-180

2. **⚠️ Example word count**: Positive example is 175 words (near max), but LLM generates 56-78 words (near min or below)

3. **⚠️ Comment vs code**: Line 2547 says "REMOVED: No longer accept length violations as 'soft' errors" but line 2243 still sets `"soft": True`

4. **⚠️ Validation happens twice**:
   - `_reflection_meets_schema()` at line 535 (hard fail)
   - `_run_post_checks()` at line 2234 (soft fail that becomes hard)

---

## 8. Recommended Corrections

### 8.1 Immediate Fixes (High Priority)

#### Fix 1: Create .env File with Consistent Values
**Action**: Create `/mnt/local/Projects/SELODSP/selo-ai/backend/.env` with:
```bash
REFLECTION_WORD_MIN=80
REFLECTION_WORD_MAX=180
REFLECTION_MESSAGE_WORD_MIN=80
REFLECTION_MESSAGE_WORD_MAX=180  # Reduce from 320 to match base
```

**Rationale**: 
- Eliminates reliance on code defaults
- Uses consistent range across all reflection types
- Matches what qwen2.5:3b can reliably generate

#### Fix 2: Update Prompt Template to Match Validation
**File**: `backend/prompt/templates/reflection_message.txt`  
**Change**: Replace all instances of "80-250" with "80-180"

**Lines to update**: 13, 30, 46, 100, 105, 117

#### Fix 3: Fix Installer Fallback Values
**File**: `install-complete.sh:927, 931`  
**Change**:
```bash
# Before:
echo "REFLECTION_WORD_MIN=${TIER_REFLECTION_WORD_MIN:-100}" >> "$be_env"
echo "REFLECTION_WORD_MAX=${TIER_REFLECTION_WORD_MAX:-150}" >> "$be_env"

# After:
echo "REFLECTION_WORD_MIN=${TIER_REFLECTION_WORD_MIN:-90}" >> "$be_env"
echo "REFLECTION_WORD_MAX=${TIER_REFLECTION_WORD_MAX:-180}" >> "$be_env"
```

#### Fix 4: Align Tier Detection Values
**File**: `detect-tier.sh:58`  
**Change**: Update MIN from 90 to 80 for consistency
```bash
# Before:
export TIER_REFLECTION_WORD_MIN=90

# After:
export TIER_REFLECTION_WORD_MIN=80
```

### 8.2 Structural Fixes (Medium Priority)

#### Fix 5: Remove "Soft" Violation Logic
**File**: `reflection/processor.py:2240-2247`  
**Change**: Remove the `"soft": True` flag since it's not actually honored

```python
# Before:
return {
    "compliant": False,
    "violations": ["content_length_out_of_bounds"],
    "soft": True,  # ← Remove this
    "word_count": word_count,
    ...
}

# After:
return {
    "compliant": False,
    "violations": ["content_length_out_of_bounds"],
    "word_count": word_count,
    ...
}
```

**OR** implement actual soft violation handling:
```python
# At line 2545, check for soft violations:
if not (post_check or {}).get("compliant", True):
    violations = set((post_check or {}).get("violations", []) or [])
    is_soft = (post_check or {}).get("soft", False)
    
    # Allow soft violations to pass with warning
    if is_soft and violations == {"content_length_out_of_bounds"}:
        word_count = (post_check or {}).get('word_count', 0)
        logger.warning(
            f"⚠️ Accepting reflection with soft length violation: "
            f"{word_count} words (expected {config_word_min}-{config_word_max})"
        )
        break  # Exit retry loop, accept the reflection
```

#### Fix 6: Add Tolerance Band
**File**: `reflection/processor.py:2234-2247`  
**Change**: Add 5-10 word tolerance for near-misses

```python
TOLERANCE = 5  # Allow 5 words under/over
if word_count < (config_word_min - TOLERANCE) or word_count > (config_word_max + TOLERANCE):
    # Only fail if significantly out of range
    ...
```

#### Fix 7: Increase Token Budget
**File**: `.env` (to be created)  
**Change**: Set explicit token budget to encourage longer output
```bash
REFLECTION_NUM_PREDICT=600  # Up from 480
REFLECTION_MAX_TOKENS=600
```

**Rationale**: More tokens = more room for LLM to generate longer content

#### Fix 8: Increase Temperature Slightly
**File**: `.env`  
**Change**: 
```bash
REFLECTION_TEMPERATURE=0.4  # Up from 0.35
```

**Rationale**: Higher temperature = more verbose, creative output

### 8.3 Alternative Approaches (Low Priority)

#### Option A: Lower Minimum to Match LLM Output
If the LLM consistently generates 60-80 words, consider:
```bash
REFLECTION_WORD_MIN=60
REFLECTION_WORD_MAX=180
```

**Pros**: Accepts current LLM behavior  
**Cons**: Reflections may be too shallow

#### Option B: Use Different Model
Switch to a larger model for reflections:
```bash
REFLECTION_LLM=llama3:8b  # Instead of qwen2.5:3b
```

**Pros**: Larger model may generate longer, richer content  
**Cons**: Slower generation, more VRAM usage

#### Option C: Add More Few-Shot Examples
**File**: `db/repositories/example.py:116-145`  
**Change**: Add 2-3 more positive examples at different word counts (90, 120, 160 words)

**Pros**: Better guidance for LLM across the range  
**Cons**: Larger prompts, more database overhead

---

## 9. Implementation Priority

### Phase 1: Critical (Do First)
1. ✅ Create `.env` file with `REFLECTION_WORD_MIN=80` and `REFLECTION_WORD_MAX=180`
2. ✅ Update `reflection_message.txt` template: change all "80-250" to "80-180"
3. ✅ Fix installer fallback values in `install-complete.sh`

### Phase 2: Important (Do Next)
4. ✅ Implement soft violation handling OR remove soft flag entirely
5. ✅ Add 5-word tolerance band to validation
6. ✅ Increase token budget to 600 tokens

### Phase 3: Optional (Test and Evaluate)
7. ⚠️ Consider temperature increase to 0.4
8. ⚠️ Add more few-shot examples
9. ⚠️ Consider model switch if issues persist

---

## 10. Testing Recommendations

After implementing fixes:

1. **Monitor retry rates**: Should drop to <5% of reflections
2. **Check word count distribution**: Should cluster around 100-150 words
3. **Validate schema compliance**: Should be >95%
4. **Track generation time**: May increase 10-20% with higher token budget
5. **Review reflection quality**: Ensure content is still meaningful

### Test Cases

Create test reflections with:
- Simple greeting (should generate ~80-100 words)
- Complex question (should generate ~120-160 words)
- Emotional share (should generate ~140-180 words)

All should pass validation without retries.

---

## 11. Long-Term Architectural Recommendations

### 11.1 Single Source of Truth
**Problem**: Word counts defined in 6+ places  
**Solution**: Create `ReflectionBounds` dataclass in `config/reflection_config.py`:

```python
@dataclass
class ReflectionBounds:
    base_min: int = 80
    base_max: int = 180
    message_min: int = 80
    message_max: int = 180
    tolerance: int = 5
    
    def get_bounds(self, reflection_type: str) -> tuple[int, int]:
        if reflection_type in {"message", "memory_triggered"}:
            return (self.message_min, self.message_max)
        return (self.base_min, self.base_max)
    
    def is_within_tolerance(self, word_count: int, reflection_type: str) -> bool:
        min_val, max_val = self.get_bounds(reflection_type)
        return (min_val - self.tolerance) <= word_count <= (max_val + self.tolerance)
```

### 11.2 Validation Refactor
**Problem**: Validation happens in 3 places with different logic  
**Solution**: Single validation function with clear soft/hard failure modes

### 11.3 Prompt Template Variables
**Problem**: Hardcoded ranges in templates  
**Solution**: Use template variables:
```
Content must be {{word_min}}-{{word_max}} words total
```

---

## 12. Quick Win: Minimal Changes for Immediate Relief

If you need the system working NOW with minimal changes:

### Step 1: Lower the minimum to 70 words
```bash
# In backend/.env (create if missing):
REFLECTION_WORD_MIN=70
REFLECTION_WORD_MAX=180
```

### Step 2: Add tolerance in validation
```python
# In reflection/processor.py:2234, change:
TOLERANCE = 10
if word_count < (config_word_min - TOLERANCE) or word_count > (config_word_max + TOLERANCE):
```

### Step 3: Honor soft violations
```python
# In reflection/processor.py:2545, add:
is_soft = (post_check or {}).get("soft", False)
if is_soft and violations == {"content_length_out_of_bounds"}:
    logger.warning(f"Accepting soft violation: {word_count} words")
    break  # Accept the reflection
```

This would allow 60-190 word reflections to pass (70±10 to 180±10).

---

## Conclusion

The reflection validation failures stem from **configuration fragmentation** across multiple layers:
- Installer writes one set of values
- Code defaults to another
- Templates specify a third range
- No .env file exists to override anything

The LLM is receiving mixed signals and generating content that falls just below the minimum threshold (78 vs 80 words), triggering hard failures.

**Recommended Path Forward**:
1. Create .env file with consistent 80-180 range
2. Update template to match (remove 250 references)
3. Add 5-word tolerance band
4. Implement proper soft violation handling
5. Fix installer fallback values for future installations

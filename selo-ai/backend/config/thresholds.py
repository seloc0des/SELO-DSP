"""
Centralized configuration for system thresholds and limits.

This module consolidates all hardcoded threshold values used throughout
the SELO AI system to make them configurable and maintainable.
"""

import os
from typing import Dict, Any


class SystemThresholds:
    """Centralized threshold configuration for SELO AI systems."""
    
    # === Persona System Thresholds ===
    
    # Trait evolution limits
    TRAIT_DELTA_MAX: float = float(os.getenv("TRAIT_DELTA_MAX", "0.2"))
    """Maximum change allowed for a single trait update (±0.2)"""
    
    TRAIT_CONFIDENCE_DEFAULT: float = float(os.getenv("TRAIT_CONFIDENCE_DEFAULT", "0.7"))
    """Default confidence score for newly created traits"""
    
    TRAIT_STABILITY_DEFAULT: float = float(os.getenv("TRAIT_STABILITY_DEFAULT", "0.3"))
    """Default stability score for newly created traits"""
    
    # Evolution confidence thresholds
    EVOLUTION_CONFIDENCE_MIN: float = float(os.getenv("EVOLUTION_CONFIDENCE_MIN", "0.6"))
    """Minimum confidence score for persona evolution"""
    
    EVOLUTION_IMPACT_DEFAULT: float = float(os.getenv("EVOLUTION_IMPACT_DEFAULT", "0.5"))
    """Default impact score for persona evolution"""
    
    # === SDL System Thresholds ===
    
    # Learning confidence and quality
    LEARNING_CONFIDENCE_MIN: float = float(os.getenv("LEARNING_CONFIDENCE_MIN", "0.6"))
    """Minimum confidence threshold for accepting learnings"""
    
    LEARNING_IMPORTANCE_MIN: float = float(os.getenv("LEARNING_IMPORTANCE_MIN", "0.5"))
    """Minimum importance threshold for learnings"""
    
    LEARNING_CONSOLIDATION_THRESHOLD: float = float(os.getenv("LEARNING_CONSOLIDATION_THRESHOLD", "0.8"))
    """Confidence threshold for triggering learning consolidation"""
    
    # Similarity thresholds
    LEARNING_SIMILARITY_THRESHOLD: float = float(os.getenv("LEARNING_SIMILARITY_THRESHOLD", "0.82"))
    """Similarity threshold for merging similar learnings"""
    
    CONCEPT_SIMILARITY_THRESHOLD: float = float(os.getenv("CONCEPT_SIMILARITY_THRESHOLD", "0.85"))
    """Similarity threshold for concept matching"""
    
    # Batch sizes and limits
    LEARNING_CONSOLIDATION_BATCH_SIZE: int = int(os.getenv("LEARNING_CONSOLIDATION_BATCH_SIZE", "100"))
    """Maximum number of learnings to fetch for consolidation (was 500, reduced for safety)"""
    
    LEARNING_RETRIEVAL_DEFAULT_LIMIT: int = int(os.getenv("LEARNING_RETRIEVAL_DEFAULT_LIMIT", "50"))
    """Default limit for learning queries"""
    
    # === Agent State Thresholds ===
    
    # Affective state adjustments
    AFFECTIVE_ENERGY_DELTA_MAX: float = float(os.getenv("AFFECTIVE_ENERGY_DELTA_MAX", "0.3"))
    """Maximum energy adjustment per reflection (±0.3)"""
    
    AFFECTIVE_STRESS_DELTA_MAX: float = float(os.getenv("AFFECTIVE_STRESS_DELTA_MAX", "0.4"))
    """Maximum stress adjustment per reflection (±0.4)"""
    
    AFFECTIVE_CONFIDENCE_DELTA_MAX: float = float(os.getenv("AFFECTIVE_CONFIDENCE_DELTA_MAX", "0.3"))
    """Maximum confidence adjustment per reflection (±0.3)"""
    
    # Homeostasis decay
    HOMEOSTASIS_DECAY_FACTOR: float = float(os.getenv("HOMEOSTASIS_DECAY_FACTOR", "0.1"))
    """Decay factor for homeostasis (0.1 = 10% decay toward baseline)"""
    
    # Baseline defaults
    AFFECTIVE_ENERGY_BASELINE: float = float(os.getenv("AFFECTIVE_ENERGY_BASELINE", "0.5"))
    """Default baseline energy level"""
    
    AFFECTIVE_STRESS_BASELINE: float = float(os.getenv("AFFECTIVE_STRESS_BASELINE", "0.4"))
    """Default baseline stress level"""
    
    AFFECTIVE_CONFIDENCE_BASELINE: float = float(os.getenv("AFFECTIVE_CONFIDENCE_BASELINE", "0.6"))
    """Default baseline confidence level"""
    
    # Goal and plan thresholds
    GOAL_SIMILARITY_THRESHOLD: float = float(os.getenv("GOAL_SIMILARITY_THRESHOLD", "0.9"))
    """Similarity threshold for detecting duplicate goals"""
    
    STEP_SIMILARITY_THRESHOLD: float = float(os.getenv("STEP_SIMILARITY_THRESHOLD", "0.9"))
    """Similarity threshold for detecting duplicate plan steps"""
    
    # === Episode System Thresholds ===
    
    EPISODE_NARRATIVE_MIN_WORDS: int = int(os.getenv("EPISODE_NARRATIVE_MIN_WORDS", "120"))
    """Minimum word count for episode narratives"""
    
    EPISODE_NARRATIVE_MAX_WORDS: int = int(os.getenv("EPISODE_NARRATIVE_MAX_WORDS", "360"))
    """Maximum word count for episode narratives"""
    
    EPISODE_IMPORTANCE_DEFAULT: float = float(os.getenv("EPISODE_IMPORTANCE_DEFAULT", "0.6"))
    """Default importance score for episodes"""
    
    # === Retry and Resilience ===
    
    VECTOR_STORE_RETRY_ATTEMPTS: int = int(os.getenv("VECTOR_STORE_RETRY_ATTEMPTS", "3"))
    """Number of retry attempts for vector store operations"""
    
    VECTOR_STORE_RETRY_DELAY: float = float(os.getenv("VECTOR_STORE_RETRY_DELAY", "1.0"))
    """Initial delay between retries (seconds), exponential backoff applied"""
    
    LLM_RETRY_ATTEMPTS: int = int(os.getenv("LLM_RETRY_ATTEMPTS", "2"))
    """Number of retry attempts for LLM calls"""
    
    LLM_RETRY_DELAY: float = float(os.getenv("LLM_RETRY_DELAY", "2.0"))
    """Initial delay between LLM retries (seconds)"""
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export all thresholds as a dictionary for inspection."""
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith("_") and key.isupper()
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all threshold values are within sensible ranges."""
        errors = []
        
        # Validate 0-1 range values
        for attr in [
            "TRAIT_DELTA_MAX", "TRAIT_CONFIDENCE_DEFAULT", "TRAIT_STABILITY_DEFAULT",
            "EVOLUTION_CONFIDENCE_MIN", "EVOLUTION_IMPACT_DEFAULT",
            "LEARNING_CONFIDENCE_MIN", "LEARNING_IMPORTANCE_MIN",
            "LEARNING_CONSOLIDATION_THRESHOLD", "LEARNING_SIMILARITY_THRESHOLD",
            "CONCEPT_SIMILARITY_THRESHOLD", "AFFECTIVE_ENERGY_DELTA_MAX",
            "AFFECTIVE_STRESS_DELTA_MAX", "AFFECTIVE_CONFIDENCE_DELTA_MAX",
            "HOMEOSTASIS_DECAY_FACTOR", "AFFECTIVE_ENERGY_BASELINE",
            "AFFECTIVE_STRESS_BASELINE", "AFFECTIVE_CONFIDENCE_BASELINE",
            "GOAL_SIMILARITY_THRESHOLD", "STEP_SIMILARITY_THRESHOLD",
            "EPISODE_IMPORTANCE_DEFAULT"
        ]:
            value = getattr(cls, attr)
            if not 0.0 <= value <= 1.0:
                errors.append(f"{attr}={value} is outside valid range [0.0, 1.0]")
        
        # Validate positive integers
        for attr in [
            "LEARNING_CONSOLIDATION_BATCH_SIZE", "LEARNING_RETRIEVAL_DEFAULT_LIMIT",
            "EPISODE_NARRATIVE_MIN_WORDS", "EPISODE_NARRATIVE_MAX_WORDS",
            "VECTOR_STORE_RETRY_ATTEMPTS", "LLM_RETRY_ATTEMPTS"
        ]:
            value = getattr(cls, attr)
            if value < 1:
                errors.append(f"{attr}={value} must be >= 1")
        
        # Validate positive floats
        for attr in ["VECTOR_STORE_RETRY_DELAY", "LLM_RETRY_DELAY"]:
            value = getattr(cls, attr)
            if value < 0:
                errors.append(f"{attr}={value} must be >= 0")
        
        # Validate logical constraints
        if cls.EPISODE_NARRATIVE_MIN_WORDS >= cls.EPISODE_NARRATIVE_MAX_WORDS:
            errors.append(
                f"EPISODE_NARRATIVE_MIN_WORDS ({cls.EPISODE_NARRATIVE_MIN_WORDS}) "
                f"must be < EPISODE_NARRATIVE_MAX_WORDS ({cls.EPISODE_NARRATIVE_MAX_WORDS})"
            )
        
        if errors:
            raise ValueError(f"Threshold validation failed:\n" + "\n".join(errors))
        
        return True


# Validate thresholds on module import
SystemThresholds.validate()

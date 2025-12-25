"""
Trait Homeostasis Manager

Manages natural drift toward baseline for trait stability, preventing
traits from getting stuck at extremes (e.g., "Stressed" at 100%).
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger("selo.persona.homeostasis")


class TraitHomeostasisManager:
    """
    Manages homeostatic regulation of persona traits.
    
    Implements multi-layered homeostasis:
    - Distance-based decay (stronger at extremes)
    - Personality-aware baselines
    - Locked trait exemption
    - Adaptive decay rates based on trait stability
    """
    
    def __init__(self, persona_repo):
        """
        Initialize the trait homeostasis manager.
        
        Args:
            persona_repo: PersonaRepository instance
        """
        self.persona_repo = persona_repo
        
        # Default baselines for common traits
        self.DEFAULT_BASELINES = {
            # Stress-related (lower baselines)
            "stressed": 0.3,
            "anxious": 0.25,
            "overwhelmed": 0.2,
            "frustrated": 0.3,
            
            # Positive states (moderate-high baselines)
            "calm": 0.6,
            "confident": 0.65,
            "content": 0.6,
            "peaceful": 0.6,
            
            # Cognitive traits (moderate-high)
            "curious": 0.7,
            "analytical": 0.6,
            "creative": 0.6,
            "focused": 0.6,
            
            # Social traits (moderate-high)
            "empathy": 0.7,
            "helpfulness": 0.75,
            "patience": 0.6,
            "politeness": 0.7,
            
            # Learning traits (moderate-high)
            "adaptability": 0.65,
            "openness": 0.7,
            
            # Default for unknown traits
            "_default": 0.5
        }
    
    async def apply_homeostatic_regulation(
        self,
        persona_id: str,
        decay_factor: float = 0.05,
        baseline_targets: Optional[Dict[str, float]] = None,
        min_change_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Apply gentle drift toward baseline for all traits.
        
        Args:
            persona_id: Persona ID
            decay_factor: Base decay rate (0.0-1.0), default 0.05 = 5% per cycle
            baseline_targets: Optional custom baselines per trait
            min_change_threshold: Minimum change to apply (avoids micro-updates)
        
        Returns:
            Dictionary with regulation results
        """
        try:
            # Get persona with traits
            persona = await self.persona_repo.get_persona(
                persona_id=persona_id,
                include_traits=True
            )
            
            if not persona:
                logger.warning(f"Persona {persona_id} not found for homeostasis")
                return {"skipped": True, "reason": "persona_not_found"}
            
            traits = getattr(persona, "traits", []) or []
            if not traits:
                logger.debug(f"No traits found for persona {persona_id}")
                return {"skipped": True, "reason": "no_traits"}
            
            # Compute personality-aware baselines
            baselines = baseline_targets or self._compute_personality_baselines(persona)
            
            changes = []
            traits_updated = 0
            
            for trait in traits:
                # Skip locked traits
                if getattr(trait, "locked", False):
                    continue
                
                trait_name = getattr(trait, "name", "")
                current_value = float(getattr(trait, "value", 0.5))
                
                # Get baseline for this trait
                baseline = baselines.get(trait_name, self.DEFAULT_BASELINES.get("_default", 0.5))
                
                # Calculate distance from baseline
                distance = abs(current_value - baseline)
                
                # Adaptive decay: stronger at extremes
                # At baseline: decay_factor
                # At extremes (0.0 or 1.0): decay_factor * 1.5
                adaptive_decay = decay_factor * (1.0 + distance * 0.5)
                
                # Apply stability factor (more stable traits decay slower)
                stability = float(getattr(trait, "stability", 0.5))
                stability_modifier = 1.0 - (stability * 0.3)  # Max 30% reduction
                final_decay = adaptive_decay * stability_modifier
                
                # Calculate new value (drift toward baseline)
                delta = (baseline - current_value) * final_decay
                new_value = current_value + delta
                
                # Clamp to valid range
                new_value = max(0.0, min(1.0, new_value))
                
                # Only update if change is meaningful
                if abs(new_value - current_value) >= min_change_threshold:
                    try:
                        await self.persona_repo.update_trait(
                            trait_id=trait.id,
                            trait_data={
                                "value": new_value,
                                "last_updated": datetime.now(timezone.utc)
                            }
                        )
                        
                        changes.append({
                            "name": trait_name,
                            "category": getattr(trait, "category", ""),
                            "old_value": round(current_value, 3),
                            "new_value": round(new_value, 3),
                            "baseline": baseline,
                            "distance_from_baseline": round(abs(new_value - baseline), 3),
                            "decay_applied": round(final_decay, 3)
                        })
                        
                        traits_updated += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to update trait {trait_name}: {e}")
                        continue
            
            result = {
                "success": True,
                "persona_id": persona_id,
                "traits_evaluated": len(traits),
                "traits_updated": traits_updated,
                "changes": changes,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if traits_updated > 0:
                logger.info(
                    f"Homeostasis applied to persona {persona_id}: "
                    f"{traits_updated} traits regulated"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying homeostasis to persona {persona_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "persona_id": persona_id
            }
    
    def _compute_personality_baselines(self, persona) -> Dict[str, float]:
        """
        Compute personality-aware baselines from persona attributes.
        
        Maps personality dimensions to trait baselines, allowing
        different personas to have different "natural" trait levels.
        
        Args:
            persona: Persona object with personality attributes
        
        Returns:
            Dictionary mapping trait names to baseline values
        """
        baselines = self.DEFAULT_BASELINES.copy()
        
        try:
            # Extract personality dimensions
            personality = getattr(persona, "personality", {}) or {}
            
            # Map personality to trait baselines
            if "openness" in personality:
                openness = float(personality.get("openness", 0.7))
                baselines["curious"] = openness
                baselines["creative"] = openness * 0.9
                baselines["adaptability"] = openness * 0.85
            
            if "agreeableness" in personality:
                agreeableness = float(personality.get("agreeableness", 0.7))
                baselines["empathy"] = agreeableness
                baselines["helpfulness"] = agreeableness * 0.95
                baselines["patience"] = agreeableness * 0.85
            
            if "conscientiousness" in personality:
                conscientiousness = float(personality.get("conscientiousness", 0.6))
                baselines["focused"] = conscientiousness
                baselines["analytical"] = conscientiousness * 0.9
            
            if "neuroticism" in personality:
                neuroticism = float(personality.get("neuroticism", 0.3))
                # Higher neuroticism = higher baseline stress/anxiety
                baselines["stressed"] = 0.2 + (neuroticism * 0.3)
                baselines["anxious"] = 0.15 + (neuroticism * 0.35)
                # Lower neuroticism = higher baseline calm
                baselines["calm"] = 0.8 - (neuroticism * 0.4)
            
            if "extraversion" in personality:
                extraversion = float(personality.get("extraversion", 0.5))
                baselines["confident"] = 0.5 + (extraversion * 0.3)
            
        except Exception as e:
            logger.warning(f"Error computing personality baselines: {e}")
        
        return baselines
    
    async def get_trait_health_report(self, persona_id: str) -> Dict[str, Any]:
        """
        Generate a health report for persona traits.
        
        Identifies traits that are:
        - At extremes (>0.9 or <0.1)
        - Far from baseline (>0.3 distance)
        - Rapidly changing (high volatility)
        
        Args:
            persona_id: Persona ID
        
        Returns:
            Health report dictionary
        """
        try:
            persona = await self.persona_repo.get_persona(
                persona_id=persona_id,
                include_traits=True
            )
            
            if not persona:
                return {"error": "Persona not found"}
            
            traits = getattr(persona, "traits", []) or []
            if not traits:
                return {"status": "no_traits"}
            
            baselines = self._compute_personality_baselines(persona)
            
            at_extremes = []
            far_from_baseline = []
            
            for trait in traits:
                name = getattr(trait, "name", "")
                value = float(getattr(trait, "value", 0.5))
                baseline = baselines.get(name, 0.5)
                distance = abs(value - baseline)
                
                # Check for extremes
                if value > 0.9 or value < 0.1:
                    at_extremes.append({
                        "name": name,
                        "value": round(value, 3),
                        "severity": "high" if (value > 0.95 or value < 0.05) else "moderate"
                    })
                
                # Check distance from baseline
                if distance > 0.3:
                    far_from_baseline.append({
                        "name": name,
                        "value": round(value, 3),
                        "baseline": baseline,
                        "distance": round(distance, 3)
                    })
            
            health_status = "healthy"
            if len(at_extremes) > 3 or len(far_from_baseline) > 5:
                health_status = "needs_attention"
            elif len(at_extremes) > 0 or len(far_from_baseline) > 0:
                health_status = "monitoring"
            
            return {
                "persona_id": persona_id,
                "status": health_status,
                "total_traits": len(traits),
                "at_extremes": at_extremes,
                "far_from_baseline": far_from_baseline,
                "recommendations": self._generate_recommendations(
                    at_extremes,
                    far_from_baseline
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating trait health report: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _generate_recommendations(
        self,
        at_extremes: List[Dict],
        far_from_baseline: List[Dict]
    ) -> List[str]:
        """Generate recommendations based on trait health."""
        recommendations = []
        
        if len(at_extremes) > 0:
            recommendations.append(
                f"Apply homeostasis more frequently - {len(at_extremes)} traits at extremes"
            )
        
        if len(far_from_baseline) > 3:
            recommendations.append(
                "Consider adjusting baseline targets or increasing decay factor"
            )
        
        if not recommendations:
            recommendations.append("Trait health is good - continue current homeostasis schedule")
        
        return recommendations

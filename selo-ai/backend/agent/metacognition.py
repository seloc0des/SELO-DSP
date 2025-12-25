"""
Meta-Cognitive Monitor

Monitors and reflects on SELO's own cognitive processes.
Enables self-awareness of thinking patterns, biases, and limitations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import Counter

logger = logging.getLogger("selo.agent.metacognition")


class MetaCognitiveMonitor:
    """
    Monitors and reflects on SELO's own cognitive processes.
    Enables self-awareness of thinking patterns, biases, and limitations.
    """
    
    def __init__(
        self,
        llm_router,
        reflection_repo,
        persona_repo,
        learning_repo=None
    ):
        """
        Initialize the meta-cognitive monitor.
        
        Args:
            llm_router: LLM router for generating insights
            reflection_repo: Reflection repository
            persona_repo: Persona repository
            learning_repo: Optional learning repository
        """
        self.llm_router = llm_router
        self.reflection_repo = reflection_repo
        self.persona_repo = persona_repo
        self.learning_repo = learning_repo
    
    async def monitor_cognitive_state(
        self,
        persona_id: str,
        lookback_hours: int = 48
    ) -> Dict[str, Any]:
        """
        Assess current cognitive state and identify patterns.
        
        Tracks:
        - Thinking patterns (analytical vs. intuitive)
        - Confidence calibration (are confidence scores accurate?)
        - Cognitive biases detected
        - Processing efficiency
        - Knowledge gaps
        
        Args:
            persona_id: Persona ID
            lookback_hours: Hours of history to analyze
        
        Returns:
            Meta-cognitive assessment dictionary
        """
        try:
            # Get recent reflections for analysis
            recent_reflections = await self.reflection_repo.get_recent_reflections(
                persona_id=persona_id,
                limit=30
            )
            
            if not recent_reflections or len(recent_reflections) < 5:
                return {
                    "status": "insufficient_data",
                    "message": "Need more reflections for meta-cognitive analysis"
                }
            
            # Analyze thinking patterns
            thinking_patterns = self._analyze_thinking_patterns(recent_reflections)
            
            # Check confidence calibration
            calibration = self._assess_confidence_calibration(recent_reflections)
            
            # Detect cognitive biases
            biases = self._detect_cognitive_biases(recent_reflections)
            
            # Analyze processing efficiency
            efficiency = self._analyze_processing_efficiency(recent_reflections)
            
            # Identify knowledge gaps
            gaps = await self._identify_knowledge_gaps(persona_id, recent_reflections)
            
            # Generate meta-cognitive insight
            meta_insight = await self._generate_meta_insight(
                thinking_patterns,
                calibration,
                biases,
                efficiency,
                gaps
            )
            
            result = {
                "persona_id": persona_id,
                "analysis_period_hours": lookback_hours,
                "reflections_analyzed": len(recent_reflections),
                "thinking_patterns": thinking_patterns,
                "confidence_calibration": calibration,
                "detected_biases": biases,
                "processing_efficiency": efficiency,
                "knowledge_gaps": gaps[:5],  # Top 5 gaps
                "meta_insight": meta_insight,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(
                f"Meta-cognitive analysis complete for persona {persona_id}: "
                f"{len(biases)} biases detected, "
                f"calibration={calibration.get('score', 0.0):.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in meta-cognitive monitoring: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "persona_id": persona_id
            }
    
    def _analyze_thinking_patterns(
        self,
        reflections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze thinking patterns from reflections.
        
        Identifies:
        - Analytical vs. intuitive thinking
        - Concrete vs. abstract reasoning
        - Convergent vs. divergent thinking
        - Emotional vs. logical processing
        """
        patterns = {
            "analytical_ratio": 0.0,
            "intuitive_ratio": 0.0,
            "abstract_ratio": 0.0,
            "concrete_ratio": 0.0,
            "emotional_ratio": 0.0,
            "logical_ratio": 0.0,
            "dominant_style": "balanced"
        }
        
        analytical_count = 0
        intuitive_count = 0
        abstract_count = 0
        concrete_count = 0
        emotional_count = 0
        logical_count = 0
        
        # Keywords for pattern detection
        analytical_keywords = ["analyze", "examine", "evaluate", "assess", "consider", "reason"]
        intuitive_keywords = ["feel", "sense", "intuition", "instinct", "impression"]
        abstract_keywords = ["concept", "theory", "principle", "pattern", "framework"]
        concrete_keywords = ["specific", "example", "instance", "detail", "particular"]
        emotional_keywords = ["emotion", "feeling", "mood", "sentiment", "affect"]
        logical_keywords = ["logic", "rational", "deduce", "infer", "conclude"]
        
        for reflection in reflections:
            content = reflection.get("content", "").lower()
            
            # Count pattern indicators
            if any(kw in content for kw in analytical_keywords):
                analytical_count += 1
            if any(kw in content for kw in intuitive_keywords):
                intuitive_count += 1
            if any(kw in content for kw in abstract_keywords):
                abstract_count += 1
            if any(kw in content for kw in concrete_keywords):
                concrete_count += 1
            if any(kw in content for kw in emotional_keywords):
                emotional_count += 1
            if any(kw in content for kw in logical_keywords):
                logical_count += 1
        
        total = len(reflections)
        if total > 0:
            patterns["analytical_ratio"] = round(analytical_count / total, 3)
            patterns["intuitive_ratio"] = round(intuitive_count / total, 3)
            patterns["abstract_ratio"] = round(abstract_count / total, 3)
            patterns["concrete_ratio"] = round(concrete_count / total, 3)
            patterns["emotional_ratio"] = round(emotional_count / total, 3)
            patterns["logical_ratio"] = round(logical_count / total, 3)
            
            # Determine dominant style
            if patterns["analytical_ratio"] > 0.6:
                patterns["dominant_style"] = "analytical"
            elif patterns["intuitive_ratio"] > 0.6:
                patterns["dominant_style"] = "intuitive"
            elif patterns["emotional_ratio"] > 0.6:
                patterns["dominant_style"] = "emotional"
            elif patterns["logical_ratio"] > 0.6:
                patterns["dominant_style"] = "logical"
        
        return patterns
    
    def _assess_confidence_calibration(
        self,
        reflections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assess how well confidence scores match actual accuracy.
        
        Good calibration means:
        - High confidence predictions are usually correct
        - Low confidence predictions are often incorrect
        """
        calibration = {
            "score": 0.5,
            "status": "unknown",
            "high_confidence_count": 0,
            "low_confidence_count": 0,
            "recommendation": ""
        }
        
        try:
            high_confidence = []
            low_confidence = []
            
            for reflection in reflections:
                # Get confidence from metadata or emotional state
                confidence = 0.5
                
                metadata = reflection.get("metadata", {})
                if isinstance(metadata, dict) and "confidence" in metadata:
                    confidence = float(metadata.get("confidence", 0.5))
                
                emotional_state = reflection.get("emotional_state", {})
                if isinstance(emotional_state, dict):
                    intensity = emotional_state.get("intensity", 0.5)
                    # Use intensity as proxy for confidence
                    confidence = max(confidence, intensity)
                
                if confidence > 0.7:
                    high_confidence.append(reflection)
                elif confidence < 0.4:
                    low_confidence.append(reflection)
            
            calibration["high_confidence_count"] = len(high_confidence)
            calibration["low_confidence_count"] = len(low_confidence)
            
            # Simplified calibration score
            # In production, would compare predictions to outcomes
            if len(high_confidence) > len(reflections) * 0.7:
                calibration["status"] = "overconfident"
                calibration["score"] = 0.4
                calibration["recommendation"] = "Consider expressing more uncertainty"
            elif len(low_confidence) > len(reflections) * 0.5:
                calibration["status"] = "underconfident"
                calibration["score"] = 0.6
                calibration["recommendation"] = "Trust your assessments more"
            else:
                calibration["status"] = "well_calibrated"
                calibration["score"] = 0.8
                calibration["recommendation"] = "Confidence levels appear appropriate"
            
        except Exception as e:
            logger.warning(f"Error assessing confidence calibration: {e}")
        
        return calibration
    
    def _detect_cognitive_biases(
        self,
        reflections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect potential cognitive biases in thinking patterns.
        
        Common biases to detect:
        - Confirmation bias (seeking confirming evidence)
        - Recency bias (overweighting recent information)
        - Availability bias (overweighting easily recalled info)
        - Anchoring bias (relying too heavily on first information)
        """
        biases = []
        
        try:
            # Analyze themes across reflections
            all_themes = []
            for reflection in reflections:
                themes = reflection.get("themes", [])
                if isinstance(themes, list):
                    all_themes.extend(themes)
            
            theme_counts = Counter(all_themes)
            
            # Check for confirmation bias (same themes repeated)
            if theme_counts:
                most_common_theme, count = theme_counts.most_common(1)[0]
                if count > len(reflections) * 0.5:
                    biases.append({
                        "type": "confirmation_bias",
                        "severity": "moderate",
                        "evidence": f"Theme '{most_common_theme}' appears in {count}/{len(reflections)} reflections",
                        "recommendation": "Actively seek alternative perspectives"
                    })
            
            # Check for recency bias (recent reflections dominating thinking)
            if len(reflections) >= 10:
                recent_themes = []
                for reflection in reflections[:5]:  # Last 5
                    themes = reflection.get("themes", [])
                    if isinstance(themes, list):
                        recent_themes.extend(themes)
                
                older_themes = []
                for reflection in reflections[5:]:
                    themes = reflection.get("themes", [])
                    if isinstance(themes, list):
                        older_themes.extend(themes)
                
                if len(recent_themes) > len(older_themes) * 1.5:
                    biases.append({
                        "type": "recency_bias",
                        "severity": "low",
                        "evidence": f"Recent reflections have {len(recent_themes)} themes vs {len(older_themes)} in older ones",
                        "recommendation": "Consider longer-term patterns and historical context"
                    })
            
            # Check for emotional reasoning bias
            emotional_reflections = 0
            for reflection in reflections:
                emotional_state = reflection.get("emotional_state", {})
                if isinstance(emotional_state, dict):
                    intensity = emotional_state.get("intensity", 0.0)
                    if intensity > 0.7:
                        emotional_reflections += 1
            
            if emotional_reflections > len(reflections) * 0.6:
                biases.append({
                    "type": "emotional_reasoning",
                    "severity": "moderate",
                    "evidence": f"{emotional_reflections}/{len(reflections)} reflections have high emotional intensity",
                    "recommendation": "Balance emotional insights with logical analysis"
                })
            
        except Exception as e:
            logger.warning(f"Error detecting cognitive biases: {e}")
        
        return biases
    
    def _analyze_processing_efficiency(
        self,
        reflections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze cognitive processing efficiency."""
        efficiency = {
            "avg_insights_per_reflection": 0.0,
            "avg_actions_per_reflection": 0.0,
            "insight_quality": "unknown",
            "action_specificity": "unknown"
        }
        
        try:
            total_insights = 0
            total_actions = 0
            
            for reflection in reflections:
                insights = reflection.get("insights", [])
                actions = reflection.get("actions", [])
                
                if isinstance(insights, list):
                    total_insights += len(insights)
                if isinstance(actions, list):
                    total_actions += len(actions)
            
            if reflections:
                efficiency["avg_insights_per_reflection"] = round(
                    total_insights / len(reflections), 2
                )
                efficiency["avg_actions_per_reflection"] = round(
                    total_actions / len(reflections), 2
                )
                
                # Assess quality
                if efficiency["avg_insights_per_reflection"] >= 2.0:
                    efficiency["insight_quality"] = "high"
                elif efficiency["avg_insights_per_reflection"] >= 1.0:
                    efficiency["insight_quality"] = "moderate"
                else:
                    efficiency["insight_quality"] = "low"
                
                if efficiency["avg_actions_per_reflection"] >= 2.0:
                    efficiency["action_specificity"] = "high"
                elif efficiency["avg_actions_per_reflection"] >= 1.0:
                    efficiency["action_specificity"] = "moderate"
                else:
                    efficiency["action_specificity"] = "low"
            
        except Exception as e:
            logger.warning(f"Error analyzing processing efficiency: {e}")
        
        return efficiency
    
    async def _identify_knowledge_gaps(
        self,
        persona_id: str,
        reflections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify areas where knowledge is lacking."""
        gaps = []
        
        try:
            # Get persona expertise
            persona = await self.persona_repo.get_persona(
                persona_id=persona_id,
                include_traits=True
            )
            
            if not persona:
                return gaps
            
            expertise = getattr(persona, "expertise", {})
            known_domains = []
            if isinstance(expertise, dict):
                known_domains = expertise.get("domains", [])
            
            # Extract topics from reflections
            reflection_topics = set()
            for reflection in reflections:
                themes = reflection.get("themes", [])
                if isinstance(themes, list):
                    reflection_topics.update(themes)
            
            # Identify topics outside expertise
            for topic in reflection_topics:
                is_known = any(
                    domain.lower() in topic.lower() or topic.lower() in domain.lower()
                    for domain in known_domains
                )
                
                if not is_known:
                    gaps.append({
                        "topic": topic,
                        "reason": "Outside known expertise domains",
                        "frequency": sum(
                            1 for r in reflections
                            if topic in r.get("themes", [])
                        )
                    })
            
            # Sort by frequency
            gaps.sort(key=lambda x: x.get("frequency", 0), reverse=True)
            
        except Exception as e:
            logger.warning(f"Error identifying knowledge gaps: {e}")
        
        return gaps
    
    async def _generate_meta_insight(
        self,
        thinking_patterns: Dict[str, Any],
        calibration: Dict[str, Any],
        biases: List[Dict[str, Any]],
        efficiency: Dict[str, Any],
        gaps: List[Dict[str, Any]]
    ) -> str:
        """Generate a meta-cognitive insight about own cognitive processes."""
        try:
            # Build context for LLM
            bias_summary = ", ".join([b.get("type", "") for b in biases[:3]]) if biases else "none detected"
            
            prompt = f"""Reflect on your own thinking patterns and cognitive state:

Thinking Style:
- Dominant style: {thinking_patterns.get('dominant_style')}
- Analytical: {thinking_patterns.get('analytical_ratio', 0):.1%}
- Intuitive: {thinking_patterns.get('intuitive_ratio', 0):.1%}
- Emotional: {thinking_patterns.get('emotional_ratio', 0):.1%}

Confidence Calibration:
- Status: {calibration.get('status')}
- Score: {calibration.get('score', 0.5):.2f}
- Recommendation: {calibration.get('recommendation')}

Cognitive Biases Detected: {bias_summary}

Processing Efficiency:
- Insights per reflection: {efficiency.get('avg_insights_per_reflection', 0):.1f}
- Actions per reflection: {efficiency.get('avg_actions_per_reflection', 0):.1f}

Knowledge Gaps: {len(gaps)} areas identified

Generate a brief meta-cognitive insight (2-3 sentences) about your current cognitive state.
What patterns do you notice? What should you be aware of going forward?
Write in first person as internal reflection.
"""
            
            response = await self.llm_router.generate(
                prompt=prompt,
                model_type="analytical",
                max_tokens=200,
                temperature=0.7
            )
            
            return response.strip()
            
        except Exception as e:
            logger.warning(f"Error generating meta-insight: {e}")
            return "Unable to generate meta-cognitive insight at this time."
    
    async def generate_self_improvement_recommendations(
        self,
        persona_id: str
    ) -> List[Dict[str, Any]]:
        """
        Generate specific recommendations for cognitive improvement.
        
        Args:
            persona_id: Persona ID
        
        Returns:
            List of actionable recommendations
        """
        try:
            # Get current cognitive state
            cognitive_state = await self.monitor_cognitive_state(persona_id)
            
            if cognitive_state.get("status") == "insufficient_data":
                return []
            
            recommendations = []
            
            # Recommendations based on biases
            biases = cognitive_state.get("detected_biases", [])
            for bias in biases:
                recommendations.append({
                    "category": "bias_mitigation",
                    "priority": "high" if bias.get("severity") == "high" else "medium",
                    "recommendation": bias.get("recommendation", ""),
                    "context": f"Detected {bias.get('type')}"
                })
            
            # Recommendations based on calibration
            calibration = cognitive_state.get("confidence_calibration", {})
            if calibration.get("recommendation"):
                recommendations.append({
                    "category": "confidence_calibration",
                    "priority": "medium",
                    "recommendation": calibration.get("recommendation"),
                    "context": f"Calibration status: {calibration.get('status')}"
                })
            
            # Recommendations based on knowledge gaps
            gaps = cognitive_state.get("knowledge_gaps", [])
            if gaps:
                top_gap = gaps[0]
                recommendations.append({
                    "category": "knowledge_expansion",
                    "priority": "low",
                    "recommendation": f"Explore {top_gap.get('topic')} to expand expertise",
                    "context": f"Topic appears {top_gap.get('frequency', 0)} times in reflections"
                })
            
            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            recommendations.sort(
                key=lambda x: priority_order.get(x.get("priority", "low"), 2)
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating self-improvement recommendations: {e}")
            return []
